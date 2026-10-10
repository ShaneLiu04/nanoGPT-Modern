# ARCHITECTURE.md — 训练基座设计文档

> AR-v0.5.0-hpc-readiness FR14 交付物。本文档描述训练基座（`training/trainer_base.py`）的模板方法链、
> 分布式/精度/检查点机制、监控降级哲学，以及五个训练器的差异矩阵。
> 目标读者：新人或未来的自己，按本文档对照代码应能准确理解框架（FR14-AC1：描述零漂移）。
>
> 相关文档：[MIGRATION.md](MIGRATION.md)（云租用 GPU 迁移 Runbook）·
> [GRPO_NOTES.md](GRPO_NOTES.md)（RLHF 家族对比笔记）·
> [TECH_REPORT.md](TECH_REPORT.md)（系统全景白皮书）· [adr/](adr/README.md)（架构决策记录）

## 1. BaseTrainer 模板方法链（`training/trainer_base.py:455`）

`BaseTrainer` 是全部五个训练器的公共基类。`__init__` 完成一次性编排，
子类**只覆写受保护钩子**，不重写 `__init__`——这保证数据/模型/优化器/调度器/AMP/检查点/
日志/断点续跑永远按正确顺序构建。

### 1.1 生命周期（构建顺序）

```
__init__(args)
  ├─ setup_distributed(backend)     # rank/local_rank/world_size；torchrun env 或单进程
  ├─ infer_device(args.device)      # cuda:<local_rank> / cpu / mps
  ├─ set_seed(args.seed)            # 所有 rank 同种子 → DDP/FSDP 初始化一致；
  │                                 #   数据乱序由 DistributedSampler / worker 种子提供
  └─ 模板方法链（子类按需覆写，全部可选）：
      ├─ _init_state()              # 子类自有状态（start_epoch/global_step/...）
      ├─ _build_data()              # 数据集与 DataLoader
      ├─ _build_model()             # 构建或加载模型（含 wrap_distributed）
      ├─ _build_optimizer()         # 优化器
      ├─ _build_scheduler()         # LR 调度器
      ├─ _setup_amp()               # 混合精度上下文 + GradScaler（默认走 setup_amp）
      ├─ _setup_checkpointing()     # 创建 CheckpointManager
      ├─ _setup_logger()            # 创建 Logger（仅 master）
      ├─ _maybe_resume()            # --resume 时加载检查点
      ├─ _setup_recorder()          # ExperimentRecorder 结构化指标（仅 master）
      └─ _setup_monitors()          # loss-spike / grad-norm 监控（仅 master）
```

训练循环本身由子类实现抽象方法 `train()`；每步结束调用 `on_step_end(step, loss, ...)`。
进程退出前调用 `cleanup()`：先 flush 异步检查点写盘，再把仍在 running 状态的 recorder
标记为 `interrupted`（train() 未正常 finalize 的证据），关闭 logger，销毁进程组。

### 1.2 on_step_end 监控管线（`trainer_base.py:606`）

每条日志步按固定顺序执行（design §4.2 flow B）：

1. **grad-norm 获取**：调用方未传 `grad_norm` 时由 `GradientNormMonitor.compute()` 现算；
2. **spike 检测**：`LossSpikeDetector.update()` 判定 loss 尖峰；命中时可返回 `lr_multiplier`，
   对当前 LR 做临时乘法衰减（下一次 `scheduler.step()` 会覆盖回计划值），并记录 `loss_spike` 事件；
3. **结构化记录**：`recorder.record_step()` 写入 loss/lr/grad_norm/tokens_per_sec/显存。

任何一步失败都降级为 `RuntimeWarning`，**记录与训练永不中断**。

## 2. 分布式机制：DDP / FSDP（`wrap_distributed`，`trainer_base.py:697`）

统一入口 `wrap_distributed(model, compile_model=...)`，三条路径：

| 模式 | 条件 | 关键参数 | raw_model |
|---|---|---|---|
| FSDP | `--fsdp` 且 distributed | FR9b：`transformer_auto_wrap_policy` 按 Block 逐层分片、`use_orig_params=True`（保留参数名、兼容 torch.compile）、`MixedPrecision`（参数 bf16/fp16、reduce fp32）、sharding 可选 full/grad/no | `model.module` |
| DDP | distributed 且未开 fsdp | FR9a：`static_graph=True`（固定结构启用图分析）+ `gradient_as_bucket_view=True`（梯度别名通信桶，省一份全尺寸梯度缓冲） | `model.module` |
| 单进程 | 其余 | 直接（可选 compile） | model 本身 |

**组合顺序（FR9b）**：`FSDP` 先包裹、`torch.compile` 应用于 FSDP 模块之后（外层）。
FSDP1 没有 `_orig_mod` 解包语义，compile-then-FSDP 会破坏 `.module` 链。

**检查点的 FSDP 感知**：`_get_model_state_dict` / `_load_model_state_dict` 在 FSDP 下用
`FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` 收集完整 state_dict 并剥掉
`_fsdp_wrapped_module.` 前缀；非 FSDP 路径剥掉 torch.compile 的 `_orig_mod.` 前缀。

## 3. AMP 混合精度

`setup_amp(use_bf16=True)` 按设备能力选择：CUDA 且支持 bf16 → `torch.autocast(bfloat16)`
无需 scaler；否则 fp16 + `GradScaler`；CPU → 空上下文。训练循环统一写法
`with self.ctx: ...` + `self.scaler.scale(loss).backward()`（scaler 为 None 时透明直通）。

## 4. Checkpoint 机制（`CheckpointManager`，`trainer_base.py:231`）

- **统一负载**：model / optimizer / scheduler / scaler / config / `resume_offset` /
  `ema_shadow` / `keep_last_n`（滚动保留）。
- **异步写**：save 走后台线程，`cleanup()` 里 `flush()` 保证进程退出前落盘。
- **训练器门面**：`configure_checkpointing(config, ema_shadow=...)` 注入 FSDP 感知的
  state_dict 读写函数；`save_checkpoint/load_checkpoint` 薄封装。
- **EMA**：`ema_shadow` 由 CheckpointManager 持有；ModernGPT 侧提供
  `init_ema / update_ema / apply_ema_weights / restore_ema_weights`（评估时换入 EMA 权重，
  评估后还原）。

## 5. 监控降级哲学

基座的原则：**辅助设施永远不能阻塞训练**。所有可选组件遵循同一模式：

```
构造失败 → warnings.warn(...) + 置 None → 继续训练
```

- `ExperimentRecorder`（结构化指标 → `<out_dir>/experiments/<exp_id>/`）：失败降级；
  `--no_metrics` 显式关闭。
- `LossSpikeDetector` / `GradientNormMonitor`：失败降级；`--no_monitoring` 显式关闭。
- Logger：仅 master 进程创建；`--use_wandb` 可选，TensorBoard 恒开。
- RL 训练的 dropout 偏差防护：`maybe_warn_dropout` 对 dropout>0 的模型在使用
  logprob 比值的训练（GRPO/DPO）前给出警告（ADR-002 关联不变量）。

## 6. 五个训练器差异矩阵

| | train_pretrain | train_sft | train_grpo | train_dpo | iterative_grpo |
|---|---|---|---|---|---|
| 基类 | BaseTrainer | BaseTrainer | BaseTrainer | BaseTrainer | **GRPOTrainer** |
| 数据 | memmap token 流（prepare.py 产物） | 指令/对话模板数据 | 算术题 prompt + 规则奖励 | 偏好对（synthetic / grpo 源） | 同 GRPO + 拒绝采样池 |
| 损失 | next-token CE | next-token CE（mask prompt） | clipped surrogate + KL(ref‖policy) | DPO/IPO/KTO（β 温控） | 同 GRPO + 周期拒绝采样 SFT |
| 参考模型 | — | — | 冻结 ref（eval + no_grad） | 冻结 ref_model | **周期性更新**（hard/EMA） |
| 优势来源 | — | — | 组内相对奖励（group/batch/none 归一） | 隐式（chosen-rejected 对比） | 同 GRPO |
| 钩子覆写 | 全链 | 全链 | `_build_model`（policy+ref）、`_maybe_resume` | `_build_data`（偏好数据源） | `_build_data`、`on_rollout_end`（ref 更新）、`on_eval_end`（best policy 快照 + 拒绝采样 SFT）、`finalize` |
| Hydra 入口 | train_pretrain_hydra | train_sft_hydra | train_grpo_hydra | — | iterative_grpo_hydra |

`*_hydra.py` 均为薄入口 shim（`@hydra.main` + import 训练器），不含算法逻辑；
迭代扩展通过 `on_rollout_end` / `on_eval_end` 钩子实现（模板方法 + 钩子模式）。
GRPO→DPO 的衔接：`train_dpo.py` 的 `--preference_source=grpo` 从 GRPO checkpoint
构建偏好对（高奖励 vs 低奖励响应），即「GRPO 探索 + DPO 精炼」组合策略（ADR-002）。

## 7. 公共 flag 与 resume

- 公共训练 flag 由 `training/args.py` 的 `add_common_trainer_args(parser)` 收敛
  （FR11a，20 个跨脚本一致的 flag；差异 default 的留在各脚本）。
- resume 语义：`--resume <ckpt>` → `_maybe_resume()` 恢复 model/optimizer/scheduler/
  scaler/global_step；GRPO 额外恢复 ref 模型。
