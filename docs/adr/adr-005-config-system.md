# ADR-005: 采用 YAML + argparse + 可选 Pydantic 的配置系统

**状态**: Accepted  
**日期**: 2026-06  
**作者**: nanoGPT-Modern Team

---

## 背景

项目需要一套配置系统管理训练/推理的超参数。要求：
1. 向后兼容（现有 CLI 脚本不变）
2. 支持嵌套覆盖（`--optimizer.lr 1e-4`）
3. 支持配置组合（实验场景需要不同配置叠加）
4. 运行时校验（防止 `n_head % n_kv_head != 0` 等错误在训练中途才暴露）

## 决策

采用 **三层配置架构**：
1. **基础层**: YAML + argparse（默认，向后兼容，v0.1.0 沿用）
2. **组合层**: Hydra/OmegaConf（可选，复杂实验配置组合）
3. **校验层**: Pydantic BaseModel（可选增强，运行时结构化校验）

## 考虑方案

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| 纯 YAML + argparse | 简单，向后兼容 | 无校验，嵌套覆盖需手动实现 | 保留为基础层 |
| **纯 Pydantic** | 强类型校验，IDE 友好 | 需重写所有 CLI 入口，学习曲线 | 作为可选增强层 |
| **纯 Hydra** | 配置组合强大 | 需要理解 config group，与 argparse 冲突 | 作为可选组合层 |
| YAML + argparse + Pydantic + Hydra | 灵活分层，各司其职 | 维护成本略增 | **采纳** |

## 分层设计

```
CLI args -> argparse Namespace
    -> YAML defaults (load_yaml_config)
    -> NestedNamespace (嵌套访问)
    -> [可选] Pydantic validate_with_pydantic() (校验)
    -> [可选] Hydra compose (config group 叠加)
    -> 最终 args 对象
```

- Pydantic 层默认 **best-effort** 模式：`maybe_validate_pydantic()` 在 Pydantic 未安装时静默跳过，不影响原有流程。
- 强制校验模式：`validate_with_pydantic()` 在实验脚本中显式调用，校验失败时抛出 `ValueError`。

## 后果

- **正向**: 现有 `training/train_*.py` 脚本无需修改；新实验可通过 Pydantic 提前捕获配置错误；复杂实验可通过 Hydra 配置组组合。
- **负向**: 新增 `utils/config_models.py` 约 200 行 Pydantic 模型，需与代码变更同步维护。
- **风险**: 低。Pydantic 校验层是可选的，未安装时完全不影响运行。

## 参考

- `utils/config.py`
- `utils/config_models.py`
- `config/hydra/` 配置组目录

---

*最后更新: 2026-06-30*
