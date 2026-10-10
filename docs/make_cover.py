# -*- coding: utf-8 -*-
"""生成 nanoGPT-Modern 封面 docs/assets/cover.png（2400x760，matplotlib 绘制，无外部素材）。

用法：
    python docs/make_cover.py                # 默认输出 docs/assets/cover.png
    python docs/make_cover.py <out.png>      # 指定输出路径
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

DEFAULT_OUT = Path(__file__).resolve().parent / "assets" / "cover.png"
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUT
OUT.parent.mkdir(parents=True, exist_ok=True)

zh = fm.FontProperties(family="Microsoft YaHei")
zh_bold = fm.FontProperties(family="Microsoft YaHei", weight="bold")

W, H = 24.0, 7.6
fig = plt.figure(figsize=(W, H), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# ---------------------------------------------------------------- background
grad = np.linspace(0, 1, 512).reshape(-1, 1)
c_top, c_mid, c_bot = np.array([6, 11, 24]), np.array([11, 18, 48]), np.array([16, 16, 58])
img = np.zeros((512, 2, 3))
for i, t in enumerate(np.linspace(0, 1, 512)):
    img[i, :, :] = (
        c_top + (c_mid - c_top) * (t / 0.55)
        if t < 0.55
        else c_mid + (c_bot - c_mid) * ((t - 0.55) / 0.45)
    )
ax.imshow(img / 255.0, extent=[-0.25, W + 0.25, -0.25, H + 0.25], aspect="auto", zorder=-10, origin="upper")

for x in np.arange(0, W + 0.6, 0.6):
    ax.plot([x, x], [0, H], color="#8B9DC9", lw=0.5, alpha=0.05, zorder=-9)
for y in np.arange(0, H + 0.6, 0.6):
    ax.plot([0, W], [y, y], color="#8B9DC9", lw=0.5, alpha=0.05, zorder=-9)

for cx, cy, r, color, a in [
    (3.4, 6.9, 6.0, "#22D3EE", 0.10),
    (20.8, 6.9, 6.4, "#8B5CF6", 0.12),
    (17.6, 0.1, 5.6, "#EC4899", 0.08),
]:
    ax.add_patch(plt.Circle((cx, cy), r, color=color, alpha=a, zorder=-8))

# left sidebar gradient bar (cyan -> violet -> pink)
c1, c2, c3 = np.array([0.13, 0.83, 0.93]), np.array([0.55, 0.36, 0.96]), np.array([0.93, 0.29, 0.60])
n_seg = 100
for i in range(n_seg):
    t = i / (n_seg - 1)
    c = c1 + (c2 - c1) * (t / 0.5) if t < 0.5 else c2 + (c3 - c2) * ((t - 0.5) / 0.5)
    ax.add_patch(Rectangle((0, H - (i + 1) * H / n_seg), 0.09, H / n_seg + 0.02, color=c, zorder=5))

# ---------------------------------------------------------------- left panel: dual-track
px0, py0, pw, ph = 0.78, 0.88, 8.55, 5.90
ax.add_patch(
    FancyBboxPatch(
        (px0, py0), pw, ph,
        boxstyle="round,pad=0.10,rounding_size=0.22",
        fc="#0B1329", ec="#26375F", lw=1.4, alpha=0.94, zorder=2,
    )
)
ax.text(px0 + 0.42, py0 + ph - 0.52, "双轨对照实验", fontsize=19, color="#E2E8F0",
        fontproperties=zh_bold, zorder=4)
ax.text(px0 + 0.44, py0 + ph - 0.98, "两套架构，同一份数据 / 种子 / 超参",
        fontsize=11.5, color="#8EA3C8", fontproperties=zh, zorder=4)

def track_card(y, name, name_color, chips, accent):
    ch = 1.42
    ax.add_patch(
        FancyBboxPatch(
            (px0 + 0.40, y), pw - 0.80, ch,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            fc="#0E1730", ec="#2A3B66", lw=1.1, zorder=3,
        )
    )
    ax.add_patch(Rectangle((px0 + 0.40, y + 0.12), 0.055, ch - 0.24, color=accent, zorder=4))
    ax.text(px0 + 0.68, y + ch / 2, name, fontsize=15.5, color=name_color,
            family="DejaVu Sans", fontweight="bold", zorder=4)
    cx = px0 + 2.50
    for label in chips:
        cw = 0.28 + 0.205 * len(label)
        ax.add_patch(
            FancyBboxPatch(
                (cx, y + ch / 2 - 0.30), cw, 0.60,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                fc="#101B36", ec="#31446F", lw=1.0, zorder=4,
            )
        )
        ax.text(cx + cw / 2, y + ch / 2, label, fontsize=10.2, color="#C7D2E8",
                ha="center", va="center", fontproperties=zh, zorder=5)
        cx += cw + 0.14

track_card(py0 + 3.05, "BaselineGPT", "#94A3B8",
           ["LayerNorm", "GELU", "绝对位置编码"], "#64748B")
ax.text(px0 + pw / 2 + 0.30, py0 + 2.72, "vs", fontsize=12.5, color="#4B5C82",
        ha="center", family="DejaVu Sans", fontweight="bold", zorder=4)
track_card(py0 + 0.95, "ModernGPT", "#7DD3FC",
           ["RMSNorm", "SwiGLU", "RoPE", "GQA"], "#22D3EE")

# ---------------------------------------------------------------- right: title block
tx = 10.35

ax.add_patch(
    FancyBboxPatch(
        (tx, 6.30), 6.45, 0.62,
        boxstyle="round,pad=0.06,rounding_size=0.16",
        fc="#12203F", ec="#3B82F6", lw=1.3, zorder=3,
    )
)
ax.text(tx + 0.26, 6.61, "LIGHTWEIGHT LLM FULL-STACK FRAMEWORK", fontsize=12.5,
        color="#7DD3FC", family="DejaVu Sans", fontweight="bold", zorder=4,
        va="center")

title_glow = [pe.withStroke(linewidth=7, foreground=(0.13, 0.83, 0.93, 0.16))]
t1 = ax.text(tx - 0.05, 5.02, "nanoGPT", fontsize=64, color="#F4F7FE",
             family="DejaVu Sans", fontweight="bold", zorder=4, va="center",
             path_effects=title_glow)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
x_end = ax.transData.inverted().transform((t1.get_window_extent(renderer).x1, 0))[0]
ax.text(x_end + 0.14, 5.02, "-Modern", fontsize=64, color="#38BDF8",
        family="DejaVu Sans", fontweight="bold", zorder=4, va="center",
        path_effects=title_glow)

ax.add_patch(
    FancyBboxPatch(
        (22.75, 6.38), 0.95, 0.48,
        boxstyle="round,pad=0.05,rounding_size=0.14",
        fc="#1A1038", ec="#8B5CF6", lw=1.2, zorder=3,
    )
)
ax.text(23.22, 6.62, "v0.3.0", fontsize=12.5, color="#C4B5FD",
        family="DejaVu Sans", fontweight="bold", ha="center", va="center", zorder=4)

ax.text(tx - 0.02, 3.98, "轻量级大语言模型全栈框架 · 双轨现代架构对照实验", fontsize=22,
        color="#C7D2E8", fontproperties=zh, zorder=4, va="center")
ax.text(tx - 0.02, 3.42, "预训练 → SFT → GRPO / DPO → 高效推理 → 评估观测 · 全生命周期覆盖",
        fontsize=14.5, color="#7C8DB5", fontproperties=zh, zorder=4, va="center")

# pipeline chips
pipe = ["预训练", "SFT", "GRPO / DPO", "高效推理", "评估观测"]
pcolors = ["#22D3EE", "#38BDF8", "#818CF8", "#A78BFA", "#E879F9"]
cxx = tx + 0.02
for i, (label, col) in enumerate(zip(pipe, pcolors)):
    cw = 0.42 + 0.29 * len(label)
    ax.add_patch(
        FancyBboxPatch(
            (cxx, 2.42), cw, 0.64,
            boxstyle="round,pad=0.05,rounding_size=0.14",
            fc="#0E1B33", ec=col, lw=1.2, alpha=0.95, zorder=3,
        )
    )
    ax.text(cxx + cw / 2, 2.74, label, fontsize=12.5, color="#DCE6F7",
            ha="center", va="center", fontproperties=zh, zorder=4)
    cxx += cw
    if i < len(pipe) - 1:
        ax.text(cxx + 0.17, 2.74, "→", fontsize=13.5, color="#4C5F88",
                ha="center", va="center", family="DejaVu Sans", zorder=4)
        cxx += 0.38

# stat chips
chips = [
    ("~50M", "参数规模（12L · 8H · 512D）"),
    ("42", "项现代化升级"),
    ("600+", "回归测试全绿"),
    ("6", "篇架构决策 ADR"),
]
cw, chh, gap = 2.94, 1.26, 0.20
cy = 0.52
for i, (num, cap) in enumerate(chips):
    cx = tx + i * (cw + gap)
    ax.add_patch(
        FancyBboxPatch(
            (cx, cy), cw, chh,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            fc="#101B36", ec="#26375F", lw=1.3, zorder=3,
        )
    )
    ax.text(cx + 0.26, cy + chh / 2 + 0.13, num, ha="left", va="center", fontsize=27,
            color="#38BDF8", family="DejaVu Sans", fontweight="bold", zorder=4)
    ax.text(cx + 0.26, cy + 0.27, cap, ha="left", va="center", fontsize=10.5,
            color="#8EA3C8", fontproperties=zh, zorder=4)

ax.text(W - 0.45, 0.32, "github.com/ShaneLiu04/nanoGPT-Modern",
        ha="right", va="center", fontsize=12, color="#3F4F75",
        family="DejaVu Sans", zorder=4)

fig.savefig(OUT, dpi=100, facecolor="#060B18")
print("wrote", OUT)
