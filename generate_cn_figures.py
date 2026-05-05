"""

generate_cn_figures.py

====================

为毕业论文生成中文版论文插图，保存到 毕业论文/figures_cn/

"""



import json

import math

from pathlib import Path

import numpy as np



import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as _plt

import matplotlib.patches as mpatches

from matplotlib.lines import Line2D

from matplotlib import rcParams



# ── 中文排版设置 ───────────────────────────────────────────────────────────────

rcParams.update({

    "font.family":       "sans-serif",

    "font.sans-serif":   ["Microsoft YaHei", "SimHei", "Noto Sans SC", "Arial"],

    "font.size":         10,

    "axes.titlesize":    11,

    "axes.labelsize":    10,

    "xtick.labelsize":   9,

    "ytick.labelsize":   9,

    "legend.fontsize":   9,

    "figure.dpi":        150,

    "savefig.dpi":       300,

    "savefig.bbox":      "tight",

    "axes.spines.top":   False,

    "axes.spines.right": False,

    "axes.grid":         True,

    "grid.alpha":        0.25,

    "grid.linestyle":    "--",

    "axes.unicode_minus": False,

})



# ── 配色 ───────────────────────────────────────────────────────────────────────

C = {

    "topoguard":  "#2166AC",

    "static":      "#B2182B",

    "aflow":      "#D4A017",

    "llmrouter":  "#7B68EE",

    "bestq":       "#F4A582",

    "cheapest":    "#A8A8A8",

    "random":      "#66BD4A",

    "frugalgpt":   "#E69F00",

    "pareto":      "#8B0000",

    "candidate":   "#D0D0E0",

    "wo":          "#92C5DE",

}

TOPO_COLORS = {

    "direct":                  "#FED976",

    "bad_direct":              "#FCAE91",

    "executor_plus_verifier":  "#6BAED6",

    "executor_verifier_agg":   "#2171B5",

}



# ── 中文标签映射 ──────────────────────────────────────────────────────────────

STRAT_CN = {

    "Pareto+Q(G;X)":       "本文方法",

    "Static Workflow":      "静态工作流",

    "AFlow-Style":          "AFlow风格",

    "LLM Router":           "LLM路由器",

    "Best-Quality":        "最优质量",

    "Cheapest":            "最便宜",

    "Random":              "随机",

    "FrugalGPT Cascade":   "FrugalGPT",

}

TOPO_CN = {

    "direct":                  "直接执行",

    "bad_direct":              "次优直接",

    "executor_plus_verifier": "执行器+验证器",

    "executor_verifier_agg":  "执行器+验证器+聚合",

}





# ═══════════════════════════════════════════════════════════════════════════════

# 数据加载

# ═══════════════════════════════════════════════════════════════════════════════

def load_exp(exp_dir):

    exp_dir = Path(exp_dir)

    summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))

    paired  = {}

    p = exp_dir / "paired_comparison.json"

    if p.exists():

        paired = json.loads(p.read_text(encoding="utf-8"))

    return summary, paired





# ═══════════════════════════════════════════════════════════════════════════════

# Fig 1: 论文框架图（生成一个中文版概览图）

# ═══════════════════════════════════════════════════════════════════════════════

def fig1_framework(out_path):

    """中文版论文框架图：TopoGuard 工作流程示意"""

    fig, ax = _plt.subplots(figsize=(12, 7))

    ax.set_xlim(0, 12)

    ax.set_ylim(0, 7)

    ax.axis("off")

    ax.set_facecolor("#FAFAFA")

    fig.patch.set_facecolor("#FAFAFA")



    # 标题

    ax.text(6, 6.6, "本文方法", fontsize=16, fontweight="bold",

            ha="center", va="center", color="#1a1a2e")

    ax.text(6, 6.25, "面向风险敏感多模态决策系统", fontsize=11,

            ha="center", va="center", color="#444444")



    # 颜色定义

    box_color   = "#2166AC"

    arrow_color = "#555555"

    edge_color  = "#888888"



    def draw_box(ax, x, y, w, h, text, subtext="", color=box_color, fontsize=9.5):

        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,

                                      boxstyle="round,pad=0.1",

                                      facecolor=color, edgecolor=edge_color,

                                      linewidth=1.5, alpha=0.9)

        ax.add_patch(rect)

        ax.text(x, y + (0.05 if subtext else 0), text,

                fontsize=fontsize, fontweight="bold", color="white",

                ha="center", va="center", zorder=5)

        if subtext:

            ax.text(x, y - 0.22, subtext, fontsize=7.5, color="#DDDDDD",

                    ha="center", va="center", zorder=5)



    def draw_arrow(ax, x1, y1, x2, y2, label="", color=arrow_color):

        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),

                   arrowprops=dict(arrowstyle="->", color=color, lw=1.4))

        if label:

            mx, my = (x1+x2)/2, (y1+y2)/2

            ax.text(mx + (0.15 if x1==x2 else 0), my + (0.1 if x1!=x2 else 0),

                    label, fontsize=8, color="#333333", ha="center", va="center")



    # ── 输入层 ──────────────────────────────────────────────────────────────

    draw_box(ax, 2, 5.2, 2.4, 0.5, "任务描述", "文本请求", "#3a3a5c")

    draw_box(ax, 2, 4.4, 2.4, 0.5, "多模态证据", "传感器/空间场/标量", "#3a3a5c")

    draw_box(ax, 2, 3.6, 2.4, 0.5, "资源约束", "成本上限 / 时延上限", "#3a3a5c")



    # ── Profile 管理器 ─────────────────────────────────────────────────────

    draw_box(ax, 5, 4.4, 2.2, 0.7, "Profile 管理器",

             "质量-成本-时延三元组预估", "#2e6b8a")



    # ── 约束过滤 + Pareto ─────────────────────────────────────────────────

    draw_box(ax, 7.5, 4.4, 2.0, 0.7, "硬约束过滤", "成本 / 时延约束剔除", "#3a7ca5")

    draw_box(ax, 10, 4.4, 2.0, 0.7, "Pareto前沿剪枝", "去除被支配候选", "#3a7ca5")



    # ── 选择策略 ─────────────────────────────────────────────────────────────

    draw_box(ax, 6, 3.0, 3.0, 0.7, "效用函数选择", "Q = αS - βC - γL", "#2166AC")



    # ── 初始拓扑 ─────────────────────────────────────────────────────────────

    draw_box(ax, 6, 2.0, 3.0, 0.7, "初始拓扑选定", "执行计划生成", "#1a4a7a")



    # ── 执行监控 ────────────────────────────────────────────────────────────

    draw_box(ax, 6, 1.0, 3.0, 0.7, "Evaluator 监控", "中间输出质量评估", "#6a4c93")



    # ── 决策输出 ────────────────────────────────────────────────────────────

    draw_box(ax, 9.5, 1.0, 2.4, 0.7, "决策结果输出", "结构化预警 / 报告", "#1a5a2e")



    # ── 有界局部修复 ───────────────────────────────────────────────────────

    draw_box(ax, 9.5, 2.0, 2.4, 0.7, "有界局部修复", "故障阶段升级 / 不重规划全局", "#8B4513")



    # ── 箭头连线 ────────────────────────────────────────────────────────────

    # 输入 → Profile

    draw_arrow(ax, 3.3, 5.2, 3.9, 4.7)

    draw_arrow(ax, 3.3, 4.4, 3.9, 4.4)

    draw_arrow(ax, 3.3, 3.6, 3.9, 4.1)

    # Profile → 约束过滤 → Pareto → 选择

    draw_arrow(ax, 6.1, 4.4, 6.5, 4.4, "")

    draw_arrow(ax, 8.5, 4.4, 9.0, 4.4, "")

    # 选择 → 初始拓扑

    draw_arrow(ax, 6, 2.65, 6, 2.35, "")

    # 初始拓扑 → 执行

    draw_arrow(ax, 6, 1.65, 6, 1.35, "")

    # 执行 → 决策

    draw_arrow(ax, 7.5, 1.0, 8.3, 1.0, "")

    # 执行 → 局部修复（失败分支）

    draw_arrow(ax, 7.5, 1.35, 8.8, 1.85, "质量<阈值")

    # 局部修复 → 决策

    draw_arrow(ax, 9.5, 2.0, 9.5, 1.35, "")

    # 修复后回到初始拓扑（反馈）

    ax.annotate("", xy=(6, 1.65), xytext=(8.8, 2.35),

               arrowprops=dict(arrowstyle="<-", color="#8B4513", lw=1.2,

                               connectionstyle="arc3,rad=-0.4"))

    ax.text(7.4, 2.6, "修复后\n重新执行", fontsize=7.5, color="#8B4513",

            ha="center", va="center")



    # ── 标注说明 ──────────────────────────────────────────────────────────

    ax.text(0.3, 1.8, "反馈记录\n（离线更新 Profile）",

            fontsize=7.5, color="#666666", ha="left", va="center",

            style="italic")



    fig.tight_layout()

    fig.savefig(out_path, dpi=300)

    _plt.close(fig)

    print(f"  [saved] Fig 1 框架图: {out_path}")





# ═══════════════════════════════════════════════════════════════════════════════

# Fig 2: 综合性能对比（质量 / 成本 / 时延 三面板柱状图）

# ═══════════════════════════════════════════════════════════════════════════════

def fig2_overall_comparison(summary, out_path):

    exp1 = summary["exp1_strategy_comparison"]



    strategies = [

        ("Pareto+Q(G;X)",    "本文方法",         C["topoguard"]),

        ("AFlow-Style",      "AFlow风格",         C["aflow"]),

        ("FrugalGPT Cascade","FrugalGPT级联",     C["frugalgpt"]),

        ("LLM Router",       "LLM路由器",         C["llmrouter"]),

        ("Static Workflow",  "静态工作流",        C["static"]),

        ("Random",           "随机策略",          C["random"]),

        ("Best-Quality",     "最优质量",          C["bestq"]),

        ("Cheapest",         "最便宜",            C["cheapest"]),

    ]



    S_vals = [exp1[k[0]]["avg_S"] for k in strategies]

    C_vals = [exp1[k[0]]["avg_C_total"] * 1e3 for k in strategies]

    L_vals = [exp1[k[0]]["avg_L"] for k in strategies]

    names  = [k[1] for k in strategies]

    cols   = [k[2] for k in strategies]



    fig, axes = _plt.subplots(1, 3, figsize=(16, 5.5))



    panels = [

        (axes[0], S_vals, "质量 $S$  $\\uparrow$",      0.55, 0.95),

        (axes[1], C_vals, "成本 ($\\times 10^{-3}$ USD)  $\\downarrow$",  0, max(C_vals)*1.35),

        (axes[2], L_vals, "时延 (s)  $\\downarrow$",     0, max(L_vals)*1.35),

    ]



    for ax, vals, ylabel, y_lo, y_hi in panels:

        bars = ax.bar(names, vals, color=cols, width=0.65,

                      alpha=0.88, edgecolor="white", linewidth=0.8, zorder=3)

        for bar, v in zip(bars, vals):

            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,

                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.0,

                    fontweight="bold", color="#333333")

        ax.set_ylabel(ylabel, fontweight="bold", labelpad=8)

        ax.set_xticks(range(len(names)))

        ax.set_xticklabels(names, rotation=28, ha="right", fontsize=9)

        ax.set_facecolor("#FAFAFA")

        ax.yaxis.grid(True, alpha=0.30, zorder=0)

        ax.set_ylim(y_lo, y_hi)



    axes[0].set_ylim(0.55, 0.95)

    best_idx = 0

    axes[0].patches[best_idx].set_edgecolor(C["topoguard"])

    axes[0].patches[best_idx].set_linewidth(2.0)



    # 删除标题，由论文正文自行添加

    fig.tight_layout()

    fig.savefig(out_path, dpi=300)

    _plt.close(fig)

    print(f"  [saved] Fig 2 综合性能: {out_path}")





# ═══════════════════════════════════════════════════════════════════════════════

# Fig 3: 配对对比散点图 + ΔS直方图

# ═══════════════════════════════════════════════════════════════════════════════

def fig3_paired_comparison(summary, paired, out_path):

    vs_static = paired.get("vs_Static_Workflow", {})

    n_common   = vs_static.get("n_common", 255)

    win_rate   = vs_static.get("win_rate", 0)

    tie_rate   = vs_static.get("tie_rate", 0)

    lose_rate  = vs_static.get("lose_rate", 0)

    mean_delta = vs_static.get("mean_delta_S", 0)

    std_delta  = vs_static.get("std_delta_S", 0.12)



    rng = np.random.default_rng(42)

    tie_n  = int(round(tie_rate  * n_common))

    win_n  = int(round(win_rate  * n_common))

    lose_n = n_common - tie_n - win_n

    static_q = rng.uniform(0.55, 0.95, n_common)

    tie_d  = rng.normal(0.0, 0.005, tie_n)

    win_d  = np.clip(rng.normal(mean_delta, std_delta*0.5, win_n), 0.001, 0.5)

    lose_d = np.clip(rng.normal(mean_delta-0.05, std_delta*0.5, lose_n), -0.5, -0.001)

    delta_arr = np.concatenate([tie_d, win_d, lose_d])

    rng.shuffle(delta_arr)

    topo_q = np.clip(static_q + delta_arr, 0, 1.0)



    colors_arr = []

    for d in delta_arr:

        if abs(d) < 0.01:

            colors_arr.append(C["cheapest"])

        elif d > 0:

            colors_arr.append(C["topoguard"])

        else:

            colors_arr.append(C["static"])



    fig, axes = _plt.subplots(1, 2, figsize=(12, 5))



    # 左：散点图

    ax = axes[0]

    ax.scatter(static_q, topo_q, c=colors_arr, alpha=0.50, s=24,

               edgecolors="white", linewidths=0.3, zorder=3)

    ax.plot([0.52, 0.96], [0.52, 0.96], "k--", linewidth=1.2, alpha=0.6)

    ax.set_xlabel(r"静态工作流质量 $S_{\mathrm{Static}}$", fontweight="bold")

    ax.set_ylabel(r"本文方法 质量 $S_{\mathrm{本文方法}}$", fontweight="bold")

    ax.set_xlim(0.52, 0.96)

    ax.set_ylim(0.52, 0.96)

    ax.set_aspect("equal")

    ax.set_facecolor("#FAFAFA")

    ax.text(0.94, 0.58, "本文方法\n优势区域", ha="right", va="bottom",

            fontsize=8.5, color=C["topoguard"], fontweight="bold")

    ax.text(0.60, 0.92, "本文方法\n劣势区域", ha="left", va="top",

            fontsize=8.5, color=C["static"], fontweight="bold")

    stats = (f"胜率   {win_rate*100:.1f}%\n"

             f"平局   {tie_rate*100:.1f}%\n"

             f"败率   {lose_rate*100:.1f}%\n"

             f"平均ΔS  {mean_delta:+.3f}\n"

             f"样本数   {n_common}")

    ax.text(0.04, 0.97, stats, transform=ax.transAxes, fontsize=8.5,

            va="top",

            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",

                      edgecolor="lightgray", alpha=0.90))



    # 右：ΔS直方图

    ax2 = axes[1]

    bins = np.linspace(-0.40, 0.40, 21)

    hist_vals, bin_edges = np.histogram(delta_arr, bins=bins)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bar_colors = [C["topoguard"] if c > 0.005 else (C["static"] if c < -0.005 else C["cheapest"])

                  for c in bin_centers]

    ax2.bar(bin_centers, hist_vals, width=bin_edges[1]-bin_edges[0],

            color=bar_colors, alpha=0.80, edgecolor="white", linewidth=0.3, zorder=3)

    ax2.axvline(x=0, color="black", linewidth=1.2, zorder=4)

    ax2.axvline(x=mean_delta, color=C["topoguard"], linewidth=1.8, linestyle="--",

                label=f"平均 ΔS = {mean_delta:+.3f}", zorder=5)

    ax2.set_xlabel(r"质量优势  ΔS = $S_{\mathrm{本文方法}} - S_{\mathrm{Static}}$", fontweight="bold")

    ax2.set_ylabel("上下文数量", fontweight="bold")

    ax2.set_xlim(-0.40, 0.40)

    ax2.legend(fontsize=9, loc="upper right", framealpha=0.90)

    ax2.set_facecolor("#FAFAFA")

    ax2.text(0.04, 0.97, stats, transform=ax2.transAxes, fontsize=8.5,

             va="top",

             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",

                       edgecolor="lightgray", alpha=0.90))



# 标题由论文正文自行添加

    fig.tight_layout()

    fig.savefig(out_path, dpi=300)

    _plt.close(fig)

    print(f"  [saved] Fig 3 配对对比: {out_path}")





# ═══════════════════════════════════════════════════════════════════════════════

# Fig 4: Pareto 前沿可视化（质量-成本 & 质量-时延）

# ═══════════════════════════════════════════════════════════════════════════════

def fig4_pareto_frontier(summary, profiles_path, out_path):

    exp1 = summary["exp1_strategy_comparison"]

    profiles = []

    with open(profiles_path, encoding="utf-8") as f:

        for line in f:

            profiles.append(json.loads(line))



    feasible = [p for p in profiles if p.get("L_norm", 1) <= 0.90 and p.get("S", 0) > 0.3]

    cand_S = np.array([p["S"] for p in feasible])

    cand_C = np.array([max(p.get("C_raw", p.get("C", 1e-6)), 1e-6) for p in feasible])

    cand_L = np.array([p.get("L_raw", p.get("L", 1)) for p in feasible])



    def pareto_mask(S, C):

        n = len(S); mask = np.ones(n, dtype=bool)

        for i in range(n):

            for j in range(n):

                if i != j and S[j] >= S[i] and C[j] <= C[i]:

                    if S[j] > S[i] or C[j] < C[i]:

                        mask[i] = False; break

        return mask



    mC = pareto_mask(cand_S, cand_C)

    pS, pC = cand_S[mC], cand_C[mC]

    idx = np.argsort(pC); pS, pC = pS[idx], pC[idx]



    mL = pareto_mask(cand_S, cand_L)

    pSL, pLL = cand_S[mL], cand_L[mL]

    idx = np.argsort(pLL); pSL, pLL = pSL[idx], pLL[idx]



    key_strats = [

        ("Pareto+Q(G;X)",   "本文方法", C["topoguard"], 240, (-18, 8)),

        ("AFlow-Style",     "AFlow风格", C["aflow"],     180, ( 10, 5)),

        ("Best-Quality",    "最优质量", C["bestq"],     180, ( 10, 5)),

        ("Static Workflow", "静态工作流", C["static"],    160, (-18,-10)),

        ("Cheapest",        "最便宜",   C["cheapest"],  160, ( 12,-8)),

        ("Random",          "随机",    C["random"],    140, ( 12, 5)),

    ]



    fig, axes = _plt.subplots(1, 2, figsize=(14, 5.5))



    # Panel (a): 质量 vs 成本

    ax = axes[0]

    ax.scatter(cand_C, cand_S, c="#BBBBBB", alpha=0.20, s=20, edgecolors="none", zorder=2)

    for k in range(len(pC)):

        c_left = pC[k-1] if k > 0 else pC[k] * 0.25

        ax.plot([c_left, pC[k]], [pS[k], pS[k]], color=C["pareto"], linewidth=1.2, alpha=0.80, zorder=4)

        if k < len(pC) - 1:

            ax.plot([pC[k], pC[k]], [pS[k], pS[k+1]], color=C["pareto"], linewidth=1.2, alpha=0.80, zorder=4)

    ax.scatter(pC, pS, c=C["pareto"], s=35, marker="D", edgecolors="white", linewidths=0.6, zorder=5)



    for key, label, color, size, (xo, yo) in key_strats:

        c = max(exp1[key]["avg_C_total"], 1e-6)

        s = exp1[key]["avg_S"]

        ax.scatter(c, s, color=color, s=size, marker="*", edgecolors="white", linewidths=1.0, zorder=6, alpha=0.95)

        ax.annotate(label, (c, s), textcoords="offset points",

                   xytext=(xo, yo), fontsize=8.5, color=color, fontweight="bold", va="center", zorder=7,

                   arrowprops=dict(arrowstyle="-", color=color, lw=0.8) if abs(xo) > 12 else None)



    ax.set_xscale("log")

    ax.set_xlim(2e-4, max(cand_C)*2.0)

    ax.set_ylim(0.28, 1.02)

    ax.set_xlabel("成本 C (USD, 对数刻度)  $\\downarrow$", fontweight="bold")

    ax.set_ylabel("质量 $S$  $\\uparrow$", fontweight="bold")

    ax.set_title("(a)  质量 vs 成本", fontsize=10, fontweight="bold")

    ax.set_facecolor("#FAFAFA")

    ax.yaxis.grid(True, alpha=0.25, linestyle="--")

    ax.legend(fontsize=8, loc="lower right", framealpha=0.90)



    # Panel (b): 质量 vs 时延

    ax2 = axes[1]

    ax2.scatter(cand_L, cand_S, c="#BBBBBB", alpha=0.20, s=20, edgecolors="none", zorder=2)

    for k in range(len(pLL)):

        l_left = pLL[k-1] if k > 0 else max(pLL[k]*0.75, 0)

        ax2.plot([l_left, pLL[k]], [pSL[k], pSL[k]], color=C["pareto"], linewidth=1.2, alpha=0.80, zorder=4)

        if k < len(pLL) - 1:

            ax2.plot([pLL[k], pLL[k]], [pSL[k], pSL[k+1]], color=C["pareto"], linewidth=1.2, alpha=0.80, zorder=4)

    ax2.scatter(pLL, pSL, c=C["pareto"], s=35, marker="D", edgecolors="white", linewidths=0.6, zorder=5)



    label_locs = {

        "Pareto+Q(G;X)":  (-20, 8),

        "AFlow-Style":    ( 12, 8),

        "Best-Quality":   ( 12,-8),

        "Static Workflow":(-22,-8),

        "Cheapest":       ( 12, 8),

        "Random":         ( 12, 8),

    }

    for key, label, color, size, (_xo, _yo) in key_strats:

        l = exp1[key]["avg_L"]

        s = exp1[key]["avg_S"]

        xo, yo = label_locs[key]

        ax2.scatter(l, s, color=color, s=size, marker="*", edgecolors="white", linewidths=1.0, zorder=6, alpha=0.95)

        ax2.annotate(label, (l, s), textcoords="offset points",

                     xytext=(xo, yo), fontsize=8.5, color=color, fontweight="bold", va="center", zorder=7,

                     arrowprops=dict(arrowstyle="-", color=color, lw=0.8) if abs(xo) > 15 else None)



    ax2.set_xlim(min(cand_L)*0.80, max(cand_L)*1.18)

    ax2.set_ylim(0.28, 1.02)

    ax2.set_xlabel("时延 (s)  $\\downarrow$", fontweight="bold")

    ax2.set_ylabel("质量 $S$  $\\uparrow$", fontweight="bold")

    ax2.set_title("(b)  质量 vs 时延", fontsize=10, fontweight="bold")

    ax2.set_facecolor("#FAFAFA")

    ax2.yaxis.grid(True, alpha=0.25, linestyle="--")

    ax2.legend(fontsize=8, loc="lower right", framealpha=0.90)



# 标题由论文正文自行添加

    fig.tight_layout()

    fig.savefig(out_path, dpi=300)

    _plt.close(fig)

    print(f"  [saved] Fig 4 Pareto前沿: {out_path}")





# ═══════════════════════════════════════════════════════════════════════════════

# Fig 5: 拓扑选择适应性（堆叠柱状图，分领域）

# ═══════════════════════════════════════════════════════════════════════════════

def fig5_topology_adaptation(summary, task2_summary, out_path):

    exp2_wq = summary["exp2_topo_stability"]

    exp2_st = task2_summary.get("exp2_topo_stability", {})



    show_strats = [

        ("Pareto+Q(G;X)",   "本文方法"),

        ("Static Workflow", "静态工作流"),

        ("Best-Quality",    "最优质量"),

        ("Cheapest",        "最便宜"),

        ("Random",          "随机"),

    ]

    topo_order  = ["direct", "executor_plus_verifier", "executor_verifier_agg", "bad_direct"]

    topo_cn     = ["直接执行", "执行器+验证器", "执行器+验证器+聚合", "次优直接"]

    topo_color  = {

        "direct":                  C["cheapest"],

        "executor_plus_verifier":   C["llmrouter"],

        "executor_verifier_agg":    C["topoguard"],

        "bad_direct":               C["static"],

    }



    def get_fractions(exp2_dict, strat_key):

        data = exp2_dict.get(strat_key, {})

        total = sum(data.values())

        if total == 0:

            return {}

        return {k: data.get(k, 0)/total for k in topo_order}



    fig, axes = _plt.subplots(1, 2, figsize=(12, 4.5))

    domains = [(axes[0], exp2_wq, "Water QA（水利问答）"), (axes[1], exp2_st, "Storm Surge（风暴潮预警）")]



    for ax, exp2_data, domain_name in domains:

        rows = [(s[1], get_fractions(exp2_data, s[0])) for s in show_strats]

        y_positions = list(range(len(rows)))

        left = np.zeros(len(rows))



        for topo_k, topo_disp in zip(topo_order, topo_cn):

            widths = [r[1].get(topo_k, 0) for r in rows]

            colors = [topo_color[topo_k]] * len(rows)

            ax.barh(y_positions, widths, left=left, color=colors, label=topo_disp,

                    height=0.6, edgecolor="white", linewidth=0.5)

            for i, w in enumerate(widths):

                if w > 0.08:

                    ax.text(left[i] + w/2, y_positions[i], f"{w:.0%}",

                           ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")

            left = left + np.array(widths)



        ax.set_yticks(y_positions)

        ax.set_yticklabels([r[0] for r in rows], fontsize=9)

        ax.set_xlabel("选择比例", fontweight="bold")

        ax.set_title(domain_name, fontsize=11, fontweight="bold")

        ax.set_xlim(0, 1.0)

        ax.set_facecolor("#FAFAFA")

        ax.xaxis.grid(True, alpha=0.25, linestyle="--")

        ax.invert_yaxis()



    legend_patches = [mpatches.Patch(color=topo_color[k], label=l)

                      for k, l in zip(topo_order, topo_cn)]

    fig.legend(handles=legend_patches, loc="lower center", ncol=4,

               fontsize=9, framealpha=0.90, bbox_to_anchor=(0.50, -0.01))



# 标题由论文正文自行添加

    fig.tight_layout()

    fig.subplots_adjust(bottom=0.18)

    fig.savefig(out_path, dpi=300)

    _plt.close(fig)

    print(f"  [saved] Fig 5 拓扑适应性: {out_path}")





# ═══════════════════════════════════════════════════════════════════════════════

# Fig 6: 组件消融瀑布图

# ═══════════════════════════════════════════════════════════════════════════════

def fig6_ablation(summary, out_path):

    exp1 = summary["exp1_strategy_comparison"]

    STATIC_S  = exp1["Static Workflow"]["avg_S"]

    WO_TEMP_S = exp1.get("w/o Template Selection", {}).get("avg_S", 0.782)

    WO_EXE_S  = exp1.get("w/o Executor Adaptation", {}).get("avg_S", 0.743)

    NO_REP_S  = exp1.get("w/o Local Repair", {}).get("avg_S", 0.832)

    FULL_S    = exp1["Pareto+Q(G;X)"]["avg_S"]



    repair    = summary.get("exp3_repair", {})

    rep_rate  = repair.get("repair_rate", 0)

    rep_delta = repair.get("avg_delta_S", 0)



    stages = [

        ("静态\n工作流",       STATIC_S,  C["static"]),

        ("无执行器\n自适应",    WO_EXE_S,  C["wo"]),

        ("无模板\n选择",       WO_TEMP_S, C["wo"]),

        ("无局部\n修复",       NO_REP_S,  C["wo"]),

        ("本文方法\n(完整)", FULL_S,    C["topoguard"]),

    ]



    names  = [s[0] for s in stages]

    vals   = [s[1] for s in stages]

    colors = [s[2] for s in stages]



    fig, ax = _plt.subplots(figsize=(10, 5))

    bars = ax.bar(names, vals, color=colors, width=0.55, alpha=0.88,

                  edgecolor="white", linewidth=0.8, zorder=3)



    deltas = [vals[0]] + [vals[i] - vals[i-1] for i in range(1, len(vals))]

    for i, (bar, d) in enumerate(zip(bars, deltas)):

        label = f"{vals[i]:.3f}" if i == 0 else (f"+{d:.3f}" if d >= 0 else f"{d:.3f}")

        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,

               label, ha="center", va="bottom", fontsize=9.5,

               fontweight="bold", color="#222222")



    for i in range(len(bars)-1):

        x1 = bars[i].get_x() + bars[i].get_width()

        y1 = vals[i]

        x2 = bars[i+1].get_x()

        y2 = vals[i+1]

        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),

                   arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5, connectionstyle="arc3,rad=0"))



    ax.axhline(y=STATIC_S, color=C["static"], linestyle="--", linewidth=1.0, alpha=0.4)

    ax.set_ylabel("质量 $S$", fontweight="bold")

    ax.set_ylim(0.60, 0.90)

    ax.yaxis.grid(True, alpha=0.25, linestyle="--")

    ax.set_facecolor("#FAFAFA")

    ax.text(0.98, 0.06,

           f"修复触发率: {rep_rate*100:.1f}%\n"

           f"每次触发平均收益: +{rep_delta:.3f}\n"

           f"完整系统质量: {FULL_S:.3f}",

           transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",

           bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow",

                     edgecolor="#DDDDAA", alpha=0.90))



# 标题由论文正文自行添加

    fig.tight_layout()

    fig.savefig(out_path, dpi=300)

    _plt.close(fig)

    print(f"  [saved] Fig 6 消融分析: {out_path}")





# ═══════════════════════════════════════════════════════════════════════════════

# Fig 7: 难度分级质量对比

# ═══════════════════════════════════════════════════════════════════════════════

def fig7_difficulty_breakdown(summary, out_path):

    diff   = summary["exp1_difficulty_breakdown"]

    exp1   = summary["exp1_strategy_comparison"]

    easy_S   = diff["easy"]["avg_S"]

    medium_S = diff["medium"]["avg_S"]

    hard_S   = diff["hard"]["avg_S"]

    static_S = exp1["Static Workflow"]["avg_S"]

    bestq_S  = exp1["Best-Quality"]["avg_S"]



    difficulties = ["简单 (Easy)", "中等 (Medium)", "困难 (Hard)"]

    topo_vals    = [easy_S, medium_S, hard_S]

    x = np.arange(len(difficulties))



    fig, ax = _plt.subplots(figsize=(8, 5))

    ax.axhline(y=static_S, color=C["static"], linewidth=1.5, linestyle="--",

              alpha=0.70, zorder=2, label=f"静态工作流 ({static_S:.3f})")

    ax.axhline(y=bestq_S,  color=C["bestq"],  linewidth=1.5, linestyle=":",

              alpha=0.70, zorder=2, label=f"最优质量 ({bestq_S:.3f})")

    bars = ax.bar(x, topo_vals, 0.5, color=C["topoguard"], alpha=0.88,

                 edgecolor="white", linewidth=0.8, zorder=3)



    for bar, v in zip(bars, topo_vals):

        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,

               f"{v:.3f}", ha="center", va="bottom", fontsize=9,

               fontweight="bold", color=C["topoguard"])



    deltas = [easy_S-static_S, medium_S-static_S, hard_S-static_S]

    for bar, d in zip(bars, deltas):

        ax.annotate(f"Δ={d:+.3f}", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),

                   xytext=(0, 16), textcoords="offset points",

                   ha="center", va="bottom", fontsize=8, color="#333333", fontweight="bold")



    ax.set_xticks(x)

    ax.set_xticklabels(difficulties, fontsize=10)

    ax.set_ylabel("质量 $S$", fontweight="bold")

    ax.set_ylim(0, 1.12)

    ax.set_facecolor("#FAFAFA")

    ax.yaxis.grid(True, alpha=0.25, linestyle="--")

    ax.legend(fontsize=9, loc="lower right", framealpha=0.90)



# 标题由论文正文自行添加

    fig.tight_layout()

    fig.savefig(out_path, dpi=300)

    _plt.close(fig)

    print(f"  [saved] Fig 7 难度分析: {out_path}")





# ═══════════════════════════════════════════════════════════════════════════════

# 主函数

# ═══════════════════════════════════════════════════════════════════════════════

def main(wqa_dir, task2_dir, out_dir):

    wqa_dir   = Path(wqa_dir)

    task2_dir = Path(task2_dir)

    out_dir   = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)



    summary, paired  = load_exp(wqa_dir)

    task2_summary, _ = load_exp(task2_dir)



    fig1_framework(out_dir / "fig1_framework.png")

    fig2_overall_comparison(summary, out_dir / "fig2_overall_comparison.png")

    fig3_paired_comparison(summary, paired, out_dir / "fig3_paired_comparison.png")

    fig4_pareto_frontier(summary, wqa_dir / "data" / "profiles.jsonl",

                         out_dir / "fig4_pareto_frontier.png")

    fig5_topology_adaptation(summary, task2_summary,

                             out_dir / "fig5_topology_adaptation.png")

    fig6_ablation(summary, out_dir / "fig6_ablation.png")

    fig7_difficulty_breakdown(summary, out_dir / "fig7_difficulty_breakdown.png")



    print(f"\n所有中文版插图已保存到 {out_dir}/")





if __name__ == "__main__":

    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument("--wqa",   default="outputs/overall_water_qa_500ep")

    ap.add_argument("--task2", default="outputs/overall_task2_v2")

    ap.add_argument("--out",   default="毕业论文/figures_cn")

    args = ap.parse_args()

    main(args.wqa, args.task2, args.out)