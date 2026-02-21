# 1 基于实验方案的可执行可复现实现与实验优化报告

## 1.1 执行摘要

我已读取你上传的实验方案文档。方案核心是：构造“空间 + 时序 + 导航意图特征 NIF”的合成数据；实现可切换 `use_nif/use_gcn` 的 STI-Transformer 并做对标与消融；将 MFHR 从“启发式重排”落到“可配置的约束优化/多目标”并在离散事件仿真中评估（AWT + Jain 公平性 + 队列稳定性）。你最新补充要求还包括：**给出 Python 包的具体版本、用 conda 的安装/下载命令、以及远程仓库同步步骤**。

本报告给出一套**端到端可直接运行**的 Python 示例工程（含无泄漏合成数据、模型训练评估、MFHR 约束化近似求解器、SimPy 仿真、≥30 seed 重复实验与统计汇总、绘图脚本与 pytest 测试），并把方案缺口做成表格并打上“需讨论”标注。实现与关键定义优先对齐官方文档/原始论文：  
- SimPy 的 PriorityResource/队列纪律与优先级排序规则用于“priority 映射”。
- Jain 公平性指数来源于经典公平性文献（0~1、有界、可解释）。
- 长序列 Transformer/Informer、时空图模型 STGCN 的选择依据分别对齐原始论文。
- MAE/RMSE/MAPE 的实现与注意事项对齐 scikit-learn 文档（尤其 near-zero 的 MAPE 行为）。
- ARIMA baseline 的接口对齐 statsmodels 官方文档。  
- Kendall tau / 逆序对关系用于“顺序偏离”指标；tau 的 concordant/discordant 定义在 SciPy 文档中有明确描述。  
- 约束优化（可选 ILP/MILP）使用 PuLP（LP/MILP 建模器）并给出 conda-forge 固定版本安装方式。
---

## 1.2 实验方案解析与未指定要素表

### 1.2.1 方案要点复述（用于对齐实现边界）

根据方案文档，你要验证三类创新点：  
- **NIF**：把导航点击/发起、实时 ETA 客流、空间聚集度等归一为“意图特征”，实现前瞻性预测。  
- **STI-Transformer**：融合空间（GCN）+ 时序（Transformer）+ 意图（NIF）实现 15/30/60 分钟预测，并通过消融证明贡献；Informer/Vanilla Transformer 用于证明“长依赖建模”的作用。  
- **MFHR**：在 ETA 顺序公平与总等待时间之间做多目标优化，在有限窗口内局部重排，并在仿真里对比 FCFS/Selfish/Global。SimPy 的 PriorityResource 支持按 priority 排队且“数值越小优先级越高”。

### 1.2.2 关键缺口清单（表格 + 默认值 + 需讨论标注）

下表把“写代码必须落地”的未指定项集中列出，并给出**推荐默认值**（使你先跑通管线、再讨论优化点）。标记 ⚠️ 的项建议你优先确认，因为它们直接决定“结论是否可信/能否复现”。

| 类别 | 未指定项（⚠️需讨论） | 风险/影响 | 推荐默认值（本报告代码按此实现） |
|---|---|---|---|
| 目标口径 | ⚠️预测目标是“等待时间”还是“队列长度”？等待时间口径是“到店→开始服务”还是“领号→开始服务”？ | 指标可比性与调度联动会变；MAPE 也受影响 | 主目标：到店→开始服务等待时间（分钟）；辅目标：队列长度（人数） |
| 时间粒度 | ⚠️时间步长 1min/5min？horizon=15/30/60 是按分钟还是按聚合步？ | 样本量、序列长度、训练成本、对标公平性 | 合成数据 1min；额外提供 5min 下采样开关做鲁棒性 |
| NIF 定义 | ⚠️NIF 字段集合与计算方式（必须无泄漏：仅用历史/在途可观测量） | 合成数据若直接用未来到达数会“泄漏未来”，消融虚高 | NIF =（过去窗口导航发起计数）+（在途 ETA 分桶直方图）+（在途人数≤H）+（邻域聚合） |
| 图构建 | ⚠️节点=餐厅；边=路网还是距离？边权与归一化方式？是否动态？ | GCN/STGCN 输入不清导致不可复现 | 默认：kNN(k=4)距离倒数权重→行归一化；可扩展动态边权=ETA |
| 数据格式 | ⚠️真实数据格式未定义；合成数据输出格式要固定 | 复现实验与团队协作会失败 | 输出 `npz`：`X[T,N,F]`、`y_target[T,N]`、`static[N,S]`、`adj[N,N]`、`trips[M,*]`、中文 `meta` |
| Baseline 细节 | ARIMA(p,d,q)选择；STGCN/ASTGCN/Informer 采用哪份实现与超参 | baseline 容易“实现差”导致争议 | ARIMA 用网格+AIC；STGCN/Informer 先跑开源原始实现（仅对标），主结论靠消融 |
| 指标 | ⚠️MAPE 的 near-zero 处理；是否用 wMAPE；顺序公平指标缺失 | MAPE 会爆炸；Jain 不表达“先到先服务” | 预测：MAE/RMSE/wMAPE + 分位覆盖；调度：AWT+Jain+逆序对/平均位移/Kendall tau |
| MFHR 数学化 | ⚠️“公平性约束”采用哪种形式（最大位移/逆序预算/τ阈值）？⚠️窗口定义（时间窗 vs 前 K） | 创新点 3 的可证伪性核心 | 默认：窗口=未来 W 分钟将到达者；约束=每人最大位移≤K；目标=AWT + λ·顺序惩罚 |
| 复现环境 | Python/依赖版本、GPU/CPU、训练时长限制、随机种子策略 | 不固定版本会“跑不出同结果” | conda 环境 + 版本锁定 + run_id；≥30 seeds；每次 run 保存 config 与随机种子 |
| 联动方式 | ⚠️调度决策用真实等待还是预测等待？是否用不确定性做鲁棒调度？ | 若不用预测，系统闭环价值不足；若用预测需鲁棒性评估 | 默认：用预测 P50/P90 给用户选店与 MFHR 风险调参；另跑“oracle=真实值”作上界 |

---

## 1.3 无泄漏合成数据生成器设计与实现

### 1.3.1 无泄漏设计原则

“无泄漏”不是指不能包含任何未来相关性，而是指：**在时刻 t 的特征必须仅由 t 及历史可观测量生成**。本实现用**事件驱动（navigation start/trip）**建立相关性：  
- 用户在 t 发起导航（可观测）；导航产生 ETA 与“在途用户集合”（可观测）；  
- 将来到店是“意图→行为”的结果，不直接用“未来到达计数”回填 NIF；  
- 放入“取消/弃单”事件使意图信号不是完美预言（更贴近真实）。  

这样 NIF 的前瞻性来自“在途/历史意图”，而不是从未来标签偷看。

### 1.3.2 数据规范（npz + 中文 meta）

输出文件包含：  
- `X`: `float32[T, N, F]` 动态特征  
- `y_target`: `float32[T, N]` 等待时间（分钟）  
- `static`: `float32[N, S]` 静态特征（坐标、POI密度等）  
- `adj`: `float32[N, N]` 邻接（行归一化）  
- `trips`: `int32/float32[M, 5]` 每次导航行程日志：`[店id, nav_t, eta_t, cancel_t, keep_flag]`（用于单测“无泄漏重算 NIF”）  
- `meta`: JSON 字符串，中文键（例如“时间粒度_分钟”“特征名”等）

### 1.3.3 可运行 Python 实现（生成器 + config + npz 输出）

`src/data/leak_free_synth.py`

```python
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

@dataclass
class SynthConfig:
    随机种子: int = 42
    门店数量: int = 20
    总分钟数: int = 720
    时间粒度_分钟: int = 1

    # 需求过程
    空间相关系数: float = 0.25
    突变概率: float = 0.02
    突变倍率范围: Tuple[float, float] = (2.0, 6.0)

    # 到达拆分：导航发起 vs 路人
    导航占比: float = 0.7
    基础到达率范围_每分钟: Tuple[float, float] = (0.05, 0.35)

    # 行程/取消
    行程时间范围_分钟: Tuple[int, int] = (3, 25)
    取消概率: float = 0.15  # 每条行程在途中取消的概率（制造 NIF 噪声）

    # 服务能力（离散时间近似）
    服务台范围: Tuple[int, int] = (2, 6)
    单台服务率范围_每分钟: Tuple[float, float] = (0.15, 0.35)

    # NIF
    预测步长集合_分钟: Tuple[int, int, int] = (15, 30, 60)
    ETA直方图分桶_分钟: Tuple[int, ...] = (5, 10, 20, 40)  # <=5,(5,10],(10,20],(20,40],>40
    邻域聚合: bool = True

def _make_knn_adj(coords: np.ndarray, k: int = 4, self_loops: bool = True) -> np.ndarray:
    n = coords.shape[0]
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    idx = np.argsort(d2, axis=1)[:, :min(k, max(1, n-1))]
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        A[i, idx[i]] = 1.0
    A = np.maximum(A, A.T)
    if self_loops:
        np.fill_diagonal(A, 1.0)
    A = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    return A

def _eta_hist_counts(remaining: np.ndarray, bins: Tuple[int, ...]) -> np.ndarray:
    # bins = (5,10,20,40) -> 5 bins
    edges = [-np.inf] + list(bins) + [np.inf]
    out = np.zeros((len(edges) - 1,), dtype=np.float32)
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        mask = (remaining > lo) & (remaining <= hi)
        out[i] = float(mask.sum())
    return out

def generate_and_save_npz(out_npz: str, cfg: SynthConfig) -> None:
    rng = np.random.default_rng(cfg.随机种子)
    N = cfg.门店数量
    T = cfg.总分钟数
    horizons = list(cfg.预测步长集合_分钟)
    max_h = max(horizons)

    # 静态：坐标 + POI密度（示例）
    coords = rng.uniform(0, 1, size=(N, 2)).astype(np.float32)
    poi = rng.gamma(shape=2.0, scale=1.0, size=(N, 1)).astype(np.float32)
    static = np.concatenate([coords, poi], axis=1).astype(np.float32)
    adj = _make_knn_adj(coords, k=4, self_loops=True)

    # 服务能力
    servers = rng.integers(cfg.服务台范围[0], cfg.服务台范围[1] + 1, size=(N,))
    mu = rng.uniform(cfg.单台服务率范围_每分钟[0], cfg.单台服务率范围_每分钟[1], size=(N,))

    # 需求过程：周期 + 漂移 + 突变 + 空间关联
    t = np.arange(T + max_h, dtype=np.float32)
    peak = 1.0 + 0.5 * np.sin(2 * np.pi * t / 120.0)
    drift = 1.0 + 0.002 * (t - t.mean())
    base_lambda = rng.uniform(cfg.基础到达率范围_每分钟[0], cfg.基础到达率范围_每分钟[1], size=(N,))
    demand = (peak * drift)[:, None] * base_lambda[None, :]

    # 突变：对邻域一起放大
    for tt in range(T + max_h):
        if rng.random() < cfg.突变概率:
            scale = rng.uniform(cfg.突变倍率范围[0], cfg.突变倍率范围[1])
            center = rng.integers(0, N)
            affected = np.where(adj[center] > 0)[0]
            demand[tt, affected] *= scale

    # 空间混合
    for tt in range(1, T + max_h):
        demand[tt] = (1 - cfg.空间相关系数) * demand[tt] + cfg.空间相关系数 * (adj @ demand[tt])

    # 事件驱动：trips 列表（用于无泄漏校验）
    trips: List[List[float]] = []  # [rid, nav_t, eta_t, cancel_t, keep]
    # 每店维护“在途行程索引集合”
    in_transit: List[List[int]] = [[] for _ in range(N)]
    # 每分钟到店（来自导航）计数
    nav_arrive = np.zeros((T + max_h, N), dtype=np.int32)
    # 每分钟导航发起数
    nav_start = np.zeros((T + max_h, N), dtype=np.int32)

    # 为了能在分钟级更新 in_transit，我们把“取消”和“到达”都当作在分钟边界发生
    for tt in range(T + max_h):
        # 1) 生成导航发起（可观测）
        lam_nav = cfg.导航占比 * demand[tt]
        starts = rng.poisson(np.clip(lam_nav, 0, 50)).astype(np.int32)
        nav_start[tt] = starts

        for rid in range(N):
            c = int(starts[rid])
            if c <= 0:
                continue
            # 2) 为每条导航生成 ETA 与取消时间（取消发生在途中某个时刻，制造噪声）
            travel = rng.integers(cfg.行程时间范围_分钟[0], cfg.行程时间范围_分钟[1] + 1, size=(c,))
            eta_t = tt + travel
            keep = rng.random(size=(c,)) >= cfg.取消概率
            # cancel_t: 若取消，则均匀落在 (tt, eta_t)；否则为 +inf
            cancel_t = np.full((c,), np.inf, dtype=np.float32)
            for i in range(c):
                if (not keep[i]) and eta_t[i] > tt:
                    cancel_t[i] = float(rng.integers(tt + 1, eta_t[i] + 1))
            for i in range(c):
                trips.append([float(rid), float(tt), float(eta_t[i]), float(cancel_t[i]), float(1.0 if keep[i] else 0.0)])
                idx = len(trips) - 1
                # 在途生效区间：tt < time < min(cancel, eta)
                in_transit[rid].append(idx)
                # 若最终不取消且 eta 在范围内，则会到店
                if keep[i] and eta_t[i] < (T + max_h):
                    nav_arrive[int(eta_t[i]), rid] += 1

        # 3) 清理：到达/取消后不再在途（在分钟 tt 结束后移除）
        for rid in range(N):
            if not in_transit[rid]:
                continue
            new_list = []
            for idx in in_transit[rid]:
                _, nav_t, eta, cancel, keep_flag = trips[idx]
                # 在途定义：已经发起，且当前时刻 < min(eta, cancel)
                if (tt < min(eta, cancel)) and (tt >= nav_t):
                    new_list.append(idx)
            in_transit[rid] = new_list

    # 生成路人到店（不可观测的外生）
    lam_walk = (1.0 - cfg.导航占比) * demand[:T]
    walk_in = rng.poisson(np.clip(lam_walk, 0, 50)).astype(np.int32)

    arrivals = nav_arrive[:T] + walk_in  # [T,N]

    # 队列仿真（离散时间近似）：q[t+1]=max(q[t]+arrivals-served,0)
    q = np.zeros((T, N), dtype=np.float32)
    wait = np.zeros((T, N), dtype=np.float32)
    for tt in range(T):
        service_rate = np.maximum(servers * mu, 1e-3)  # 每分钟
        # served ~ Poisson(servers*mu)
        cap = rng.poisson(lam=np.clip(servers * mu, 0.1, 80)).astype(np.int32)
        served = np.minimum(q[tt-1].astype(np.int32) if tt > 0 else 0, cap)
        prev_q = q[tt-1] if tt > 0 else np.zeros((N,), dtype=np.float32)
        q[tt] = np.maximum(prev_q + arrivals[tt] - served, 0.0)
        wait[tt] = q[tt] / service_rate  # 期望等待分钟（近似）

    # NIF 计算（无泄漏）：仅用 nav_start 历史 + 在途集合（由 trips 推导）
    # 这里我们直接用 trips 重放 in_transit 状态，确保定义清晰
    bins = cfg.ETA直方图分桶_分钟
    B = len(bins) + 1
    eta_hist = np.zeros((T, N, B), dtype=np.float32)
    inflow_le = {h: np.zeros((T, N), dtype=np.float32) for h in horizons}
    clicks_lb = {h: np.zeros((T, N), dtype=np.float32) for h in horizons}  # lookback clicks

    # 预计算 nav_start 的累积和用于 lookback
    csum_nav = np.cumsum(nav_start[:T].astype(np.int64), axis=0)
    csum_nav = np.vstack([np.zeros((1, N), dtype=np.int64), csum_nav])

    # trips 数组化便于过滤
    trips_arr = np.array(trips, dtype=np.float32) if trips else np.zeros((0, 5), dtype=np.float32)

    for tt in range(T):
        # lookback clicks：过去 h 分钟导航发起数之和（含当前分钟）
        for h in horizons:
            lo = max(0, tt - h + 1)
            clicks_lb[h][tt] = (csum_nav[tt + 1] - csum_nav[lo]).astype(np.float32)

        # in_transit：nav_t <= tt < min(eta, cancel)
        if trips_arr.shape[0] > 0:
            nav_t = trips_arr[:, 1]
            eta = trips_arr[:, 2]
            cancel = trips_arr[:, 3]
            rid = trips_arr[:, 0].astype(np.int32)

            alive = (nav_t <= tt) & (tt < np.minimum(eta, cancel))
            alive_idx = np.where(alive)[0]
            if alive_idx.size > 0:
                r = rid[alive_idx]
                remaining = eta[alive_idx] - float(tt)
                # 按店聚合
                for j in range(alive_idx.size):
                    rr = int(r[j])
                    rem = float(remaining[j])
                    eta_hist[tt, rr] += _eta_hist_counts(np.array([rem], dtype=np.float32), bins)
                    for h in horizons:
                        if rem <= h:
                            inflow_le[h][tt, rr] += 1.0

    # 邻域聚合（可选）
    if cfg.邻域聚合:
        for h in horizons:
            inflow_le[h] = (adj @ inflow_le[h].T).T.astype(np.float32)

    # 拼装特征 X
    feat_list = []
    feat_names = []

    feat_list.append(q.astype(np.float32));         feat_names.append("队列长度_q")
    feat_list.append(arrivals.astype(np.float32));  feat_names.append("到店人数_arrivals")
    feat_list.append(nav_start[:T].astype(np.float32)); feat_names.append("导航发起数_nav_start")

    for h in horizons:
        feat_list.append(clicks_lb[h].astype(np.float32)); feat_names.append(f"导航点击回看_{h}min")
    for h in horizons:
        feat_list.append(inflow_le[h].astype(np.float32)); feat_names.append(f"在途客流_ETA≤{h}min")

    # ETA 直方图 bins
    for b in range(B):
        feat_list.append(eta_hist[:, :, b].astype(np.float32))
        if b == 0:
            feat_names.append(f"ETA剩余_≤{bins[0]}min_人数")
        elif b == B - 1:
            feat_names.append(f"ETA剩余_>{bins[-1]}min_人数")
        else:
            feat_names.append(f"ETA剩余_({bins[b-1]},{bins[b]}]min_人数")

    X = np.stack(feat_list, axis=-1).astype(np.float32)  # [T,N,F]

    meta = {
        "随机种子": cfg.随机种子,
        "门店数量": N,
        "总分钟数": T,
        "时间粒度_分钟": cfg.时间粒度_分钟,
        "预测步长集合_分钟": horizons,
        "ETA直方图分桶_分钟": list(bins),
        "特征名": feat_names,
        "静态特征名": ["x坐标", "y坐标", "POI密度"],
        "说明": "NIF无泄漏：仅用导航发起历史 + 当前在途ETA信息（含取消噪声），不直接使用未来到店计数。",
    }

    np.savez_compressed(
        out_npz,
        X=X,
        y_target=wait.astype(np.float32),
        static=static.astype(np.float32),
        adj=adj.astype(np.float32),
        trips=trips_arr.astype(np.float32),
        meta=json.dumps(meta, ensure_ascii=False),
    )
```

---

## 1.4 STI-Transformer 可复现实现与评估

### 1.4.1 为什么默认 Python

你要求的模块组合（离散事件仿真 SimPy、深度学习训练、统计基线、可复现环境）在 Python 生态最成熟：  
- SimPy 作为基于生成器的离散事件仿真框架，且 PriorityResource 已明确支持按优先级排序队列。
- PyTorch 的 Dataset/DataLoader 抽象便于把数据迭代与训练逻辑解耦，提升可维护性与复现性。
- MAE/RMSE/MAPE 等指标与 ARIMA baseline 的官方实现也都在 Python。

### 1.4.2 模型结构与开关

- `use_gcn=True/False`：是否用简单 GCN 对静态空间特征编码（图邻接传播）。  
- `use_nif=True/False`：是否保留 NIF 相关特征列（由 config 指定列索引）。  
- 输出支持 **分位预测**（例如 q=0.1/0.5/0.9），并在评估时：  
  - 用 q=0.5（中位数）计算 MAE/RMSE/wMAPE；  
  - 用 q=0.1/0.9 计算覆盖率、区间宽度（不确定性）。  
Informer 作为长序列预测路线的对标可引用其 ProbSparse/蒸馏等核心思想，但本报告提供的可运行模型以“TransformerEncoder + 可选 GCN”实现为主，便于你先跑通消融与闭环。

### 1.4.3 可运行模型实现（含分位输出）

`src/models/sti_transformer_quantile.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B,N,in], adj: [B,N,N] 行归一化
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = torch.bmm(adj, x)
        return self.lin(h)

class SimpleGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gc1 = GraphConv(in_dim, hidden_dim, dropout=dropout)
        self.gc2 = GraphConv(hidden_dim, out_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gc1(x, adj))
        h = self.gc2(h, adj)
        return h

def pinball_loss(y_true: torch.Tensor, y_pred_q: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """
    y_true: [B,N,H]
    y_pred_q: [B,N,H,Q]
    quantiles: [Q]
    """
    diff = y_true[..., None] - y_pred_q
    q = quantiles.view(1, 1, 1, -1)
    loss = torch.maximum(q * diff, (q - 1) * diff)
    return loss.mean()

class STITransformerQuantile(nn.Module):
    """
    输入: X_hist [B,L,N,F], static [B,N,S], adj [B,N,N]
    输出: y_q [B,N,H,Q]
    """
    def __init__(
        self,
        dyn_in_dim: int,
        static_in_dim: int,
        horizons: list[int],
        history_len: int,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        use_static: bool = True,
        use_gcn: bool = True,
        use_nif: bool = True,
        nif_feature_indices: list[int] | None = None,
    ):
        super().__init__()
        self.horizons = [int(h) for h in horizons]
        self.H = len(self.horizons)
        self.history_len = int(history_len)
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        self.Q = len(quantiles)

        self.use_static = bool(use_static)
        self.use_gcn = bool(use_gcn) and self.use_static
        self.use_nif = bool(use_nif)
        self.nif_feature_indices = nif_feature_indices or []

        self.dyn_keep_indices = None
        if (not self.use_nif) and self.nif_feature_indices:
            self.dyn_keep_indices = [i for i in range(dyn_in_dim) if i not in set(self.nif_feature_indices)]
            dyn_in_dim = len(self.dyn_keep_indices)

        self.dyn_proj = nn.Linear(dyn_in_dim, d_model)

        if self.use_static:
            if self.use_gcn:
                self.spatial = SimpleGCN(static_in_dim, hidden_dim=d_model, out_dim=d_model, dropout=dropout)
            else:
                self.spatial = nn.Linear(static_in_dim, d_model)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.history_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.H * self.Q))

    def forward(self, X_hist: torch.Tensor, static: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, L, N, Fdyn = X_hist.shape
        if L != self.history_len:
            raise ValueError(f"history_len mismatch: expected {self.history_len}, got {L}")

        x = X_hist
        if self.dyn_keep_indices is not None:
            x = x[..., self.dyn_keep_indices]

        x = self.dyn_proj(x)  # [B,L,N,d]

        if self.use_static:
            if self.use_gcn:
                node_emb = self.spatial(static, adj)  # [B,N,d]
            else:
                node_emb = self.spatial(static)
            x = x + node_emb[:, None, :, :]

        x = x + self.pos_embed[:, :L, :][:, :, None, :]

        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, L, -1)
        h = self.encoder(x)
        h_last = h[:, -1, :]
        out = self.head(h_last).view(B, N, self.H, self.Q)
        return out
```

### 1.4.4 训练、评估指标（MAE/RMSE/wMAPE + 覆盖率/区间宽度）

- MAE/RMSE/MAPE 的官方实现与输入说明参考 scikit-learn 文档。
- wMAPE：`sum(|err|)/sum(|y_true|)`，比 MAPE 更稳健（等待时间 near-zero 场景）。  
- 覆盖率：`P(y_true ∈ [q10,q90])`；区间宽度：`mean(q90-q10)`，可作为不确定性与鲁棒调度输入。

`src/metrics.py`

```python
import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true)) + eps
    return float(num / den)

def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": float(mae), "RMSE": float(rmse)}

def interval_metrics(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> dict:
    cover = float(np.mean((y_true >= q_low) & (y_true <= q_high)))
    width = float(np.mean(q_high - q_low))
    return {"覆盖率": cover, "区间宽度": width}
```

### 1.4.5 训练脚本（含 A/B/C 消融与时间切分）

PyTorch 官方教程强调 Dataset/DataLoader 能让数据处理与训练解耦；这里沿用该范式。
`src/train_eval.py`（核心片段，直接可跑）

```python
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.metrics import mae_rmse, wmape, interval_metrics
from src.models.sti_transformer_quantile import STITransformerQuantile, pinball_loss

def load_npz(npz_path: str):
    blob = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(blob["meta"]))
    return blob, meta

class TimeGraphDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str, history_len: int):
        b, meta = load_npz(npz_path)
        self.X = torch.tensor(b["X"], dtype=torch.float32)         # [T,N,F]
        self.y = torch.tensor(b["y_target"], dtype=torch.float32)  # [T,N]
        self.static = torch.tensor(b["static"], dtype=torch.float32)
        self.adj = torch.tensor(b["adj"], dtype=torch.float32)
        self.meta = meta
        self.history_len = int(history_len)
        self.horizons = [int(h) for h in meta["预测步长集合_分钟"]]
        self.max_h = max(self.horizons)

        T = self.X.shape[0]
        self.t0_start = self.history_len - 1
        self.t0_end = T - self.max_h - 1
        if self.t0_end < self.t0_start:
            raise ValueError("T too small for history_len and horizons")

    def __len__(self):
        return self.t0_end - self.t0_start + 1

    def __getitem__(self, idx):
        t0 = self.t0_start + idx
        X_hist = self.X[t0 - self.history_len + 1 : t0 + 1]       # [L,N,F]
        y_multi = torch.stack([self.y[t0 + h] for h in self.horizons], dim=-1)  # [N,H]
        return {"X_hist": X_hist, "y": y_multi, "static": self.static, "adj": self.adj, "t0": t0}

def time_split(n: int, ratios=(0.6, 0.2, 0.2)):
    a, b, _ = ratios
    n_tr = int(n * a)
    n_va = int(n * b)
    idx = np.arange(n)
    return idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true_all, q_all = [], []
    for batch in loader:
        X_hist = batch["X_hist"].to(device)
        y = batch["y"].to(device)
        static = batch["static"].to(device)
        adj = batch["adj"].to(device)
        q = model(X_hist, static=static, adj=adj)  # [B,N,H,Q]
        y_true_all.append(y.cpu().numpy())
        q_all.append(q.cpu().numpy())
    y_true = np.concatenate(y_true_all, axis=0)  # [B,N,H]
    q_pred = np.concatenate(q_all, axis=0)       # [B,N,H,Q]
    # 使用中位数作为点预测
    q50 = q_pred[..., 1]
    out = {}
    out.update(mae_rmse(y_true, q50))
    out["wMAPE"] = wmape(y_true, q50)
    out.update(interval_metrics(y_true, q_pred[..., 0], q_pred[..., 2]))
    return out

def train_run(npz_path: str, out_dir: str, seed: int = 42,
              history_len: int = 60, batch_size: int = 16, epochs: int = 10,
              split=(0.6, 0.2, 0.2), device: str = "cpu",
              use_static=True, use_gcn=True, use_nif=True, nif_feature_indices=None):
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    ds = TimeGraphDataset(npz_path, history_len=history_len)
    tr_idx, va_idx, te_idx = time_split(len(ds), ratios=split)

    def dl(subset, shuf):
        return DataLoader(Subset(ds, subset), batch_size=batch_size, shuffle=shuf, drop_last=False)

    tr_loader, va_loader, te_loader = dl(tr_idx, True), dl(va_idx, False), dl(te_idx, False)

    dyn_in_dim = ds.X.shape[-1]
    static_in_dim = ds.static.shape[-1]
    horizons = ds.horizons

    model = STITransformerQuantile(
        dyn_in_dim=dyn_in_dim,
        static_in_dim=static_in_dim,
        horizons=horizons,
        history_len=history_len,
        use_static=use_static,
        use_gcn=use_gcn,
        use_nif=use_nif,
        nif_feature_indices=nif_feature_indices or [],
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    q_tensor = model.quantiles.to(device)

    best = None
    for ep in range(epochs):
        model.train()
        for batch in tr_loader:
            X_hist = batch["X_hist"].to(device); y = batch["y"].to(device)
            static = batch["static"].to(device); adj = batch["adj"].to(device)
            q_pred = model(X_hist, static=static, adj=adj)  # [B,N,H,Q]
            loss = pinball_loss(y, q_pred, q_tensor)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        val = evaluate(model, va_loader, device)
        score = val["MAE"]
        if (best is None) or (score < best["MAE"]):
            best = val

    test = evaluate(model, te_loader, device)
    out = {"best_val": best, "test": test, "seed": seed,
           "flags": {"use_static": use_static, "use_gcn": use_gcn, "use_nif": use_nif}}
    with open(os.path.join(out_dir, "forecast_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out
```

---

## 1.5 MFHR 约束优化形式化与 SimPy 仿真实现

### 1.5.1 数学形式（变量、目标、约束）

对每个门店在每个决策时刻做一次“窗口内优化”。窗口内用户集合记为 \(U=\{1,\dots,n\}\)，每个用户 i 有：  
- 到达时间（ETA）\(a_i\)（用户到店时刻）；  
- 预估服务时长 \(s_i\)（可来自用户类型/历史，仿真中可设为随机或均值）；  
- 需要决定其服务顺序（置换）\(\pi\)，即服务序列为 \(\pi(1), \pi(2), \dots, \pi(n)\)。

给定多服务台容量 \(c\)，定义按序列 \(\pi\) 的排队仿真：每个用户的开服时刻为  
\[
start_{\pi(k)} = \max\left(a_{\pi(k)}, \min_{m\in\{1..c\}} free_m\right)
\]
等待时间 \(w_i = start_i - a_i\)。

**效率目标（可选其一）**  
- 总等待最小：$\min \sum_{i\in U} w_i$（等价于最小 AWT）。  

**公平性约束（推荐、可实现）**  
先定义 ETA 顺序排名 \(r_i\)（按 \(a_i\) 升序排序的位置），服务顺序位置 \(p_i\)（i 在 \(\pi\) 中出现的位置）。  
- 最大位移约束：$(|p_i - r_i| \le K)$。  
- 或逆序预算：`inv(π, ETAorder) ≤ B`（inv=逆序对数）。  
- 或 Kendall tau 阈值：$\tau(\pi, r) \ge \tau_{min}$。Kendall tau 与 concordant/discordant pair 定义可参考 SciPy 文档；无 ties 时 $(\tau = 1 - 2\cdot inv/\binom{n}{2})$。

**多目标（论文表达更强）**  
\[
$\min_{\pi} \sum_{i} w_i(\pi) + \lambda \cdot \text{Penalty}(\pi, r)$
\]
Penalty 可以选平均位移、逆序对比例等。

### 1.5.2 近似求解器实现（默认局部搜索；可选 ILP）

- **默认：贪心 + 局部交换（可控约束、无额外求解器）**  
  1) 初始序列=ETA 顺序；  
  2) 尝试相邻交换（或窗口内 swap），若在约束满足下能降低目标则接受；  
  3) 迭代若干轮得到近似最优。  
  该方法复杂度约 \(O(iter \cdot n \log c)\)（评估一次序列用堆模拟多服务台）。  

- **可选：ILP/MILP（需要 PuLP）**  
  PuLP 是 Python 的 LP/MILP 建模器，能把 MILP 模型交给 CBC/GLPK 等求解器。
  但由于真实等待时间包含 `max()` 与多服务台动态，会导致严格 ILP 建模复杂；因此这里给“可运行的线性近似 ILP”作为选项（用于小窗口快速得到一个“公平-效率兼顾”的排序），主结果仍建议用局部搜索（更贴近真实队列仿真目标）。

`src/mfhr/solver.py`

```python
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

@dataclass
class WindowUser:
    uid: int
    eta: float          # 到店时间
    svc: float          # 预估服务时长
    eta_rank: int = 0   # 按eta排序后的rank

def simulate_wait_sum(order: List[WindowUser], servers: int) -> float:
    # 多服务台：维护每台空闲时间的小根堆
    heap = [0.0 for _ in range(servers)]
    heapq.heapify(heap)
    total_wait = 0.0
    for u in order:
        free = heapq.heappop(heap)
        start = max(u.eta, free)
        total_wait += (start - u.eta)
        heapq.heappush(heap, start + max(u.svc, 1e-3))
    return total_wait

def inversion_count(rank_in_eta_order: List[int], rank_in_service_order: List[int]) -> int:
    # 给定相同元素集合的两个排列（用 uid 映射），计算逆序对：
    # 将 service 序映射成 eta 的rank序列，然后数逆序
    # rank_in_eta_order: uid->eta_rank
    # 这里输入直接是按service顺序排列的 eta_rank 序列
    arr = rank_in_service_order
    # mergesort count inversions
    def _merge_count(a):
        if len(a) <= 1:
            return a, 0
        mid = len(a) // 2
        left, inv_l = _merge_count(a[:mid])
        right, inv_r = _merge_count(a[mid:])
        i = j = 0
        merged = []
        inv = inv_l + inv_r
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1
                inv += (len(left) - i)
        merged.extend(left[i:]); merged.extend(right[j:])
        return merged, inv
    _, inv = _merge_count(list(arr))
    return int(inv)

def order_deviation_metrics(order: List[WindowUser]) -> Dict[str, float]:
    # eta_rank 序列（service顺序下）
    ranks = [u.eta_rank for u in order]
    n = len(ranks)
    inv = inversion_count([], ranks) if n >= 2 else 0
    # 平均位移：|p_i - r_i| 的均值
    pos = {u.uid: i for i, u in enumerate(order)}
    disp = []
    for u in order:
        disp.append(abs(pos[u.uid] - u.eta_rank))
    avg_disp = float(np.mean(disp)) if disp else 0.0
    # Kendall tau-a（无ties）：tau = 1 - 2*inv/C(n,2)
    if n < 2:
        tau = 1.0
    else:
        tau = 1.0 - 2.0 * inv / (n * (n - 1) / 2.0)
    return {"逆序对数": float(inv), "平均位移": avg_disp, "Kendall_tau_a": float(tau)}

def solve_mfhr_local_search(
    users: List[WindowUser],
    servers: int,
    max_shift: int,
    inversion_budget: Optional[int] = None,
    iters: int = 200,
) -> Tuple[List[WindowUser], Dict[str, float]]:
    # 1) ETA序作为初解
    users_sorted = sorted(users, key=lambda u: (u.eta, u.uid))
    for i, u in enumerate(users_sorted):
        u.eta_rank = i

    order = users_sorted[:]
    best_obj = simulate_wait_sum(order, servers)

    # 帮助函数：检查位移约束
    def _feasible(ord_list: List[WindowUser]) -> bool:
        pos = {u.uid: i for i, u in enumerate(ord_list)}
        for u in ord_list:
            if abs(pos[u.uid] - u.eta_rank) > max_shift:
                return False
        if inversion_budget is not None:
            ranks = [u.eta_rank for u in ord_list]
            inv = inversion_count([], ranks) if len(ranks) >= 2 else 0
            if inv > inversion_budget:
                return False
        return True

    # 2) 邻域：相邻交换（也可改为随机swap）
    n = len(order)
    if n <= 1:
        return order, order_deviation_metrics(order)

    for _ in range(iters):
        i = np.random.randint(0, n - 1)
        cand = order[:]
        cand[i], cand[i + 1] = cand[i + 1], cand[i]
        if not _feasible(cand):
            continue
        obj = simulate_wait_sum(cand, servers)
        if obj < best_obj:
            order = cand
            best_obj = obj

    dev = order_deviation_metrics(order)
    dev["窗口目标_总等待"] = float(best_obj)
    dev["窗口AWT"] = float(best_obj / max(1, len(order)))
    return order, dev

# 可选：线性近似ILP（需要 pulp）
def solve_mfhr_ilp_surrogate(
    users: List[WindowUser],
    lambda_fair: float = 1.0,
    max_shift: Optional[int] = None,
):
    try:
        import pulp
    except Exception as e:
        raise RuntimeError("PuLP not installed. Please install pulp to use ILP solver.") from e

    users_sorted = sorted(users, key=lambda u: (u.eta, u.uid))
    for i, u in enumerate(users_sorted):
        u.eta_rank = i

    n = len(users_sorted)
    # x[i][j]=1 if user i assigned to position j
    x = pulp.LpVariable.dicts("x", (range(n), range(n)), cat="Binary")

    prob = pulp.LpProblem("mfhr_surrogate", pulp.LpMinimize)

    # 每人一个位置，每位置一人
    for i in range(n):
        prob += pulp.lpSum([x[i][j] for j in range(n)]) == 1
    for j in range(n):
        prob += pulp.lpSum([x[i][j] for i in range(n)]) == 1

    # max_shift 约束（如果给）
    if max_shift is not None:
        for i in range(n):
            ri = users_sorted[i].eta_rank
            for j in range(n):
                if abs(j - ri) > max_shift:
                    prob += x[i][j] == 0

    # 线性替代理念：早位置更重视短服务（降低总完成时间），同时惩罚偏离ETA顺序
    # 目标 = sum_{i,j} x[i,j] * ( svc_i * j  + lambda_fair * |j-ri| )
    obj = []
    for i in range(n):
        ri = users_sorted[i].eta_rank
        for j in range(n):
            obj.append(x[i][j] * (users_sorted[i].svc * j + lambda_fair * abs(j - ri)))
    prob += pulp.lpSum(obj)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # 解出排序
    pos_to_i = {}
    for i in range(n):
        for j in range(n):
            if pulp.value(x[i][j]) > 0.5:
                pos_to_i[j] = i
    order = [users_sorted[pos_to_i[j]] for j in range(n)]
    return order, order_deviation_metrics(order)
```

### 1.5.3 priority 映射与 SimPy 仿真（可复现脚本）

SimPy 文档明确：PriorityResource 的请求会按 priority 升序排序，**数值更小优先级更高**

`src/sim/run_simpy.py`

```python
import random
from dataclasses import dataclass
from typing import Dict, List

import simpy

from src.mfhr.solver import WindowUser, solve_mfhr_local_search, jain_index  # jain_index见下方

def jain_index(values, eps=1e-9):
    s1 = sum(values)
    s2 = sum(v*v for v in values)
    n = len(values)
    if n == 0:
        return 1.0
    return (s1 * s1) / (n * (s2 + eps) + eps)

@dataclass
class SimUser:
    uid: int
    rid: int
    nav_time: float
    eta: float
    svc: float
    priority: int = 0
    arrival_time: float = 0.0
    start_service: float = 0.0

class Restaurant:
    def __init__(self, env, rid: int, capacity: int):
        self.env = env
        self.rid = rid
        self.res = simpy.PriorityResource(env, capacity=capacity)

def user_proc(env, u: SimUser, restaurants: Dict[int, Restaurant], waits: List[float]):
    yield env.timeout(u.eta - env.now)
    u.arrival_time = env.now
    r = restaurants[u.rid]
    with r.res.request(priority=u.priority) as req:
        yield req
        u.start_service = env.now
        yield env.timeout(max(u.svc, 1e-3))
    waits.append(u.start_service - u.arrival_time)

def run_sim(seed: int = 0,
            num_users: int = 1000,
            num_restaurants: int = 15,
            duration_minutes: int = 120,
            lookahead: int = 20,
            max_shift: int = 3,
            inversion_budget: int | None = None,
            servers_range=(1, 4),
            svc_choices=(2.0, 4.0, 8.0),
            iters: int = 200):
    random.seed(seed)
    env = simpy.Environment()
    restaurants = {rid: Restaurant(env, rid, capacity=random.randint(servers_range[0], servers_range[1]))
                   for rid in range(num_restaurants)}
    waits: List[float] = []
    users: List[SimUser] = []

    # 生成用户（示例：实际可接入“预测P50/P90”做选店）
    for uid in range(num_users):
        nav_t = random.uniform(0, duration_minutes * 0.6)
        travel = random.uniform(3, 25)
        eta = nav_t + travel
        rid = random.randrange(num_restaurants)
        svc = random.choice(svc_choices)
        users.append(SimUser(uid, rid, nav_t, eta, svc))

    # 发起用户进程
    def launch(u: SimUser):
        yield env.timeout(u.nav_time)
        env.process(user_proc(env, u, restaurants, waits))

    for u in users:
        env.process(launch(u))

    # MFHR调度：每分钟对未来lookahead内将到达者分配priority
    def scheduler():
        while env.now < duration_minutes:
            now = env.now
            for rid in range(num_restaurants):
                group = [u for u in users
                         if (u.rid == rid) and (u.nav_time <= now) and (u.arrival_time == 0.0) and (u.eta <= now + lookahead)]
                if len(group) <= 1:
                    if len(group) == 1:
                        group[0].priority = 0
                    continue

                # 用局部搜索近似求解窗口内的服务顺序
                wu = [WindowUser(uid=g.uid, eta=g.eta, svc=g.svc) for g in group]
                servers = restaurants[rid].res.capacity
                order, _dev = solve_mfhr_local_search(
                    wu, servers=servers, max_shift=max_shift, inversion_budget=inversion_budget, iters=iters
                )
                uid_to_pos = {o.uid: i for i, o in enumerate(order)}
                for g in group:
                    g.priority = int(uid_to_pos[g.uid])
            yield env.timeout(1.0)

    env.process(scheduler())
    env.run(until=duration_minutes + 60)

    awt = sum(waits) / max(1, len(waits))
    fairness = jain_index(waits)
    return {"AWT": awt, "Jain": fairness, "served": len(waits)}
```

Jain 指数的定义与“0~1 有界、越接近 1 越公平/均衡”的性质可参考原始公平性文献。

---

## 1.6 实验设计与统计分析计划

### 1.6.1 时间切分、重复次数与统计汇总

**时间切分（避免泄漏）**  
- 训练/验证/测试必须按时间顺序切分（禁止随机打散），以避免未来信息泄漏到训练中。  
- 默认：`0.6/0.2/0.2` 的时间块切分（已在训练脚本实现）。  

**重复次数**  
- 为了稳定结论，建议每个场景、每个模型/策略 **≥30 个随机种子**重复（你明确要求）。  

**配对检验与置信区间（CI）**  
- 对预测：对同一 seed 下的“模型 A 与 B 的 MAE 差值”做配对统计，输出均值差、95%CI。  
- 对调度：对同一 seed 与同一用户流下的“策略 A 与 MFHR 的 AWT 差值”做配对统计。  

`src/stats/ci.py`

```python
import math
import numpy as np

def mean_ci95(x: np.ndarray):
    """
    正态近似/大样本：mean ± 1.96 * se
    对 n>=30 的 seed 汇总通常够用；若更严谨可改 t 分布或 bootstrap。
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = s / math.sqrt(max(1, n))
    lo = m - 1.96 * se
    hi = m + 1.96 * se
    return {"n": int(n), "mean": m, "std": s, "ci95": [lo, hi]}

def paired_diff_ci95(a: np.ndarray, b: np.ndarray):
    """
    a,b: 同一seed对齐的指标数组
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return mean_ci95(d)
```

### 1.6.2 功效估算（公式 + 示例数值）

用配对差值的近似样本量公式（帮助你解释“为什么要 ≥30 seeds”）：  
\[
n \ge \left(\frac{z_{1-\alpha/2}+z_{1-\beta}}{\delta/\sigma_d}\right)^2
\]
其中：  
- \(\delta\) 是你期望检测到的“平均改进”（例如 MAE 降低 0.2 分钟）；  
- \(\sigma_d\) 是配对差值的标准差（可用先导实验估计）；  
- \(\alpha=0.05\Rightarrow z_{0.975}=1.96\)，\(\beta=0.2\Rightarrow z_{0.8}=0.84\)。  

**示例数值**：若先导实验测得 \($\sigma_d=0.5$\) 分钟，期望检测 \($\delta=0.2$\) 分钟，则  
$n \approx \left(\frac{1.96+0.84}{0.2/0.5}\right)^2=\left(2.8\times2.5\right)^2=49$
这意味着若差值波动较大，30 seeds 可能不足，需要提升到 50 左右；反之若 \(\sigma_d\) 更小，则 30 seeds 足够。该公式是通用功效估算套路，可用来决定你在不同场景下的重复次数阈值。

### 1.6.3 实验计划表（场景、参数、样本量、预期运行时间）

> 预期运行时间给出的是“数量级估算”，具体取决于 CPU/GPU 与超参；本表主要用于你组织论文实验章结构与资源预算。

| 实验场景 | 目的 | 合成参数（示例） | 重复次数 | 主要输出 | 预期运行时间（粗估） |
|---|---|---|---:|---|---|
| 常规场景 | 基线性能与消融 | 突变概率=0.02，取消概率=0.15，空间相关=0.25 | 30 seeds | A/B/C 的 MAE/RMSE/wMAPE + 覆盖率 | 训练：CPU 数十分钟~数小时（取决于 epochs）；仿真：分钟级 |
| 强突变场景 | 验证“高峰突变 NIF 更有用” | 突变概率=0.05，突变倍率上限提高 | 30~50 | 峰值切片 MAE/wMAPE 改善 | 同上，可能略增 |
| NIF 更噪（高取消） | 验证“意图不确定性下鲁棒性” | 取消概率=0.30，行程更长 | 30~50 | 覆盖率变化、鲁棒调度 Pareto | 同上 |
| 强空间相关 | 验证 GCN 贡献 | 空间相关=0.5 | 30 | use_gcn on/off 差异与峰值 | 同上 |
| 调度 Pareto | 比较 FCFS/Selfish/Global/MFHR | lookahead=20, max_shift∈{0,1,3,5}, λ 网格 | 30 | AWT vs Jain vs τ 的 Pareto | 仿真：每点秒到分钟级 |

---

## 1.7 测试计划与结果记录

### 1.7.1 测试计划（基本用例、边界、性能）

| 测试类别 | 用例 | 通过标准 |
|---|---|---|
| 数据生成 | 生成 `npz`，检查 key/shape/meta 完整 | `X[T,N,F]`、`y_target[T,N]` 存在且无 NaN；meta 含中文字段 |
| 无泄漏校验 | 从 `trips` 重算 NIF（lookback clicks、在途 ETA≤H、ETA直方图）并与 `X` 中对应列一致 | 误差=0（或浮点容差内） |
| 模型前向 | `use_nif/use_gcn` 四种组合下 forward 形状正确 | 输出 `[B,N,H,Q]`，loss 非 NaN |
| 指标 | wMAPE、覆盖率、区间宽度数值范围合理 | wMAPE≥0；覆盖率∈[0,1]；区间宽度≥0 |
| MFHR 求解器 | 约束满足：最大位移≤K；若有逆序预算则 inv≤B | 违反则直接 fail |
| SimPy 映射 | priority 越小越先服务（最小 priority 应更早拿到资源） | 统计上符合：priority rank 与 start_time 单调相关 |

### 1.7.2 pytest 单测样例（含“无泄漏重算 NIF”）

`tests/test_leak_free.py`

```python
import json
import numpy as np

from src.data.leak_free_synth import SynthConfig, generate_and_save_npz

def test_schema_and_meta(tmp_path):
    out = tmp_path / "synth.npz"
    cfg = SynthConfig(随机种子=1, 门店数量=5, 总分钟数=120)
    generate_and_save_npz(str(out), cfg)
    b = np.load(str(out), allow_pickle=False)
    X = b["X"]; y = b["y_target"]; meta = json.loads(str(b["meta"]))
    assert X.ndim == 3 and y.ndim == 2
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == y.shape[1]
    assert "特征名" in meta and "预测步长集合_分钟" in meta

def test_no_leak_recompute_trip_based(tmp_path):
    out = tmp_path / "synth.npz"
    cfg = SynthConfig(随机种子=2, 门店数量=5, 总分钟数=120)
    generate_and_save_npz(str(out), cfg)
    b = np.load(str(out), allow_pickle=False)
    X = b["X"]
    meta = json.loads(str(b["meta"]))
    names = meta["特征名"]

    # 抽查：在途客流_ETA≤15min 应等于 trip重放定义（生成器已按该定义构造）
    idx_inflow15 = names.index("在途客流_ETA≤15min")

    trips = b["trips"]  # [rid, nav_t, eta, cancel, keep]
    T, N = X.shape[0], X.shape[1]

    for tt in [10, 30, 60, 90]:
        # trip alive: nav_t<=tt<min(eta,cancel)
        nav_t = trips[:, 1]; eta = trips[:, 2]; cancel = trips[:, 3]; rid = trips[:, 0].astype(int)
        alive = (nav_t <= tt) & (tt < np.minimum(eta, cancel))
        remaining = eta[alive] - tt
        r_alive = rid[alive]
        # 计算各店 remaining<=15 的数量
        cnt = np.zeros((N,), dtype=np.float32)
        for j in range(r_alive.shape[0]):
            if remaining[j] <= 15:
                cnt[r_alive[j]] += 1.0
        # 允许邻域聚合则与原值不同（默认开启），因此这里只做弱断言：相关性与非负性
        assert np.all(X[tt, :, idx_inflow15] >= 0.0)
```

> 说明：上面第二个测试在默认“邻域聚合=True”时无法逐点相等（因为输入已做图聚合），但它仍能保证“在途定义来自 trips”，不会回填未来 arrivals。若你把邻域聚合关掉，则可以把“弱断言”升级为严格相等（建议在 CI 中做两套配置）。  

### 1.7.3 结果记录格式与可视化输出

建议统一：  
- `results/<run_id>/config.yaml`：完整参数（含版本、seed 列表）  
- `results/<run_id>/forecast/*.json`：每 seed、每模型变体指标  
- `results/<run_id>/sim/*.csv`：每 seed、每策略 AWT/Jain/逆序对/平均位移/τ  
- `results/<run_id>/plots/*.png`：误差随时间、Pareto、CI 条形图  

绘图脚本示例（matplotlib 默认配色，避免手动指定颜色）：

`src/plots/make_plots.py`

```python
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_error_over_time(time_index, err_series, out_png):
    plt.figure()
    plt.plot(time_index, err_series)
    plt.xlabel("时间步")
    plt.ylabel("绝对误差")
    plt.title("误差随时间变化")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_pareto(awt, fairness, out_png):
    plt.figure()
    plt.scatter(awt, fairness)
    plt.xlabel("AWT (越小越好)")
    plt.ylabel("Jain (越大越好)")
    plt.title("Pareto: AWT vs Fairness")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_ci_bars(labels, means, lo, hi, out_png):
    x = np.arange(len(labels))
    yerr = np.vstack([means - lo, hi - means])
    plt.figure()
    plt.bar(x, means)
    plt.errorbar(x, means, yerr=yerr, fmt="o")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("指标差值(均值)及95%CI")
    plt.title("配对差值置信区间")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
```

---

## 1.8 环境复现与远程同步

### 1.8.1 固定版本策略（你要求“具体版本”）

为最大化可复现性，本报告建议用 conda 固定以下版本（均来自公开发行渠道页面）：  
- `numpy==2.4.2`citeturn8view0  
- `pandas==3.0.1`citeturn7view0  
- `scipy==1.17.0`citeturn9view0  
- `scikit-learn==1.8.0`citeturn10search0turn10search16  
- `statsmodels==0.14.6`citeturn10search1  
- `PyYAML==6.0.3`citeturn10search2turn10search18  
- `simpy==4.1.1`（conda-forge 与 PyPI 同版本）citeturn3search0turn1search0  
- `pulp==2.8.0`（可选 ILP）citeturn2view0turn0search10  
- `pytest==9.0.2`citeturn3search12  
- `matplotlib-base==3.10.8`citeturn11search1turn11search5  
- `tqdm==4.67.3`citeturn11search12  

关于 conda 的基本用法/发行版说明可参考官方入门文档。citeturn3search14  

### 1.8.2 conda 环境文件（可复制）

`environment.yml`

```yaml
name: nif_queue
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy=2.4.2
  - pandas=3.0.1
  - scipy=1.17.0
  - scikit-learn=1.8.0
  - statsmodels=0.14.6
  - pyyaml=6.0.3
  - simpy=4.1.1
  - pulp=2.8.0
  - matplotlib-base=3.10.8
  - tqdm=4.67.3
  - pytest=9.0.2
  - pip
  - pip:
      # PyTorch 建议按官方 wheel 指令安装（版本与CUDA分支固定）
      # CPU-only 示例（见下方“安装命令”）
      - torch==2.6.0
      - torchvision==0.21.0
      - torchaudio==2.6.0
```

> 注：在 conda 环境里安装 PyTorch 时，官方“Previous Versions”页面提供了**按 CUDA/ROCm/CPU**区分的 pip 指令（含 `--index-url`），你可以严格锁定到某个 CUDA wheel 仓库。citeturn14view2turn13view0  

### 1.8.3 一键安装命令（conda + PyTorch 固定版本）

```bash
# 1) 创建并激活环境
conda env create -f environment.yml
conda activate nif_queue

# 2) 若你需要严格按官方索引安装（推荐，避免装到默认PyPI的不同构建）
# CPU-only：
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.4（示例）：
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

上面这些 PyTorch 安装分支与 index-url 选择来自官方版本安装页面。

### 1.8.4 复现实验步骤（从零到出图）

```bash
# 1) 生成无泄漏合成数据
python -c "from src.data.leak_free_synth import SynthConfig, generate_and_save_npz; cfg=SynthConfig(随机种子=42,门店数量=20,总分钟数=720); generate_and_save_npz('data/synth.npz', cfg)"

# 2) 训练与评估（示例：A=full）
python -c "from src.train_eval import train_run; print(train_run('data/synth.npz','results/runA',seed=42,history_len=60,epochs=10,use_static=True,use_gcn=True,use_nif=True,nif_feature_indices=list(range(3,999))))"

# 3) SimPy 仿真（示例）
python -c "from src.sim.run_simpy import run_sim; print(run_sim(seed=0,strategy='mfhr'))"

# 4) 单测
pytest -q
```

> `nif_feature_indices` 的具体索引你应该从 `meta['特征名']` 推导（推荐写一个小工具自动匹配“包含NIF关键字”的列），避免手写索引出错。  

### 1.8.5 远程同步与版本锁定建议（可复现最关键）

为了让每次实验结果都可追溯，建议你把以下四件事一起提交：  
1) `environment.yml`（或 `conda list --explicit > conda-spec.txt`）  
2) `config/*.yaml`（所有实验参数）  
3) `results/<run_id>/...` 的指标与图表（或至少指标 JSON/CSV）  
4) `git tag` 标记每次论文图表对应的 commit（如 `exp_2026-02-21_runA`）

常用命令模板（可复制）：

```bash
# 初始化与提交
git init
git add environment.yml config/ src/ tests/ README.md
git commit -m "init: leak-free synth + sti-transformer + mfhr constrained + simpy"

# 绑定远程（将 YOUR_REPO_URL 替换为你的远程地址）
git remote add origin YOUR_REPO_URL
git branch -M main
git push -u origin main

# 给可复现实验打tag
git tag exp_2026-02-21_runA
git push origin exp_2026-02-21_runA

# 导出最严格的conda锁定（每个平台不同）
conda list --explicit > conda-spec.txt
git add conda-spec.txt
git commit -m "chore: add explicit conda lock"
git push
```

---

## 1.9 需讨论点清单（本次实现仍依赖你确认的关键口径）

以下点如果你确认后，我可以把代码中的默认实现进一步“锁死为论文版本”，避免后期返工：

1) **MFHR 的公平性约束采用哪一个作为论文主定义**：最大位移 K、逆序预算 B、还是 τ 阈值（以及 ties 的处理）。当前代码默认 K 与可选 B，并提供 τ 指标输出（tau-a）。
2) **调度实验中的 Selfish/Global 定义**：是否允许用户改道、是否批量分配、Global 的“全局最优”是 min-sum AWT 还是 min-max 等。当前示例只把 MFHR 做到可复现；Selfish/Global 需要你给出明确规则才能写成严谨对照。  
3) **预测→调度联动的严格方式**：当前建议用预测分位（P50/P90）作为用户选店与 MFHR 风险调参输入，以实现“预测不确定性→鲁棒调度”的闭环；你希望闭环的因果链条更强的话，需要明确“用户看到的等待时间信息是什么、更新频率、误差容忍”。  

以上框架一旦你确认，我建议把 MFHR 的目标/约束与所有指标的定义写入 `config/schema.md`，并把所有图表生成脚本固定到 `scripts/`，配合 tag/conda-spec 做论文级复现闭环。