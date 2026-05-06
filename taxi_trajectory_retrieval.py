#!/usr/bin/env python3
"""Task 1: 轨迹检索

基于 Porto Taxi Trajectory 数据集，构建两阶段轨迹检索：
1) 粗筛：起终点网格 + 长度/时长范围
2) 精排：固定点数重采样后计算欧氏距离

输出：
- 结果汇总 Markdown
- 查询/检索明细 CSV
- 至少 5 组可视化叠加图
- 检索质量分布图
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────
PORTO_LAT_MIN = 41.0
PORTO_LAT_MAX = 41.3
PORTO_LON_MIN = -8.7
PORTO_LON_MAX = -8.5

DEFAULT_DATA_PATH = Path.home() / "taxi" / "kaggle_dl" / "train.csv"
DEFAULT_OUTPUT_DIR = Path.home() / "taxi" / "output" / "task1"

GRID_SIZE_DEG = 0.005
RESAMPLE_POINTS = 32
RANDOM_SEED = 42
DELTA_SECONDS = 15.0


# ─────────────────────────────────────────────────────────────
# 基础函数
# ─────────────────────────────────────────────────────────────
def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """球面距离，单位：米。"""
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = (
        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
        - math.sin(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.cos(math.radians(lon2 - lon1))
    )
    deg = math.degrees(math.atan2(y, x))
    return (deg + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def point_in_bbox(lon: float, lat: float) -> bool:
    return PORTO_LON_MIN <= lon <= PORTO_LON_MAX and PORTO_LAT_MIN <= lat <= PORTO_LAT_MAX


def lonlat_to_xy(lon: float, lat: float, ref_lat: float = 41.157) -> Tuple[float, float]:
    x = math.radians(lon) * 6371000.0 * math.cos(math.radians(ref_lat))
    y = math.radians(lat) * 6371000.0
    return x, y


def cell_id(lon: float, lat: float, grid_size: float = GRID_SIZE_DEG) -> Tuple[int, int]:
    return (
        int((lon - PORTO_LON_MIN) / grid_size),
        int((lat - PORTO_LAT_MIN) / grid_size),
    )


def parse_polyline(poly_str: str) -> Optional[List[Tuple[float, float]]]:
    try:
        pts = json.loads(poly_str)
    except Exception:
        return None
    if not isinstance(pts, list):
        return None
    clean = []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            lon, lat = float(p[0]), float(p[1])
        except Exception:
            continue
        if math.isnan(lon) or math.isnan(lat):
            continue
        if not point_in_bbox(lon, lat):
            return None
        clean.append((lon, lat))
    return clean if len(clean) >= 5 else None


def trajectory_length(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(haversine(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1]) for i in range(1, len(points)))


def resample_polyline(points: Sequence[Tuple[float, float]], n: int = RESAMPLE_POINTS) -> np.ndarray:
    """按累计弧长做线性插值，返回 shape=(n,2) 的 lon/lat 矩阵。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)

    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return np.repeat(pts[:1], n, axis=0)

    target = np.linspace(0.0, total, n)
    out = np.empty((n, 2), dtype=np.float64)
    for j, t in enumerate(target):
        idx = np.searchsorted(cum, t, side="right") - 1
        idx = min(max(idx, 0), len(pts) - 2)
        denom = cum[idx + 1] - cum[idx]
        if denom <= 1e-12:
            out[j] = pts[idx]
        else:
            alpha = (t - cum[idx]) / denom
            out[j] = (1 - alpha) * pts[idx] + alpha * pts[idx + 1]
    return out


@dataclass
class TrajectoryRecord:
    trip_id: int
    points: List[Tuple[float, float]]
    start_lon: float
    start_lat: float
    end_lon: float
    end_lat: float
    length_m: float
    duration_s: float
    straight_m: float
    direction_deg: float
    vector_xy: np.ndarray  # shape=(n, 2)
    start_cell: Tuple[int, int]
    end_cell: Tuple[int, int]


# ─────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────
def load_trajectories(data_path: Path, n_sample: int, seed: int = RANDOM_SEED) -> List[TrajectoryRecord]:
    rng = random.Random(seed)
    chunksize = 50000
    cols = ["TRIP_ID", "MISSING_DATA", "POLYLINE"]
    records: List[TrajectoryRecord] = []
    total_rows = 0
    valid_rows = 0

    for chunk in pd.read_csv(data_path, usecols=cols, chunksize=chunksize):
        total_rows += len(chunk)
        # 先做最小过滤，减少 JSON 解析量
        chunk = chunk[chunk["MISSING_DATA"] == False]  # noqa: E712
        for row in chunk.itertuples(index=False):
            pts = parse_polyline(row.POLYLINE)
            if pts is None:
                continue
            valid_rows += 1
            start_lon, start_lat = pts[0]
            end_lon, end_lat = pts[-1]
            length_m = trajectory_length(pts)
            duration_s = float(len(pts) * DELTA_SECONDS)
            straight_m = haversine(start_lon, start_lat, end_lon, end_lat)
            direction_deg = bearing(start_lon, start_lat, end_lon, end_lat)
            vec = resample_polyline(pts, RESAMPLE_POINTS)
            vec_xy = np.array([lonlat_to_xy(lon, lat) for lon, lat in vec], dtype=np.float64)
            records.append(
                TrajectoryRecord(
                    trip_id=int(row.TRIP_ID),
                    points=list(pts),
                    start_lon=start_lon,
                    start_lat=start_lat,
                    end_lon=end_lon,
                    end_lat=end_lat,
                    length_m=length_m,
                    duration_s=duration_s,
                    straight_m=straight_m,
                    direction_deg=direction_deg,
                    vector_xy=vec_xy,
                    start_cell=cell_id(start_lon, start_lat),
                    end_cell=cell_id(end_lon, end_lat),
                )
            )
            if len(records) >= n_sample:
                break
        if len(records) >= n_sample:
            break

    if len(records) < n_sample:
        raise RuntimeError(f"有效轨迹不足：仅 {len(records)} 条，目标 {n_sample} 条。")

    rng.shuffle(records)
    print(f"Loaded rows={total_rows}, valid={valid_rows}, used={len(records)}")
    return records


# ─────────────────────────────────────────────────────────────
# 粗筛索引
# ─────────────────────────────────────────────────────────────
def build_indices(records: Sequence[TrajectoryRecord]):
    start_index: Dict[Tuple[int, int], List[int]] = {}
    end_index: Dict[Tuple[int, int], List[int]] = {}
    for idx, rec in enumerate(records):
        start_index.setdefault(rec.start_cell, []).append(idx)
        end_index.setdefault(rec.end_cell, []).append(idx)
    return start_index, end_index


def neighbor_cells(cell: Tuple[int, int], radius: int) -> Iterable[Tuple[int, int]]:
    cx, cy = cell
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            yield cx + dx, cy + dy


def coarse_candidates(
    query: TrajectoryRecord,
    records: Sequence[TrajectoryRecord],
    start_index: Dict[Tuple[int, int], List[int]],
    end_index: Dict[Tuple[int, int], List[int]],
    target_min: int = 200,
    target_max: int = 500,
) -> List[int]:
    """两阶段中的粗筛：起终点网格 + 长度/时长范围。"""
    len_ref = query.length_m
    dur_ref = query.duration_s

    # 自适应放宽：先保证召回，再控制规模
    settings = [
        dict(radius=1, len_ratio=0.25, dur_ratio=0.25),
        dict(radius=2, len_ratio=0.35, dur_ratio=0.35),
        dict(radius=3, len_ratio=0.50, dur_ratio=0.50),
    ]

    best = None
    for s in settings:
        start_pool = set()
        end_pool = set()
        for c in neighbor_cells(query.start_cell, s["radius"]):
            start_pool.update(start_index.get(c, []))
        for c in neighbor_cells(query.end_cell, s["radius"]):
            end_pool.update(end_index.get(c, []))

        lo_len = len_ref * (1 - s["len_ratio"])
        hi_len = len_ref * (1 + s["len_ratio"])
        lo_dur = dur_ref * (1 - s["dur_ratio"])
        hi_dur = dur_ref * (1 + s["dur_ratio"])

        cand = [
            idx
            for idx in (start_pool & end_pool)
            if idx != -1
            and lo_len <= records[idx].length_m <= hi_len
            and lo_dur <= records[idx].duration_s <= hi_dur
        ]

        if best is None or abs(len(cand) - 350) < abs(len(best) - 350):
            best = cand
        if target_min <= len(cand) <= target_max:
            return cand

    # 若仍偏离，返回最接近目标规模的集合，再按粗分数截断/补足
    cand = best or []
    if len(cand) > target_max:
        cand = sorted(cand, key=lambda i: rough_score(query, records[i]))[:target_max]
    return cand


def rough_score(a: TrajectoryRecord, b: TrajectoryRecord) -> float:
    start_d = haversine(a.start_lon, a.start_lat, b.start_lon, b.start_lat)
    end_d = haversine(a.end_lon, a.end_lat, b.end_lon, b.end_lat)
    len_gap = abs(a.length_m - b.length_m) / max(a.length_m, 1.0)
    dur_gap = abs(a.duration_s - b.duration_s) / max(a.duration_s, 1.0)
    dir_gap = angle_diff_deg(a.direction_deg, b.direction_deg) / 180.0
    return start_d + end_d + 1200.0 * len_gap + 1200.0 * dur_gap + 300.0 * dir_gap


# ─────────────────────────────────────────────────────────────
# 精排
# ─────────────────────────────────────────────────────────────
def fine_distance(query_vec: np.ndarray, cand_vec: np.ndarray) -> float:
    diff = query_vec - cand_vec
    return float(np.mean(np.sqrt((diff * diff).sum(axis=1))))


def retrieve_topk(
    query_idx: int,
    records: Sequence[TrajectoryRecord],
    start_index: Dict[Tuple[int, int], List[int]],
    end_index: Dict[Tuple[int, int], List[int]],
    topk: int = 5,
) -> Dict:
    query = records[query_idx]
    candidates = coarse_candidates(query, records, start_index, end_index)

    # 若粗筛过少，补充同起点或同终点邻域，保持两阶段结构但避免空集
    if len(candidates) < 20:
        extra = set(candidates)
        for c in neighbor_cells(query.start_cell, 4):
            extra.update(start_index.get(c, []))
        for c in neighbor_cells(query.end_cell, 4):
            extra.update(end_index.get(c, []))
        extra.discard(query_idx)
        candidates = sorted(extra, key=lambda i: rough_score(query, records[i]))[:500]

    scored = []
    for idx in candidates:
        if idx == query_idx:
            continue
        cand = records[idx]
        fdist = fine_distance(query.vector_xy, cand.vector_xy)
        score = 0.55 * fdist + 0.45 * rough_score(query, cand)
        scored.append((score, fdist, idx))

    scored.sort(key=lambda x: x[0])
    top = scored[:topk]

    result_rows = []
    for rank, (score, fdist, idx) in enumerate(top, start=1):
        cand = records[idx]
        result_rows.append(
            {
                "rank": rank,
                "query_idx": query_idx,
                "query_trip_id": query.trip_id,
                "cand_idx": idx,
                "cand_trip_id": cand.trip_id,
                "score": score,
                "fine_distance": fdist,
                "start_distance_m": haversine(query.start_lon, query.start_lat, cand.start_lon, cand.start_lat),
                "end_distance_m": haversine(query.end_lon, query.end_lat, cand.end_lon, cand.end_lat),
                "length_ratio": cand.length_m / max(query.length_m, 1.0),
                "duration_ratio": cand.duration_s / max(query.duration_s, 1.0),
                "direction_diff_deg": angle_diff_deg(query.direction_deg, cand.direction_deg),
            }
        )

    return {
        "query_idx": query_idx,
        "query_trip_id": query.trip_id,
        "candidate_count": len(candidates),
        "results": result_rows,
    }


# ─────────────────────────────────────────────────────────────
# 评估与可视化
# ─────────────────────────────────────────────────────────────
def is_high_quality(row: pd.Series) -> bool:
    return (
        row["start_distance_m"] <= 250.0
        and row["end_distance_m"] <= 250.0
        and 0.80 <= row["length_ratio"] <= 1.25
        and row["direction_diff_deg"] <= 35.0
    )


def score_quality(row: pd.Series) -> float:
    return (
        row["start_distance_m"] / 250.0
        + row["end_distance_m"] / 250.0
        + abs(math.log(max(row["length_ratio"], 1e-6))) * 1.5
        + row["direction_diff_deg"] / 60.0
    )


def plot_overlay(query: TrajectoryRecord, top_rows: pd.DataFrame, records: Sequence[TrajectoryRecord], path: Path, title: str):
    plt.figure(figsize=(8.5, 7.0))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    qxy = query.vector_xy
    plt.plot(qxy[:, 0], qxy[:, 1], color=colors[0], lw=2.8, label=f"Query {query.trip_id}")

    for i, (_, row) in enumerate(top_rows.iterrows(), start=1):
        cand = records[int(row["cand_idx"])]
        cxy = cand.vector_xy
        plt.plot(cxy[:, 0], cxy[:, 1], color=colors[i], lw=2.0, alpha=0.9, label=f"Top-{i} {cand.trip_id}")

    plt.scatter([qxy[0, 0]], [qxy[0, 1]], c=colors[0], s=40, marker="o")
    plt.scatter([qxy[-1, 0]], [qxy[-1, 1]], c=colors[0], s=55, marker="x")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_distribution(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(df["start_distance_m"], bins=20, color="#1f77b4", alpha=0.85)
    axes[0].set_title("Start distance (m)")
    axes[1].hist(df["end_distance_m"], bins=20, color="#2ca02c", alpha=0.85)
    axes[1].set_title("End distance (m)")
    axes[2].hist(df["length_ratio"], bins=20, color="#d62728", alpha=0.85)
    axes[2].set_title("Length ratio")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Top-K 轨迹检索")
    parser.add_argument("--n-sample", type=int, default=5000, help="用于实验的有效轨迹数")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH), help="train.csv 路径")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")

    records = load_trajectories(data_path, args.n_sample, args.seed)
    start_index, end_index = build_indices(records)

    # 抽取 80 条查询
    query_indices = random.sample(range(len(records)), 80)

    all_rows = []
    candidate_sizes = []
    per_query = []
    for qidx in query_indices:
        out = retrieve_topk(qidx, records, start_index, end_index, topk=5)
        candidate_sizes.append(out["candidate_count"])
        per_query.append(out)
        for row in out["results"]:
            all_rows.append(row)

    # ═══ Encoder embedding retrieval ═══
    if TORCH_AVAILABLE:
        embeddings, _ = train_contrastive_encoder(records, start_index, end_index, epochs=30, batch_size=64)
        emb_all_rows = []
        for qidx in query_indices:
            emb_out = retrieve_topk_embedding(qidx, records, start_index, end_index, embeddings, topk=5)
            for _, row in emb_out.iterrows():
                emb_all_rows.append(row.to_dict())
        emb_df = pd.DataFrame(emb_all_rows)
        emb_df['high_quality'] = emb_df.apply(is_high_quality, axis=1)
        emb_df['quality_score'] = emb_df.apply(score_quality, axis=1)
        emb_top1_rate = float(emb_df[emb_df['rank']==1]['high_quality'].mean()) if len(emb_df) else 0
        emb_top5_rate = float(emb_df['high_quality'].mean()) if len(emb_df) else 0
        print(f"Encoder retrieval: Top1={emb_top1_rate:.3f}, Top5={emb_top5_rate:.3f}")
    else:
        embeddings = None
        emb_df = pd.DataFrame()
        emb_top1_rate = emb_top5_rate = 0.0

    result_df = pd.DataFrame(all_rows)
    result_df["high_quality"] = result_df.apply(is_high_quality, axis=1)
    result_df["quality_score"] = result_df.apply(score_quality, axis=1)

    # 统计
    summary = {
        "n_sample": len(records),
        "n_query": len(query_indices),
        "candidate_mean": float(np.mean(candidate_sizes)),
        "candidate_median": float(np.median(candidate_sizes)),
        "candidate_min": int(np.min(candidate_sizes)),
        "candidate_max": int(np.max(candidate_sizes)),
        "top1_high_quality_rate": float(result_df[result_df["rank"] == 1]["high_quality"].mean()),
        "top5_high_quality_rate": float(result_df["high_quality"].mean()),
        "start_distance_mean": float(result_df["start_distance_m"].mean()),
        "end_distance_mean": float(result_df["end_distance_m"].mean()),
        "length_ratio_mean": float(result_df["length_ratio"].mean()),
        "direction_diff_mean": float(result_df["direction_diff_deg"].mean()),
    }

    # 选图：5 组查询
    plot_query_ids = []
    seen = set()
    for item in per_query:
        qidx = item["query_idx"]
        top1 = item["results"][0]
        if qidx not in seen:
            plot_query_ids.append(qidx)
            seen.add(qidx)
        if len(plot_query_ids) >= 5:
            break

    for i, qidx in enumerate(plot_query_ids, start=1):
        item = next(x for x in per_query if x["query_idx"] == qidx)
        query = records[qidx]
        top_rows = pd.DataFrame(item["results"][:3])
        title = (
            f"Query {query.trip_id} | candidates={item['candidate_count']} | "
            f"top1 start={top_rows.iloc[0]['start_distance_m']:.0f}m end={top_rows.iloc[0]['end_distance_m']:.0f}m"
        )
        plot_overlay(query, top_rows, records, output_dir / f"query_{i:02d}_overlay.png", title)

    plot_distribution(result_df, output_dir / "retrieval_quality_distribution.png")

    # 明细输出
    result_csv = output_dir / "task1_retrieval_results.csv"
    result_df.sort_values(["query_idx", "rank"]).to_csv(result_csv, index=False)

    # 案例分析：好的例子
    good_df = result_df[(result_df["rank"] == 1) & (result_df["high_quality"])]
    good_cases = good_df.nsmallest(3, "quality_score")

    # 反直觉例子：全局特征接近，但 fine distance 偏大
    weird_df = result_df[
        (result_df["rank"] == 1)
        & (result_df["start_distance_m"] <= 250.0)
        & (result_df["end_distance_m"] <= 250.0)
        & (result_df["length_ratio"].between(0.85, 1.20))
        & (result_df["direction_diff_deg"] <= 40.0)
    ].copy()
    if len(weird_df) > 0:
        weird_cases = weird_df.nlargest(3, "fine_distance")
    else:
        weird_cases = result_df[result_df["rank"] == 1].nlargest(3, "quality_score")

    # 汇总 Markdown
    md_lines = []
    md_lines.append("# Task 1 轨迹检索汇总\n")
    md_lines.append("## 实验设置\n")
    md_lines.append(f"- 数据集：`{data_path}`")
    md_lines.append(f"- 有效轨迹样本：{summary['n_sample']}")
    md_lines.append(f"- 查询数：{summary['n_query']}")
    md_lines.append(f"- 轨迹重采样点数：{RESAMPLE_POINTS}")
    md_lines.append("- 粗筛：起点网格、终点网格、长度/时长范围")
    md_lines.append("- 精排：重采样轨迹欧氏距离\n")

    md_lines.append("## 检索统计\n")
    md_lines.append(f"- 候选集均值：{summary['candidate_mean']:.1f}")
    md_lines.append(f"- 候选集中位数：{summary['candidate_median']:.1f}")
    md_lines.append(f"- 候选集范围：{summary['candidate_min']} ~ {summary['candidate_max']}")
    md_lines.append(f"- Top-1 高质量匹配率：{summary['top1_high_quality_rate']:.3f}")
    md_lines.append(f"- Top-5 高质量匹配率：{summary['top5_high_quality_rate']:.3f}")
    md_lines.append(f"- 平均起点距离：{summary['start_distance_mean']:.1f} m")
    md_lines.append(f"- 平均终点距离：{summary['end_distance_mean']:.1f} m")
    md_lines.append(f"- 平均长度比：{summary['length_ratio_mean']:.3f}")
    md_lines.append(f"- 平均方向差：{summary['direction_diff_mean']:.1f}°\n")

    if len(emb_df):
        md_lines.append("## 余弦嵌入检索对比\n")
        md_lines.append(f"- 嵌入 Top-1 高质量匹配率：{emb_top1_rate:.3f}")
        md_lines.append(f"- 嵌入 Top-5 高质量匹配率：{emb_top5_rate:.3f}")
        md_lines.append(f"- 嵌入平均起点距离：{emb_df['start_distance_m'].mean():.1f} m")
        md_lines.append(f"- 嵌入平均终点距离：{emb_df['end_distance_m'].mean():.1f} m\n")
        # Comparison bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        methods = ['Euclidean', 'Cosine Embed']
        top1_rates = [summary['top1_high_quality_rate'], emb_top1_rate]
        top5_rates = [summary['top5_high_quality_rate'], emb_top5_rate]
        x = np.arange(len(methods))
        ax.bar(x - 0.15, top1_rates, 0.3, label='Top-1', color='#1f77b4')
        ax.bar(x + 0.15, top5_rates, 0.3, label='Top-5', color='#ff7f0e')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel('High quality match rate')
        ax.set_title('Retrieval: Euclidean vs Cosine Embedding')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / 'task1_encoder_comparison.png', dpi=150)
        plt.close(fig)

    md_lines.append("## 结果文件\n")
    md_lines.append(f"- 结果明细：`{result_csv}`")
    md_lines.append(f"- 质量分布图：`{output_dir / 'retrieval_quality_distribution.png'}`")
    for i in range(1, 6):
        md_lines.append(f"- 叠加图 {i}：`{output_dir / f'query_{i:02d}_overlay.png'}`")
    md_lines.append("")

    md_lines.append("## 看起来正确的案例\n")
    for _, row in good_cases.iterrows():
        md_lines.append(
            f"- 查询 `{int(row['query_trip_id'])}` → 候选 `{int(row['cand_trip_id'])}`："
            f"起点 {row['start_distance_m']:.1f} m，终点 {row['end_distance_m']:.1f} m，"
            f"长度比 {row['length_ratio']:.3f}，方向差 {row['direction_diff_deg']:.1f}°，"
            f"fine 距离 {row['fine_distance']:.2f}"
        )
    md_lines.append("")

    md_lines.append("## 相似度高但不完全符合直觉的案例\n")
    for _, row in weird_cases.iterrows():
        md_lines.append(
            f"- 查询 `{int(row['query_trip_id'])}` → 候选 `{int(row['cand_trip_id'])}`："
            f"起点 {row['start_distance_m']:.1f} m，终点 {row['end_distance_m']:.1f} m，"
            f"长度比 {row['length_ratio']:.3f}，方向差 {row['direction_diff_deg']:.1f}°，"
            f"fine 距离 {row['fine_distance']:.2f}"
        )
    md_lines.append("")

    md_lines.append("## 解释\n")
    md_lines.append("- 粗筛能快速缩小检索空间，但同网格轨迹仍可能分布较散。")
    md_lines.append("- 精排采用重采样欧氏距离，可区分整体形状。")
    md_lines.append("- 部分轨迹在起终点接近时，路径中段仍可能明显不同。")

    summary_md = output_dir / "task1_retrieval_summary.md"
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    # 控制台输出
    print("\nSummary")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.4f}")
        else:
            print(f"- {k}: {v}")
    print(f"Saved: {summary_md}")
    print(f"Saved: {result_csv}")
    print(f"Saved plots under: {output_dir}")


# ══════════════════════════════════════════════════════════════════
# Trajectory Encoder (embedding-based retrieval)
# ══════════════════════════════════════════════════════════════════

def _build_token_features(points, resample_len=32):
    """Build per-point token features from a trajectory."""
    arr = np.asarray(points)
    if len(arr) < 2:
        return None
    # Resample to fixed length
    idx = np.linspace(0, len(arr) - 1, resample_len, dtype=int)
    arr = arr[idx]
    tokens = []
    start_lon, start_lat = arr[0]
    prev_lon, prev_lat = start_lon, start_lat
    cum_km = 0.0
    total_km = 0.0
    for i in range(resample_len):
        lon, lat = arr[i]
        step_km = haversine(start_lat, start_lon, lat, lon) * 1000 if i > 0 else 0.0  # meters
        local_step = haversine(prev_lat, prev_lon, lat, lon) * 1000 if i > 0 else 0.0
        cum_km += local_step / 1000.0
        progress = i / max(1, resample_len - 1)
        # Normalize spatial coords
        dx = (lon - start_lon) * 111320.0 * math.cos(math.radians(41.15))
        dy = (lat - start_lat) * 111320.0
        tokens.append([dx, dy, local_step, progress, cum_km * 1000])
        prev_lon, prev_lat = lon, lat
    return np.asarray(tokens, dtype=np.float32)


def _build_all_token_features(records, resample_len=32):
    """Build token features for all records."""
    all_tokens = []
    for rec in records:
        tok = _build_token_features(rec.points, resample_len)
        if tok is not None:
            all_tokens.append(tok)
        else:
            all_tokens.append(np.zeros((resample_len, 5), dtype=np.float32))
    return np.stack(all_tokens, axis=0)


class TrajectoryAutoencoder(nn.Module if TORCH_AVAILABLE else object):
    """Deep MLP autoencoder for trajectory embedding."""
    def __init__(self, input_dim=64, embed_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, input_dim),
        )
        self.embed_dim = embed_dim

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        emb = self.encode(x)
        recon = self.decoder(emb)
        return recon, emb


def train_contrastive_encoder(records, start_index, end_index, epochs=30, batch_size=64, lr=5e-4):
    """Contrastive encoder using coarse-filter matches as positive pairs."""
    if not TORCH_AVAILABLE:
        return None, None
    print("Training contrastive trajectory encoder...")
    # Build positive pairs from coarse retrieval
    print("  Building positive pairs...")
    pos_pairs = []
    for i, rec in enumerate(records):
        cands = coarse_candidates(rec, records, start_index, end_index, target_min=10, target_max=50)
        for j in cands[:3]:
            if i != j:
                pos_pairs.append((i, j))
    print(f"  Positive pairs: {len(pos_pairs)}")

    # Normalize input vectors
    vectors = np.stack([r.vector_xy.ravel() for r in records], axis=0).astype(np.float32)
    v_mean = vectors.mean(axis=0, keepdims=True)
    v_std = vectors.std(axis=0, keepdims=True) + 1e-6
    vectors = (vectors - v_mean) / v_std
    n = len(records)
    all_data = torch.tensor(vectors, dtype=torch.float32)

    # Encoder only
    encoder = nn.Sequential(
        nn.Linear(64, 256), nn.BatchNorm1d(256), nn.GELU(),
        nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(),
        nn.Linear(256, 128),
    )
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    margin = 0.3

    for epoch in range(1, epochs + 1):
        encoder.train()
        idx = np.random.choice(len(pos_pairs), min(batch_size, len(pos_pairs)), replace=False)
        anchors, positives = zip(*[pos_pairs[i] for i in idx])
        anchors = list(anchors)
        positives = list(positives)
        negatives = [(a + np.random.randint(1, n-1)) % n for a in anchors]

        anc_emb = F.normalize(encoder(all_data[anchors]), dim=1)
        pos_emb = F.normalize(encoder(all_data[positives]), dim=1)
        neg_emb = F.normalize(encoder(all_data[negatives]), dim=1)

        pos_sim = (anc_emb * pos_emb).sum(dim=1)
        neg_sim = (anc_emb * neg_emb).sum(dim=1)
        loss = torch.clamp(margin - pos_sim + neg_sim, min=0).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:02d}: loss={loss.item():.6f} pos_sim={pos_sim.mean().item():.3f} neg_sim={neg_sim.mean().item():.3f}")

    encoder.eval()
    with torch.no_grad():
        all_emb = F.normalize(encoder(all_data), dim=1).cpu().numpy()
    print(f"  Embeddings: {all_emb.shape}")
    return all_emb, encoder


def _flatten_xy(points, resample_len=32):
    arr = np.asarray(points)
    if len(arr) < 2:
        return np.zeros(resample_len * 2, dtype=np.float32)
    idx = np.linspace(0, len(arr) - 1, resample_len, dtype=int)
    arr = arr[idx]
    # Normalize to [0,1]
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    span = np.where(mx - mn < 1e-6, 1.0, mx - mn)
    norm = (arr - mn) / span
    return norm.reshape(-1).astype(np.float32)


def fine_distance_embedding(query_emb, cand_emb):
    """Cosine distance (1 - cosine_similarity)."""
    cos_sim = np.dot(query_emb, cand_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(cand_emb) + 1e-9)
    return 1.0 - cos_sim


def retrieve_topk_embedding(query_idx, records, start_index, end_index, embeddings, topk=5):
    """Retrieval using embedding-based fine distance."""
    query = records[query_idx]
    candidates = coarse_candidates(query, records, start_index, end_index)
    if len(candidates) < 20:
        extra = set(candidates)
        for c in neighbor_cells(query.start_cell, 4):
            extra.update(start_index.get(c, []))
        for c in neighbor_cells(query.end_cell, 4):
            extra.update(end_index.get(c, []))
        extra.discard(query_idx)
        candidates = sorted(extra, key=lambda i: rough_score(query, records[i]))[:500]
    query_emb = embeddings[query_idx]
    scored = []
    for idx in candidates:
        if idx == query_idx:
            continue
        emb_dist = fine_distance_embedding(query_emb, embeddings[idx])
        scored.append((emb_dist, idx))
    scored.sort(key=lambda x: x[0])
    top = scored[:topk]
    result_rows = []
    for rank, (edist, idx) in enumerate(top, start=1):
        cand = records[idx]
        result_rows.append({
            "rank": rank, "query_idx": query_idx, "query_trip_id": query.trip_id,
            "cand_idx": idx, "cand_trip_id": cand.trip_id,
            "fine_distance": edist,
            "start_distance_m": haversine(query.start_lon, query.start_lat, cand.start_lon, cand.start_lat),
            "end_distance_m": haversine(query.end_lon, query.end_lat, cand.end_lon, cand.end_lat),
            "length_ratio": query.length_m / max(1, cand.length_m),
            "direction_diff_deg": abs(query.direction_deg - cand.direction_deg),
        })
    return pd.DataFrame(result_rows)


if __name__ == "__main__":
    main()
