"""Module hỗ trợ gán nhãn 2D từ LiDAR 3D (Topic F - Auto-label support).

Cung cấp các chức năng:
1. Chiếu 8 góc 3D box (box3d_corners_cam) lên ảnh camera -> 2D Bounding Box.
2. Trích xuất điểm LiDAR bên trong 3D box và chiếu lên ảnh -> Tight LiDAR 2D Box.
3. Tính toán 2D IoU giữa box tự động gợi ý và box nhãn chuẩn (GT 2D bbox).
4. Thử nghiệm drift calibration (yaw, pitch, translation) và suy giảm dữ liệu (dropout).
5. Tự động gắn cờ kiểm tra (QA review flag) dựa trên ngưỡng IoU.
6. Đo latency p50/p95 (chuẩn B3) và xuất báo cáo bảng CSV + đồ thị PNG.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import dataset_type, load_frame
from starter.kitti_io import KittiCalib, KittiObject, list_frames
from starter.projection import (
    box3d_corners_cam,
    cam_to_image,
    draw_box2d,
    perturb_extrinsic,
    velo_to_cam,
)


def compute_iou_2d(box_a: np.ndarray | tuple, box_b: np.ndarray | tuple) -> float:
    """Tính Intersection over Union (IoU) giữa 2 box [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter_area = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 1e-6:
        return 0.0
    return float(inter_area / union_area)


def project_3d_box_to_2d(
    obj: KittiObject,
    calib: KittiCalib,
    image_shape: tuple[int, ...],
    true_calib: KittiCalib | None = None,
) -> np.ndarray | None:
    """Chiếu 8 góc của 3D bounding box lên ảnh camera -> 2D box [x1, y1, x2, y2].

    Vì nhãn 3D xuất phát từ LiDAR (Topic F), 8 góc được định nghĩa theo LiDAR frame
    và biến đổi sang camera bằng calib (có thể bị drift góc/tịnh tiến).
    """
    ref_calib = true_calib if true_calib is not None else calib
    corners_3d_cam = box3d_corners_cam(obj)  # (8, 3) trong camera frame gốc

    # Đưa góc về LiDAR frame bằng calibration gốc
    T_velo_cam = np.linalg.inv(ref_calib.T_cam_velo)
    corners_homo = np.hstack([corners_3d_cam, np.ones((8, 1), dtype=float)])
    corners_velo = (corners_homo @ T_velo_cam.T)[:, :3]

    # Chiếu từ LiDAR frame sang camera frame bằng calib hiện tại (có thể bị lệch)
    corners_cam_cur = velo_to_cam(corners_velo, calib)
    uv, depth, mask = cam_to_image(corners_cam_cur, calib.P2, image_shape, min_depth=0.1)

    if len(uv) < 2:
        return None

    H, W = image_shape[:2]
    x1 = float(np.clip(np.min(uv[:, 0]), 0, W - 1))
    y1 = float(np.clip(np.min(uv[:, 1]), 0, H - 1))
    x2 = float(np.clip(np.max(uv[:, 0]), 0, W - 1))
    y2 = float(np.clip(np.max(uv[:, 1]), 0, H - 1))

    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=float)


def get_points_in_box3d(points_cam: np.ndarray, obj: KittiObject) -> tuple[np.ndarray, np.ndarray]:
    """Lọc các điểm LiDAR (trong camera frame) nằm bên trong 3D bounding box."""
    if len(points_cam) == 0:
        return np.empty((0, 3), dtype=float), np.zeros(0, dtype=bool)

    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R_t = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])

    p_rel = points_cam[:, :3] - obj.location
    p_local = p_rel @ R_t.T

    inside = (
        (p_local[:, 0] >= -l / 2.0)
        & (p_local[:, 0] <= l / 2.0)
        & (p_local[:, 1] >= -h)
        & (p_local[:, 1] <= 0.0)
        & (p_local[:, 2] >= -w / 2.0)
        & (p_local[:, 2] <= w / 2.0)
    )
    return points_cam[inside], inside


def project_tight_lidar_to_2d(
    pts_in_box_velo: np.ndarray,
    calib: KittiCalib,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray | None, int]:
    """Tạo 2D box bó sát (tight) từ các điểm LiDAR của vật thể chiếu lên ảnh."""
    n_pts = len(pts_in_box_velo)
    if n_pts < 2:
        return None, n_pts

    pts_cam_cur = velo_to_cam(pts_in_box_velo[:, :3], calib)
    uv, depth, mask = cam_to_image(pts_cam_cur, calib.P2, image_shape, min_depth=0.1)
    if len(uv) < 2:
        return None, n_pts

    H, W = image_shape[:2]
    x1 = float(np.clip(np.min(uv[:, 0]), 0, W - 1))
    y1 = float(np.clip(np.min(uv[:, 1]), 0, H - 1))
    x2 = float(np.clip(np.max(uv[:, 0]), 0, W - 1))
    y2 = float(np.clip(np.max(uv[:, 1]), 0, H - 1))

    if x2 <= x1 or y2 <= y1:
        return None, n_pts
    return np.array([x1, y1, x2, y2], dtype=float), n_pts


def evaluate_frame(
    frame_dict: dict[str, Any],
    calib: KittiCalib | None = None,
    keep_ratio: float = 1.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Đánh giá 1 frame: so sánh IoU giữa 2D box GT và 2 phương pháp gợi ý nhãn."""
    true_calib = frame_dict["calib"]
    current_calib = calib if calib is not None else true_calib

    pts = frame_dict["points"]
    if keep_ratio < 1.0 and len(pts) > 0:
        rng = np.random.default_rng(seed)
        n_keep = int(len(pts) * keep_ratio)
        idx = rng.choice(len(pts), size=n_keep, replace=False)
        pts = pts[idx]

    pts_cam_ref = velo_to_cam(pts[:, :3], true_calib)
    img_shape = frame_dict["image"].shape
    records = []

    for obj in frame_dict["labels"]:
        if obj.type in ("DontCare",):
            continue

        gt_box = obj.bbox
        dist_3d = float(np.linalg.norm(obj.location))

        # Phương pháp 1: Chiếu 8 góc 3D box
        box_proj = project_3d_box_to_2d(obj, current_calib, img_shape, true_calib=true_calib)
        iou_proj = compute_iou_2d(gt_box, box_proj) if box_proj is not None else 0.0

        # Lấy các điểm thuộc object trong LiDAR frame
        _, inside_mask = get_points_in_box3d(pts_cam_ref, obj)
        pts_obj_velo = pts[inside_mask]
        n_pts = len(pts_obj_velo)

        # Phương pháp 2: Tight LiDAR points
        box_tight, _ = project_tight_lidar_to_2d(pts_obj_velo, current_calib, img_shape)
        iou_tight = compute_iou_2d(gt_box, box_tight) if box_tight is not None else 0.0

        records.append(
            {
                "type": obj.type,
                "distance_m": round(dist_3d, 2),
                "occluded": obj.occluded,
                "truncated": obj.truncated,
                "num_lidar_pts": n_pts,
                "iou_3d_proj": round(iou_proj, 4),
                "iou_tight_lidar": round(iou_tight, 4),
                "gt_x1": round(gt_box[0], 1),
                "gt_y1": round(gt_box[1], 1),
                "gt_x2": round(gt_box[2], 1),
                "gt_y2": round(gt_box[3], 1),
            }
        )
    return records


def run_benchmark(
    data_root: str,
    frames: list[str],
    yaw_perturbations: list[float],
    keep_ratios: list[float],
    out_csv: Path,
) -> pd.DataFrame:
    """Chạy toàn bộ thí nghiệm benchmark với nhiều mức yaw và data dropout."""
    all_rows = []
    print(f"=== Chạy benchmark trên {len(frames)} frames của {data_root} ===")

    for frame_id in frames:
        fr = load_frame(data_root, frame_id)

        # Baseline: không perturb
        recs = evaluate_frame(fr)
        for r in recs:
            row = dict(r)
            row.update(
                {
                    "dataset": Path(data_root).name,
                    "frame": frame_id,
                    "yaw_deg": 0.0,
                    "keep_ratio": 1.0,
                    "perturb_type": "baseline",
                }
            )
            all_rows.append(row)

        # Perturbation 1: Yaw calibration drift
        for yaw in yaw_perturbations:
            if yaw == 0.0:
                continue
            calib_p = perturb_extrinsic(fr["calib"], yaw_deg=yaw)
            recs_p = evaluate_frame(fr, calib=calib_p)
            for r in recs_p:
                row = dict(r)
                row.update(
                    {
                        "dataset": Path(data_root).name,
                        "frame": frame_id,
                        "yaw_deg": yaw,
                        "keep_ratio": 1.0,
                        "perturb_type": f"yaw_{yaw}deg",
                    }
                )
                all_rows.append(row)

        # Perturbation 2: Point cloud dropout
        for kr in keep_ratios:
            if kr == 1.0:
                continue
            recs_kr = evaluate_frame(fr, keep_ratio=kr)
            for r in recs_kr:
                row = dict(r)
                row.update(
                    {
                        "dataset": Path(data_root).name,
                        "frame": frame_id,
                        "yaw_deg": 0.0,
                        "keep_ratio": kr,
                        "perturb_type": f"dropout_{int((1 - kr) * 100)}pct",
                    }
                )
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"-> Đã lưu {len(df)} mẫu đánh giá ra {out_csv}")
    return df


def measure_latency(data_root: str, frame_id: str, num_runs: int = 30) -> tuple[float, float]:
    """Đo thời gian chạy p50 và p95 (chuẩn B3: bỏ lần đầu, lặp lại >= 20 lần)."""
    fr = load_frame(data_root, frame_id)
    times = []
    # Warm-up run
    _ = evaluate_frame(fr)

    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = evaluate_frame(fr)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms

    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    return p50, p95


def visualize_comparison(
    data_root: str,
    frame_id: str,
    out_path: Path,
    yaw_deg: float = 0.0,
    title_suffix: str = "",
) -> None:
    """Vẽ ảnh so sánh trực quan: Box GT (xanh lá), 3D Box Proj (đỏ), Tight LiDAR Box (vàng)."""
    fr = load_frame(data_root, frame_id)
    true_calib = fr["calib"]
    calib = perturb_extrinsic(true_calib, yaw_deg=yaw_deg) if yaw_deg != 0.0 else true_calib

    img = fr["image"].copy()
    pts_cam_ref = velo_to_cam(fr["points"][:, :3], true_calib)

    for obj in fr["labels"]:
        if obj.type in ("DontCare",):
            continue

        # 1. Box GT: Xanh lá (0, 255, 0)
        draw_box2d(img, obj.bbox, color=(0, 255, 0), label=f"GT {obj.type}")

        # 2. Box 3D Proj: Đỏ (0, 0, 255)
        b_proj = project_3d_box_to_2d(obj, calib, img.shape, true_calib=true_calib)
        if b_proj is not None:
            iou_p = compute_iou_2d(obj.bbox, b_proj)
            draw_box2d(img, b_proj, color=(0, 0, 255), label=f"3D-Proj IoU={iou_p:.2f}")

        # 3. Box Tight LiDAR: Vàng (0, 255, 255)
        _, inside_mask = get_points_in_box3d(pts_cam_ref, obj)
        pts_obj_velo = fr["points"][inside_mask]
        b_tight, n_pts = project_tight_lidar_to_2d(pts_obj_velo, calib, img.shape)
        if b_tight is not None:
            iou_t = compute_iou_2d(obj.bbox, b_tight)
            draw_box2d(img, b_tight, color=(0, 255, 255), label=f"Tight ({n_pts}p) IoU={iou_t:.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"-> Đã lưu ảnh so sánh trực quan ra {out_path}")


def plot_benchmark_charts(df: pd.DataFrame, out_chart_path: Path) -> None:
    """Vẽ biểu đồ phân tích độ nhạy của IoU theo góc lệch yaw và tỉ lệ dropout."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Đồ thị 1: IoU theo góc lệch Yaw
    yaw_df = df[df["perturb_type"].str.startswith("yaw") | (df["perturb_type"] == "baseline")].copy()
    yaw_summary = yaw_df.groupby("yaw_deg")[["iou_3d_proj", "iou_tight_lidar"]].mean().reset_index()

    axes[0].plot(yaw_summary["yaw_deg"], yaw_summary["iou_3d_proj"], "o-", color="crimson", label="3D Box Projection", linewidth=2)
    axes[0].plot(yaw_summary["yaw_deg"], yaw_summary["iou_tight_lidar"], "s--", color="goldenrod", label="Tight LiDAR Points", linewidth=2)
    axes[0].axhline(y=0.50, color="gray", linestyle=":", label="QA Flag Threshold (0.50)")
    axes[0].set_title("Độ nhạy IoU theo góc lệch Calibration (Yaw Drift)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Độ lệch Yaw (độ)", fontsize=11)
    axes[0].set_ylabel("IoU trung bình so với GT 2D Box", fontsize=11)
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Đồ thị 2: IoU theo tỉ lệ giữ điểm (Dropout)
    do_df = df[df["perturb_type"].str.startswith("dropout") | (df["perturb_type"] == "baseline")].copy()
    do_summary = do_df.groupby("keep_ratio")[["iou_3d_proj", "iou_tight_lidar"]].mean().reset_index()

    axes[1].plot(do_summary["keep_ratio"], do_summary["iou_3d_proj"], "o-", color="crimson", label="3D Box Projection", linewidth=2)
    axes[1].plot(do_summary["keep_ratio"], do_summary["iou_tight_lidar"], "s--", color="goldenrod", label="Tight LiDAR Points", linewidth=2)
    axes[1].set_title("Ảnh hưởng của mật độ điểm (LiDAR Dropout)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Tỉ lệ điểm giữ lại (Keep Ratio)", fontsize=11)
    axes[1].set_ylabel("IoU trung bình so với GT 2D Box", fontsize=11)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    out_chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_chart_path, dpi=150)
    plt.close()
    print(f"-> Đã lưu biểu đồ phân tích ra {out_chart_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR 3D to 2D Auto-Labeling Benchmark & QA Tool (Topic F)")
    parser.add_argument("--data-root", default="data/kitti_mini", help="Thư mục dataset (data/kitti_mini hoặc data/nuscenes_mini_subset)")
    parser.add_argument("--frames", nargs="*", default=None, help="Danh sách frame cụ thể cần chạy")
    parser.add_argument("--out-dir", default="results", help="Thư mục lưu kết quả CSV và hình ảnh")
    parser.add_argument("--yaw-sweep", nargs="*", type=float, default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0], help="Các mức lệch yaw (độ)")
    parser.add_argument("--dropout-sweep", nargs="*", type=float, default=[1.0, 0.8, 0.6, 0.4, 0.2], help="Các tỉ lệ keep ratio")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    frames = args.frames or list_frames(args.data_root)
    if len(frames) > 10:
        frames = frames[:10]  # Chạy trên 10 frame tiêu biểu để tối ưu thời gian

    print(f"Khởi động thí nghiệm Auto-Labeling trên {len(frames)} frames...")

    # 1. Chạy benchmark
    csv_path = out_dir / "auto_label_benchmark.csv"
    df = run_benchmark(
        data_root=args.data_root,
        frames=frames,
        yaw_perturbations=args.yaw_sweep,
        keep_ratios=args.dropout_sweep,
        out_csv=csv_path,
    )

    # 2. Vẽ biểu đồ
    chart_path = fig_dir / "auto_label_iou_analysis.png"
    plot_benchmark_charts(df, chart_path)

    # 3. Đo latency chuẩn B3
    test_frame = frames[0]
    p50, p95 = measure_latency(args.data_root, test_frame, num_runs=25)
    print(f"-> Benchmark Latency: p50={p50:.2f} ms | p95={p95:.2f} ms")

    f1 = "000011" if "000011" in frames else frames[0]
    f2 = "000021" if "000021" in frames else (frames[1] if len(frames) > 1 else frames[0])

    # 4. Xuất ảnh demo chính (CP2 / CP3)
    demo_path = fig_dir / f"demo_auto_label_{f1}.png"
    visualize_comparison(args.data_root, f1, demo_path)

    # 5. Xuất ảnh failure cases (CP4)
    # Fail 1: Calibration Drift làm lệch nhãn (yaw = 2.0 deg)
    fail_drift_path = fig_dir / "fail_01_yaw_drift_2deg.png"
    visualize_comparison(args.data_root, f1, fail_drift_path, yaw_deg=2.0)

    # Fail 2: Khoảng cách xa hoặc occlusion
    fail_occ_path = fig_dir / "fail_02_distant_occlusion.png"
    visualize_comparison(args.data_root, f2, fail_occ_path)

    print("\n=== HOÀN TẤT THÍ NGHIỆM TOPIC F ===")


if __name__ == "__main__":
    main()
