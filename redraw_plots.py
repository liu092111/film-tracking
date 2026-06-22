# -*- coding: utf-8 -*-
"""
重繪 IMG_7129 / IMG_7110 的分析圖。

原則（事實忠實）：
- 所有「數值曲線」一律取自既有的 *_pos_angle_speed.csv，不重新做追蹤分析，確保數據與原結果完全一致。
- 只有 trajectory 圖的 8 個「虛線方框」需要逐幀的 minAreaRect 頂點，CSV 沒有保存，
  因此僅針對方框從影片重新偵測（方框純為視覺呈現，不參與任何數值）。
    * 兩支影片都使用各自的 tracked.mp4（解析度 720x790 / 948x720）來偵測方框。
      注意：CSV 是從這些 tracked.mp4 同解析度的影片產生的；
      film_success/IMG_7129.MOV 是 428x640 的不同解析度版本，像素座標對不上，故不採用。
      tracked.mp4 偵測中心經驗證與 CSV 像素座標吻合（誤差 <1px）。

套用的修改需求：
  1. 所有圖移除 grid
  2. IMG_7129 的 speed 與 orientation 拆成兩張獨立圖（不再是 subplot）
  3. trajectory 圖移除標題
  4. IMG_7129 位置/軌跡圖：Y 軸最上方為正值（不為負）
  5. position 圖標題改為 "Position"
"""

import os
import importlib.util
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = os.path.dirname(os.path.abspath(__file__))


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(BASE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _load("filming_straight_and_speed", "filming_straight_and_speed.py")
R = _load("filming_rotation_and_speed", "filming_rotation_and_speed.py")


def detect_boxes(video_path, detector, n_expected):
    """逐幀偵測 minAreaRect 方框頂點，回傳 list[(frame_idx, box or None)]。"""
    cap = cv2.VideoCapture(video_path)
    boxes = []
    idx = 0
    prev_angle = None
    use_prev = "prev_angle" in detector.__code__.co_varnames
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        meas = detector(frame, prev_angle=prev_angle) if use_prev else detector(frame)
        if meas is not None:
            box = meas[8]
            boxes.append((idx, box))
            ang = meas[6]
            if use_prev and ang is not None and np.isfinite(ang):
                prev_angle = ang
        else:
            boxes.append((idx, None))
        idx += 1
    cap.release()
    if idx != n_expected:
        print(f"[警告] {os.path.basename(video_path)} 幀數 {idx} 與 CSV 列數 {n_expected} 不符")
    return boxes


# =============================================================
# IMG_7129：直走型（filming_straight_and_speed.py）
# =============================================================
def redraw_7129():
    out_dir = os.path.join(BASE, "IMG_7129_analysis_output")
    csv_path = os.path.join(out_dir, "IMG_7129_pos_angle_speed.csv")
    # 方框來源：與 CSV 同解析度的 tracked.mp4（非 film_success 的低解析 MOV）
    mov_path = os.path.join(out_dir, "IMG_7129_tracked.mp4")
    prefix = "IMG_7129"

    df = pd.read_csv(csv_path)

    mm_per_px = float(df["mm_per_px"].dropna().iloc[0])
    # 起點絕對座標（與原腳本一致：x_abs = x_px_filt * mm_per_px，x0 取第一個有效值）
    x_px = df["x_px_filt"].to_numpy()
    y_px = df["y_px_filt"].to_numpy()
    valid_px = np.isfinite(x_px) & np.isfinite(y_px)
    x0 = x_px[valid_px][0] * mm_per_px
    y0 = y_px[valid_px][0] * mm_per_px

    # 主方向旋轉角 phi（與原腳本 ALIGN_TRAJ_TO_Y=True 一致）
    x_mm = df["x_mm"].to_numpy()
    y_mm = df["y_mm"].to_numpy()
    phi = S.principal_direction_xy(x_mm, y_mm)

    def to_plot_xy(x, y):
        """僅旋轉到主方向對齊 +Y；不變號、不縮放。
        旋轉後 start≈0、end≈-69，配合標準座標軸即 start 在上、end 在下，
        軸頂為正、軸底為負，尺度與原圖一致（約 80mm 跨度）。"""
        return S.rotate_xy(x, y, phi)

    x_plot, y_plot = to_plot_xy(x_mm, y_mm)

    # ---------- 圖A：Position ----------
    figA, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    ax.plot(x_plot, y_plot, lw=2, label="Trajectory")
    vp = np.vstack([x_plot, y_plot]).T
    vp = vp[~np.isnan(vp).any(axis=1)]
    if len(vp) > 0:
        ax.scatter([vp[0, 0]], [vp[0, 1]], s=100, c="green", marker="o", label="Start", zorder=5)
        ax.scatter([vp[-1, 0]], [vp[-1, 1]], s=100, c="red", marker="o", label="End", zorder=5)
        xmin, xmax = np.nanmin(x_plot), np.nanmax(x_plot)
        ymin, ymax = np.nanmin(y_plot), np.nanmax(y_plot)
        xc, yc = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        half = max(0.5 * max(xmax - xmin, ymax - ymin), 1e-6) * 1.35
        ax.set_xlim(xc - half, xc + half)
        ax.set_ylim(yc - half, yc + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)", fontsize=24)
    ax.set_ylabel("y (mm)", fontsize=24)
    ax.set_title("Position", fontsize=22)            # 需求5
    ax.legend(loc="best", fontsize=14)
    p = os.path.join(out_dir, f"{prefix}_position.png")
    figA.savefig(p, dpi=1200, bbox_inches='tight', pad_inches=0.3)
    plt.close(figA)
    print(f"[輸出] {p}")

    # ---------- 圖B-1：Speed（獨立，需求2） ----------
    figS, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ax.plot(df["t_s"], df["speed_mm_s"], lw=2, label="Speed (mm/s)")
    sp = df["speed_mm_s"].to_numpy()
    t = df["t_s"].to_numpy()
    fin = np.isfinite(sp)
    if np.any(fin):
        idxs = np.where(fin)[0]
        i_max = idxs[int(np.nanargmax(sp[fin]))]
        ax.plot([t[i_max]], [sp[i_max]], marker="o", markersize=8, color="red",
                label=f"Max: {sp[i_max]:.2f} mm/s @ {t[i_max]:.2f}s")
    ax.set_xlabel("Time (s)", fontsize=24)
    ax.set_ylabel("Speed (mm/s)", fontsize=24)
    ax.set_title("Speed vs Time", fontsize=22)
    ax.legend(loc="best", fontsize=14)
    p = os.path.join(out_dir, f"{prefix}_speed.png")
    figS.savefig(p, dpi=1200, bbox_inches='tight', pad_inches=0.3)
    plt.close(figS)
    print(f"[輸出] {p}")

    # ---------- 圖B-2：Orientation（獨立，需求2） ----------
    figO, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ang_raw = df["angle_deg_raw"].to_numpy()
    valid_ang = np.isfinite(ang_raw)
    if valid_ang.any():
        first_angle = float(ang_raw[valid_ang][0])
        ang_vis = ang_raw - first_angle
        ang_vis = S.moving_average(ang_vis, S.SMOOTH_WIN_ANGLE)
        vo = np.isfinite(ang_vis)
        avg_off = float(np.nanmean(ang_vis[vo])) if np.any(vo) else float("nan")
    else:
        ang_vis = ang_raw
        first_angle = 0.0
        avg_off = float("nan")
    label = (f"Orientation offset (deg)\nAvg: {avg_off:.2f}°"
             if np.isfinite(avg_off) else "Orientation offset (deg)")
    ax.plot(df["t_s"], ang_vis, lw=2, label=label)
    ax.set_xlabel("Time (s)", fontsize=24)
    ax.set_ylabel("Angle offset from start (deg)", fontsize=24)
    ax.set_title(f"Orientation vs Time (start: {first_angle:.1f}°)", fontsize=22)
    if S.ORIENT_YLIM_DEG is not None and np.isfinite(S.ORIENT_YLIM_DEG):
        ax.set_ylim(-float(S.ORIENT_YLIM_DEG), float(S.ORIENT_YLIM_DEG))
    else:
        vv = ang_vis[np.isfinite(ang_vis)]
        if len(vv) > 0:
            dr = max(abs(np.nanmin(vv)), abs(np.nanmax(vv)))
            yl = max(10.0, dr * 1.2)
            ax.set_ylim(-yl, yl)
    ax.legend(loc="best", fontsize=14)
    p = os.path.join(out_dir, f"{prefix}_orientation.png")
    figO.savefig(p, dpi=1200, bbox_inches='tight', pad_inches=0.3)
    plt.close(figO)
    print(f"[輸出] {p}")

    # 移除舊的合併圖（speed/orientation 已拆開）
    old = os.path.join(out_dir, f"{prefix}_speed_orientation.png")
    if os.path.exists(old):
        os.remove(old)
        print(f"[移除] 舊合併圖 {old}")

    # ---------- 圖C：Trajectory（含 8 個虛線方框，無標題） ----------
    boxes = detect_boxes(mov_path, S.find_target_and_angle, len(df))
    figC, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    ax.plot(x_plot, y_plot, lw=2, color='blue', label="Trajectory", alpha=0.7)

    frame_to_box = {fi: b for fi, b in boxes if b is not None}
    valid_frames = [int(f) for f in df["frame"].to_numpy() if int(f) in frame_to_box]
    if len(valid_frames) >= 8:
        sel = np.linspace(0, len(valid_frames) - 1, 8, dtype=int)
        for s in sel:
            fi = valid_frames[s]
            box = frame_to_box[fi]
            box_mm = np.array([[vx * mm_per_px - x0, vy * mm_per_px - y0] for vx, vy in box])
            bx, by = to_plot_xy(box_mm[:, 0], box_mm[:, 1])
            bx = np.append(bx, bx[0])
            by = np.append(by, by[0])
            ax.plot(bx, by, linestyle='--', linewidth=1.5, color=(0.5, 0.7, 1.0), alpha=0.7)

    vp = np.vstack([x_plot, y_plot]).T
    vp = vp[~np.isnan(vp).any(axis=1)]
    if len(vp) > 0:
        ax.scatter([vp[0, 0]], [vp[0, 1]], s=100, c="green", marker="o", label="Start", zorder=5)
        ax.scatter([vp[-1, 0]], [vp[-1, 1]], s=100, c="red", marker="o", label="End", zorder=5)
        xmin, xmax = np.nanmin(x_plot), np.nanmax(x_plot)
        ymin, ymax = np.nanmin(y_plot), np.nanmax(y_plot)
        xc, yc = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        half = max(0.5 * max(xmax - xmin, ymax - ymin), 1e-6) * 1.35
        ax.set_xlim(xc - half, xc + half)
        ax.set_ylim(yc - half, yc + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)", fontsize=24)
    ax.set_ylabel("y (mm)", fontsize=24)
    # 需求3：無標題
    ax.legend(loc="best", fontsize=14)
    p = os.path.join(out_dir, f"{prefix}_trajectory_center_only.png")
    figC.savefig(p, dpi=1200, bbox_inches='tight', pad_inches=0.3)
    plt.close(figC)
    print(f"[輸出] {p}")


# =============================================================
# IMG_7110：旋轉型（filming_rotation_and_speed.py）
# =============================================================
def redraw_7110():
    out_dir = os.path.join(BASE, "IMG_7110_analysis_output")
    csv_path = os.path.join(out_dir, "IMG_7110_pos_angle_speed.csv")
    mp4_path = os.path.join(out_dir, "IMG_7110_tracked.mp4")
    prefix = "IMG_7110"

    df = pd.read_csv(csv_path)
    mm_per_px = float(df["mm_per_px"].dropna().iloc[0])
    x_px = df["x_px_filt"].to_numpy()
    y_px = df["y_px_filt"].to_numpy()
    valid_px = np.isfinite(x_px) & np.isfinite(y_px)
    x0 = x_px[valid_px][0] * mm_per_px
    y0 = y_px[valid_px][0] * mm_per_px

    x_plot = df["x_mm"].to_numpy()
    y_plot = df["y_mm"].to_numpy()
    # 需求：7110 改為「上正下負」標準座標軸（正值在上、負值在下）。
    # 標準 matplotlib 軸即為此向；原腳本的 INVERT_Y_AXIS=True 會反成「上負下正」，故關閉。
    INVERT_Y = False

    # ---------- 圖A：Position ----------
    figA, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    ax.plot(x_plot, y_plot, lw=2, label="Trajectory")
    vp = np.vstack([x_plot, y_plot]).T
    vp = vp[~np.isnan(vp).any(axis=1)]
    if len(vp) > 0:
        ax.scatter([vp[0, 0]], [vp[0, 1]], s=100, c="green", marker="o", label="Start", zorder=5)
        ax.scatter([vp[-1, 0]], [vp[-1, 1]], s=100, c="red", marker="o", label="End", zorder=5)
        xmin, xmax = np.nanmin(x_plot), np.nanmax(x_plot)
        ymin, ymax = np.nanmin(y_plot), np.nanmax(y_plot)
        xc, yc = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        half = max(0.5 * max(xmax - xmin, ymax - ymin), 1e-6) * 1.35
        ax.set_xlim(xc - half, xc + half)
        ax.set_ylim(yc - half, yc + half)
    if INVERT_Y:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)", fontsize=24)
    ax.set_ylabel("y (mm)", fontsize=24)
    ax.set_title("Position", fontsize=22)            # 需求5
    ax.legend(loc="upper right", fontsize=14)        # 需求：legend 移到右上角
    p = os.path.join(out_dir, f"{prefix}_position.png")
    figA.savefig(p, dpi=1200, bbox_inches='tight', pad_inches=0.3)
    plt.close(figA)
    print(f"[輸出] {p}")

    # ---------- 圖B：Angular Speed vs Time ----------
    figW, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ax.plot(df["t_s"], df["angular_vel_dps"], lw=2, label="Angular speed (deg/s)")
    w = df["angular_vel_dps"].to_numpy()
    t = df["t_s"].to_numpy()
    fin = np.isfinite(w)
    if np.any(fin):
        idxs = np.where(fin)[0]
        i_max = idxs[int(np.nanargmax(np.abs(w[fin])))]
        ax.plot([t[i_max]], [w[i_max]], marker="o", markersize=8, color="red",
                label=f"Max: {w[i_max]:.2f} deg/s @ {t[i_max]:.2f}s")
    ax.set_xlabel("Time (s)", fontsize=24)
    ax.set_ylabel("Angular speed (deg/s)", fontsize=24)
    ax.set_title("Angular Speed vs Time", fontsize=22)
    ax.legend(loc="best", fontsize=14)
    p = os.path.join(out_dir, f"{prefix}_angular_speed.png")
    figW.savefig(p, dpi=1200, bbox_inches='tight', pad_inches=0.3)
    plt.close(figW)
    print(f"[輸出] {p}")

    # ---------- 圖B2：Theta vs Time ----------
    figT, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ax.plot(df["t_s"], df["angle_deg_unwrapped"], lw=2, color='blue', label="θ (deg)")
    th = df["angle_deg_unwrapped"].to_numpy()
    fin = np.isfinite(th)
    if np.any(fin):
        tv = t[fin]
        thv = th[fin]
        ax.scatter([tv[0]], [thv[0]], s=80, c="green", marker="o", label=f"Start: {thv[0]:.2f}°", zorder=5)
        ax.scatter([tv[-1]], [thv[-1]], s=80, c="red", marker="o", label=f"End: {thv[-1]:.2f}°", zorder=5)
        ax.set_title(f"Theta (Angle) vs Time (Total rotation: {thv[-1]-thv[0]:.2f}°)", fontsize=22)
    else:
        ax.set_title("Theta (Angle) vs Time", fontsize=22)
    ax.set_xlabel("Time (s)", fontsize=24)
    ax.set_ylabel("θ (deg)", fontsize=24)
    ax.legend(loc="best", fontsize=14)
    p = os.path.join(out_dir, f"{prefix}_theta_vs_time.png")
    figT.savefig(p, dpi=1200, bbox_inches='tight', pad_inches=0.3)
    plt.close(figT)
    print(f"[輸出] {p}")

    # ---------- 圖C：Trajectory（方框由 tracked.mp4 重建，無標題） ----------
    boxes = detect_boxes(mp4_path, R.find_target_and_angle, len(df))
    figC, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    ax.plot(x_plot, y_plot, lw=2, color='blue', label="Trajectory", alpha=0.7)

    frame_to_box = {fi: b for fi, b in boxes if b is not None}
    valid_frames = [int(f) for f in df["frame"].to_numpy() if int(f) in frame_to_box]
    all_x = [x_plot[~np.isnan(x_plot)]]
    all_y = [y_plot[~np.isnan(y_plot)]]
    if len(valid_frames) >= 8:
        sel = np.linspace(0, len(valid_frames) - 1, 8, dtype=int)
        for s in sel:
            fi = valid_frames[s]
            box = frame_to_box[fi]
            box_mm = np.array([[vx * mm_per_px - x0, vy * mm_per_px - y0] for vx, vy in box])
            bx = np.append(box_mm[:, 0], box_mm[0, 0])
            by = np.append(box_mm[:, 1], box_mm[0, 1])
            ax.plot(bx, by, linestyle='--', linewidth=1.5, color=(0.5, 0.7, 1.0), alpha=0.7)
            all_x.append(box_mm[:, 0])
            all_y.append(box_mm[:, 1])

    vp = np.vstack([x_plot, y_plot]).T
    vp = vp[~np.isnan(vp).any(axis=1)]
    if len(vp) > 0:
        ax.scatter([vp[0, 0]], [vp[0, 1]], s=100, c="green", marker="o", label="Start", zorder=5)
        ax.scatter([vp[-1, 0]], [vp[-1, 1]], s=100, c="red", marker="o", label="End", zorder=5)

    MIN_AXIS_RANGE = 15
    ax_all = np.concatenate(all_x)
    ay_all = np.concatenate(all_y)
    dxmin, dxmax = np.nanmin(ax_all), np.nanmax(ax_all)
    dymin, dymax = np.nanmin(ay_all), np.nanmax(ay_all)
    xc, yc = 0.5 * (dxmin + dxmax), 0.5 * (dymin + dymax)
    half = max(0.5 * (dxmax - dxmin) * 1.2, 0.5 * (dymax - dymin) * 1.2, MIN_AXIS_RANGE)
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(yc - half, yc + half)
    if INVERT_Y:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)", fontsize=24)
    ax.set_ylabel("y (mm)", fontsize=24)
    # 需求3：無標題
    ax.legend(loc="best", fontsize=14)
    p = os.path.join(out_dir, f"{prefix}_trajectory_center_only.png")
    figC.savefig(p, dpi=1200, bbox_inches='tight', pad_inches=0.3)
    plt.close(figC)
    print(f"[輸出] {p}")


if __name__ == "__main__":
    print("===== 重繪 IMG_7129 =====")
    redraw_7129()
    print("===== 重繪 IMG_7110 =====")
    redraw_7110()
    print("完成。")
