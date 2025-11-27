# -*- coding: utf-8 -*-
"""
Rotation 版（符合最新需求）
-----------------------------------------
1) tracked 影片：每幀都有紅色輪廓框 + 綠色中心點 + 疊字 Pos/Angle/Speed
2) Position 圖：不旋轉、不平移，以絕對 mm 繪製；僅在起點位置標示 "(0, 0)"
3) 分析圖：改為單張 Angular Speed vs Time（最大值紅點），不輸出 Orientation 圖
4) CSV：輸出 x_mm, y_mm（絕對）、angle_raw/unwrapped、angular_vel、speed_mm_s
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 使用者設定 =========
GRID_SPACING_MM = 5.0
VIDEO_PATH = "IMG_7110.mov"            # ← 換成你的影片路徑
OUT_PREFIX = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

# 顏色遮罩（黃＋白）
HSV_YELLOW_LO = np.array([15, 60, 120], dtype=np.uint8)
HSV_YELLOW_HI = np.array([35, 255, 255], dtype=np.uint8)
HSV_WHITE_LO  = np.array([0,  0, 200], dtype=np.uint8)
HSV_WHITE_HI  = np.array([180, 60, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 50
PROCESS_EVERY_N  = 1                   # 逐幀處理
SMOOTH_WIN_ANGLE = 5                   # 角度/角速度平滑視窗
SMOOTH_WIN_SPEED = 5                   # 線速度平滑視窗

# 比例（mm/px）
AUTO_GRID_MM_PER_PX = True
MANUAL_MM_PER_PX    = None

# 視覺化選項
INVERT_Y_AXIS       = True             # Position 圖是否反轉 Y 軸
ORIENT_YLIM_DEG     = 60               #（此版不畫 Orientation 圖；保留作為可擴充用）
# ============================

def estimate_mm_per_px_single_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=60, maxLineGap=10)
    if lines is None or len(lines) < 8:
        return None

    horiz, vert = [], []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        dx, dy = x2-x1, y2-y1
        ang = np.degrees(np.arctan2(dy, dx))
        if ang < -90: ang += 180
        if ang > 90:  ang -= 180
        if abs(ang) < 10:
            horiz.extend([y1, y2])
        elif abs(abs(ang)-90) < 10:
            vert.extend([x1, x2])

    def spacing(pos):
        if len(pos) < 6: return None
        pos = np.array(sorted(pos))
        uniq = [pos[0]]
        for v in pos[1:]:
            if abs(v - uniq[-1]) > 2:
                uniq.append(v)
        uniq = np.array(uniq)
        if len(uniq) < 5: return None
        diffs = np.diff(uniq)
        diffs = diffs[(diffs > 5) & (diffs < 200)]
        if len(diffs) == 0: return None
        return float(np.median(diffs))

    sp_h = spacing(horiz)
    sp_v = spacing(vert)
    if sp_h and sp_v:
        px_per_cell = (sp_h + sp_v) / 2.0
    elif sp_h:
        px_per_cell = sp_h
    elif sp_v:
        px_per_cell = sp_v
    else:
        return None

    return (GRID_SPACING_MM / px_per_cell) if (px_per_cell and px_per_cell > 0) else None

def unwrap_angles_deg(angles_deg):
    if len(angles_deg) == 0:
        return angles_deg
    rad = np.deg2rad(angles_deg)
    rad_unwrap = np.unwrap(rad)
    return np.rad2deg(rad_unwrap)

def moving_average(a, w):
    if w is None or w <= 1:
        return a
    a = np.asarray(a, dtype=float)
    if len(a) < w:
        return a
    kernel = np.ones(w) / float(w)
    return np.convolve(a, kernel, mode='same')

def make_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf

def find_target_and_angle(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask_y = cv2.inRange(hsv, HSV_YELLOW_LO, HSV_YELLOW_HI)
    mask_w = cv2.inRange(hsv, HSV_WHITE_LO , HSV_WHITE_HI)
    mask = cv2.bitwise_or(mask_y, mask_w)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
        return None

    rect = cv2.minAreaRect(cnt)  # (center(x,y), (w,h), angle)
    (cx, cy), (rw, rh), rect_angle = rect

    # 角度正規化到 [-90, 90)
    if rw >= rh:
        angle_deg = rect_angle + 90.0
    else:
        angle_deg = rect_angle
    while angle_deg >= 90.0: angle_deg -= 180.0
    while angle_deg < -90.0: angle_deg += 180.0

    x, y, w, h = cv2.boundingRect(cnt)
    return (cx, cy, x, y, w, h, angle_deg, cnt)

def create_output_directory(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"{base_name}_analysis_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def finite_diff(values, t, smooth_win=1):
    """對齊原序列的差分；首元素為 NaN。"""
    v = np.full_like(values, np.nan, dtype=float)
    for i in range(1, len(values)):
        if np.isfinite(values[i]) and np.isfinite(values[i-1]) and (t[i] > t[i-1]):
            v[i] = (values[i] - values[i-1]) / (t[i] - t[i-1])
    if smooth_win and smooth_win > 1:
        v = moving_average(v, smooth_win)
    return v

def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"找不到影片：{VIDEO_PATH}")

    output_dir = create_output_directory(VIDEO_PATH)
    print(f"[資訊] 輸出資料夾：{output_dir}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("讀取影片第一幀失敗")

    # mm/px
    if AUTO_GRID_MM_PER_PX:
        mm_per_px = estimate_mm_per_px_single_frame(first)
    else:
        mm_per_px = MANUAL_MM_PER_PX
    if mm_per_px is None:
        mm_per_px = 0.1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_dir, f"{OUT_PREFIX}_tracked.mp4")  #（去掉 rotation 字樣）
    writer = cv2.VideoWriter(out_path, fourcc, max(5.0, fps/PROCESS_EVERY_N), (W, H))

    # Kalman 初始化
    kf = make_kalman()
    init_meas = find_target_and_angle(first)
    if init_meas is not None:
        cx0, cy0 = float(init_meas[0]), float(init_meas[1])
    else:
        cx0, cy0 = W/2.0, H/2.0
    kf.statePost = np.array([[cx0], [cy0], [0.0], [0.0]], dtype=np.float32)

    # 重新從第 0 幀開始
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rec = []  # frame, t_s, x_px_filt, y_px_filt, x_mm, y_mm, angle_deg_raw, mm_per_px
    rec_corners = []  # 記錄四個頂點的座標 (frame, t_s, c1_x_mm, c1_y_mm, c2_x_mm, c2_y_mm, c3_x_mm, c3_y_mm, c4_x_mm, c4_y_mm)
    frame_idx = 0

    # 疊字即時線速度（mm/s）
    last_x_mm = None
    last_y_mm = None
    last_t    = None
    inst_speed = np.nan

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % PROCESS_EVERY_N != 0:
            frame_idx += 1
            continue

        pred = kf.predict()
        meas = find_target_and_angle(frame)
        angle_deg = np.nan
        corners_mm = [np.nan] * 8  # 4個頂點的 x, y 座標（mm）

        if meas is not None:
            cx, cy, x, y, wbox, hbox, angle_deg, cnt = meas
            est = kf.correct(np.array([[cx],[cy]], dtype=np.float32))
            fx, fy = float(est[0,0]), float(est[1,0])

            # --- 覆蓋需求：紅色輪廓框 + 綠色中心點 ---
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 4)
            if np.isfinite(fx) and np.isfinite(fy):
                cv2.circle(frame, (int(round(fx)), int(round(fy))), 6, (0,255,0), -1)
            
            # 計算旋轉矩形的四個頂點
            rect = cv2.minAreaRect(cnt)
            box_points = cv2.boxPoints(rect)  # 返回4個頂點座標 (x, y)
            # 將頂點轉換為 mm 單位
            for i, (px, py) in enumerate(box_points):
                corners_mm[i*2] = px * mm_per_px
                corners_mm[i*2 + 1] = py * mm_per_px
        else:
            fx = fy = np.nan

        # 絕對 mm
        fx_mm = fx * mm_per_px if np.isfinite(fx) else np.nan
        fy_mm = fy * mm_per_px if np.isfinite(fy) else np.nan

        # 即時線速度（相鄰幀）
        t_s = frame_idx / fps
        if last_x_mm is not None and last_y_mm is not None and last_t is not None and \
           np.isfinite(fx_mm) and np.isfinite(fy_mm) and (t_s > last_t):
            inst_speed = np.hypot(fx_mm - last_x_mm, fy_mm - last_y_mm) / (t_s - last_t)
        last_x_mm, last_y_mm, last_t = fx_mm, fy_mm, t_s

        # 疊字：Pos/Angle/Speed（全為絕對 mm 與當下估算）
        cv2.putText(frame, f"Pos(mm): ({fx_mm:.2f}, {fy_mm:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        if np.isnan(angle_deg):
            cv2.putText(frame, "Angle: NaN",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(frame, f"Angle: {angle_deg:+.2f} deg",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        if np.isnan(inst_speed):
            cv2.putText(frame, "Speed: NaN mm/s",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(frame, f"Speed: {inst_speed:.2f} mm/s",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        writer.write(frame)

        # 記錄中心點
        rec.append((frame_idx, t_s, fx, fy, fx_mm, fy_mm, angle_deg, mm_per_px))
        # 記錄四個頂點
        rec_corners.append([frame_idx, t_s] + corners_mm)
        frame_idx += 1

    cap.release()
    writer.release()

    # ---- 匯總為 DataFrame ----
    df = pd.DataFrame(rec, columns=[
        "frame","t_s","x_px_filt","y_px_filt","x_mm","y_mm","angle_deg_raw","mm_per_px"
    ])
    
    # 四個頂點的 DataFrame
    df_corners = pd.DataFrame(rec_corners, columns=[
        "frame","t_s",
        "c1_x_mm","c1_y_mm","c2_x_mm","c2_y_mm",
        "c3_x_mm","c3_y_mm","c4_x_mm","c4_y_mm"
    ])

    # === 將座標平移，使起點為 (0, 0) ===
    x_abs = df["x_mm"].to_numpy()
    y_abs = df["y_mm"].to_numpy()
    valid = np.isfinite(x_abs) & np.isfinite(y_abs)
    if valid.any():
        x0, y0 = x_abs[valid][0], y_abs[valid][0]
        df["x_mm"] = x_abs - x0
        df["y_mm"] = y_abs - y0
        
        # 對四個頂點也進行相同的平移
        for i in range(1, 5):
            df_corners[f"c{i}_x_mm"] = df_corners[f"c{i}_x_mm"] - x0
            df_corners[f"c{i}_y_mm"] = df_corners[f"c{i}_y_mm"] - y0
    else:
        df["x_mm"] = x_abs
        df["y_mm"] = y_abs


    # 角度展開 + 平滑
    angle_raw = df["angle_deg_raw"].to_numpy()
    mask_ang  = np.isfinite(angle_raw)
    angle_unwrapped = np.full_like(angle_raw, np.nan, dtype=float)
    if mask_ang.any():
        angle_unwrapped[mask_ang] = unwrap_angles_deg(angle_raw[mask_ang])
        angle_unwrapped = moving_average(angle_unwrapped, SMOOTH_WIN_ANGLE)
    df["angle_deg_unwrapped"] = angle_unwrapped

    # 角速度（deg/s）
    t = df["t_s"].to_numpy()
    ang_vel = np.full_like(angle_unwrapped, np.nan, dtype=float)
    idx = np.where(mask_ang)[0]
    if len(idx) >= 2:
        for i0, i1 in zip(idx[:-1], idx[1:]):
            dt = t[i1] - t[i0]
            if dt > 0:
                ang_vel[i1] = (angle_unwrapped[i1] - angle_unwrapped[i0]) / dt
        ang_vel = moving_average(ang_vel, SMOOTH_WIN_ANGLE)
    df["angular_vel_dps"] = ang_vel

    # 線速度（mm/s）—可參考 CSV 用；圖不畫
    vx = finite_diff(df["x_mm"].to_numpy(), t, smooth_win=SMOOTH_WIN_SPEED)
    vy = finite_diff(df["y_mm"].to_numpy(), t, smooth_win=SMOOTH_WIN_SPEED)
    speed = np.sqrt(vx**2 + vy**2)
    df["vx_mm_s"] = vx
    df["vy_mm_s"] = vy
    df["speed_mm_s"] = speed

    # CSV 輸出
    csv_path = os.path.join(output_dir, f"{OUT_PREFIX}_pos_angle_speed.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[輸出] 追蹤影片：{out_path}")
    print(f"[輸出] 數據 CSV：{csv_path}")
    try:
        mmpp_preview = df['mm_per_px'].dropna().iloc[0]
        print(f"[資訊] 使用 mm/px ≈ {mmpp_preview:.6f}")
    except Exception:
        pass

    # ===== 視覺化 =====
    # A) Position（絕對 mm；不旋轉、不平移；起點標示 "(0, 0)"）
    figA, ax_pos = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    x_plot = df["x_mm"].to_numpy()
    y_plot = df["y_mm"].to_numpy()

    ax_pos.plot(x_plot, y_plot, lw=2, label="Trajectory")

    valid_pos = np.vstack([x_plot, y_plot]).T
    valid_pos = valid_pos[~np.isnan(valid_pos).any(axis=1)]
    if len(valid_pos) > 0:
        x_start, y_start = valid_pos[0,0], valid_pos[0,1]
        x_end,   y_end   = valid_pos[-1,0], valid_pos[-1,1]
        ax_pos.scatter([x_start], [y_start], s=100, c="green", marker="o", label="Start", zorder=5)
        ax_pos.scatter([x_end],   [y_end],   s=100, c="red",   marker="o", label="End",   zorder=5)

        # 自動視窗
        xmin, xmax = np.nanmin(x_plot), np.nanmax(x_plot)
        ymin, ymax = np.nanmin(y_plot), np.nanmax(y_plot)
        xc = 0.5 * (xmin + xmax)
        yc = 0.5 * (ymin + ymax)
        half = 0.5 * max(xmax - xmin, ymax - ymin)
        half = max(half, 1e-6) * 1.35
        ax_pos.set_xlim(xc - half, xc + half)
        ax_pos.set_ylim(yc - half, yc + half)

    if INVERT_Y_AXIS:
        ax_pos.invert_yaxis()
    ax_pos.set_aspect("equal", adjustable="box")
    ax_pos.set_xlabel("x (mm)")
    ax_pos.set_ylabel("y (mm)")
    ax_pos.set_title("Position (Trajectory)")
    ax_pos.grid(True, linestyle="--", alpha=0.4)
    ax_pos.legend(loc="best")

    plot_pos_path = os.path.join(output_dir, f"{OUT_PREFIX}_position.png")
    figA.savefig(plot_pos_path, dpi=220)
    plt.close(figA)
    print(f"[輸出] 圖片：{plot_pos_path}")

    # B) Angular Speed vs Time（最大值紅點）
    figW, ax_w = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ax_w.plot(df["t_s"], df["angular_vel_dps"], lw=2, label="Angular speed (deg/s)")

    w_all = df["angular_vel_dps"].to_numpy()
    t_all = df["t_s"].to_numpy()
    finite = np.isfinite(w_all)
    if np.any(finite):
        # 以「絕對值」找最大角速度
        idx_rel = int(np.nanargmax(np.abs(w_all[finite])))
        idxs = np.where(finite)[0]
        i_max = idxs[idx_rel]
        w_max = float(w_all[i_max])
        t_at  = float(t_all[i_max])
        ax_w.plot([t_at], [w_max], marker="o", markersize=8, color="red",
                  label=f"Max: {w_max:.2f} deg/s @ {t_at:.2f}s")

    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("Angular speed (deg/s)")
    ax_w.set_title("Angular Speed vs Time")
    ax_w.grid(True, linestyle="--", alpha=0.4)
    ax_w.legend(loc="best")

    plot_w_path = os.path.join(output_dir, f"{OUT_PREFIX}_angular_speed.png")
    figW.savefig(plot_w_path, dpi=220)
    plt.close(figW)
    print(f"[輸出] 圖片：{plot_w_path}")

    # C) 五點軌跡圖（4個頂點用虛線 + 中心用實線）
    figC, ax_five = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    
    # 繪製四個頂點的軌跡（虛線、更明顯的顏色）
    corner_colors = ['cyan', 'magenta', 'lime', 'orange']  # 使用更明顯的顏色
    corner_labels = ['Corner 1', 'Corner 2', 'Corner 3', 'Corner 4']
    
    for i in range(1, 5):
        x_corner = df_corners[f"c{i}_x_mm"].to_numpy()
        y_corner = df_corners[f"c{i}_y_mm"].to_numpy()
        ax_five.plot(x_corner, y_corner, linestyle='--', linewidth=2.0,  # 增加線寬
                    color=corner_colors[i-1], alpha=0.7, label=corner_labels[i-1])
    
    # 繪製中心點軌跡（實線，較粗）
    ax_five.plot(x_plot, y_plot, lw=3.0, color='blue', label="Center", zorder=5)
    
    # 標記起點和終點
    valid_pos = np.vstack([x_plot, y_plot]).T
    valid_pos = valid_pos[~np.isnan(valid_pos).any(axis=1)]
    if len(valid_pos) > 0:
        x_start, y_start = valid_pos[0,0], valid_pos[0,1]
        x_end,   y_end   = valid_pos[-1,0], valid_pos[-1,1]
        ax_five.scatter([x_start], [y_start], s=120, c="green", marker="o", 
                       label="Start", zorder=10, edgecolors='black', linewidths=2)
        ax_five.scatter([x_end],   [y_end],   s=120, c="red",   marker="o", 
                       label="End",   zorder=10, edgecolors='black', linewidths=2)
        
        # 自動視窗 - 考慮所有點（中心+頂點）
        all_x = [x_plot]
        all_y = [y_plot]
        for i in range(1, 5):
            all_x.append(df_corners[f"c{i}_x_mm"].to_numpy())
            all_y.append(df_corners[f"c{i}_y_mm"].to_numpy())
        
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        
        xmin, xmax = np.nanmin(all_x), np.nanmax(all_x)
        ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)
        xc = 0.5 * (xmin + xmax)
        yc = 0.5 * (ymin + ymax)
        half = 0.5 * max(xmax - xmin, ymax - ymin)
        half = max(half, 1e-6) * 1.35
        ax_five.set_xlim(xc - half, xc + half)
        ax_five.set_ylim(yc - half, yc + half)
    
    if INVERT_Y_AXIS:
        ax_five.invert_yaxis()
    ax_five.set_aspect("equal", adjustable="box")
    ax_five.set_xlabel("x (mm)")
    ax_five.set_ylabel("y (mm)")
    ax_five.set_title("5-Point Trajectory (4 Corners + Center)")
    ax_five.grid(True, linestyle="--", alpha=0.4)
    ax_five.legend(loc="best", fontsize=8)
    
    plot_five_path = os.path.join(output_dir, f"{OUT_PREFIX}_five_points_trajectory.png")
    figC.savefig(plot_five_path, dpi=220)
    plt.close(figC)
    print(f"[輸出] 圖片：{plot_five_path}")

    print(f"\n所有輸出檔案已整理至資料夾：{output_dir}")

if __name__ == "__main__":
    main()
