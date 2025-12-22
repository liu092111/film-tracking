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
VIDEO_PATH = "film/load carrying rotate.mp4"            # ← 換成你的影片路徑
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

def unwrap_angles_deg(angles_deg, max_angular_vel_dps=300.0, fps=30.0):
    """
    手動展開角度序列，支援多圈連續旋轉（超過 360°）。
    
    原理：
    - cv2.minAreaRect 回傳的角度範圍是 [-90°, 90°)，週期為 180°
    - 當相鄰幀角度差超過 90° 時，代表發生了跳變
    - 透過將差值正規化到 [-90, 90) 範圍來自動修正跳變
    - 累積差值到前一個展開值
    
    參數：
    - angles_deg: 原始角度序列 (範圍 [-90, 90))
    - max_angular_vel_dps: 最大合理角速度 (deg/s)，用於異常檢測（備用）
    - fps: 幀率（備用）
    """
    if len(angles_deg) == 0:
        return angles_deg
    
    angles_deg = np.asarray(angles_deg, dtype=float)
    unwrapped = np.zeros_like(angles_deg)
    
    # 找第一個有效值作為起點
    first_valid_idx = 0
    for i in range(len(angles_deg)):
        if np.isfinite(angles_deg[i]):
            first_valid_idx = i
            break
    
    unwrapped[first_valid_idx] = angles_deg[first_valid_idx]
    
    # 向前填充 NaN（如果有的話）
    for i in range(first_valid_idx):
        unwrapped[i] = np.nan
    
    # 從第一個有效值開始累積展開
    last_valid_idx = first_valid_idx
    last_valid_unwrapped = unwrapped[first_valid_idx]
    last_valid_raw = angles_deg[first_valid_idx]
    
    for i in range(first_valid_idx + 1, len(angles_deg)):
        if np.isnan(angles_deg[i]):
            unwrapped[i] = np.nan
            continue
        
        # 計算與上一個有效原始角度的差值
        delta = angles_deg[i] - last_valid_raw
        
        # 將差值正規化到 [-90, 90) 範圍
        # 這樣可以自動處理 ±90° 邊界的跳變
        while delta >= 90.0:
            delta -= 180.0
        while delta < -90.0:
            delta += 180.0
        
        # 累積到上一個有效展開值
        unwrapped[i] = last_valid_unwrapped + delta
        
        # 更新追蹤變數
        last_valid_idx = i
        last_valid_unwrapped = unwrapped[i]
        last_valid_raw = angles_deg[i]
    
    return unwrapped

def moving_average(a, w):
    """
    移動平均平滑，正確處理 NaN 值和邊界。
    """
    if w is None or w <= 1:
        return a
    a = np.asarray(a, dtype=float)
    if len(a) < w:
        return a
    
    # 創建輸出陣列
    result = np.full_like(a, np.nan, dtype=float)
    half_w = w // 2
    
    for i in range(len(a)):
        # 定義窗口範圍
        start = max(0, i - half_w)
        end = min(len(a), i + half_w + 1)
        
        # 提取窗口內的值並排除 NaN
        window = a[start:end]
        valid = window[np.isfinite(window)]
        
        if len(valid) > 0:
            result[i] = np.mean(valid)
    
    return result

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
    box = cv2.boxPoints(rect)    # 4 頂點 (float)
    box = np.array(box, dtype=np.float32)

    # 角度正規化到 [-90, 90)
    if rw >= rh:
        angle_deg = rect_angle + 90.0
    else:
        angle_deg = rect_angle
    while angle_deg >= 90.0: angle_deg -= 180.0
    while angle_deg < -90.0: angle_deg += 180.0

    x, y, w, h = cv2.boundingRect(cnt)
    return (cx, cy, x, y, w, h, angle_deg, cnt, box)

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
    box_rec = []  # 記錄每一幀的 box (四個頂點) 用於後續繪製
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
        box = None

        if meas is not None:
            cx, cy, x, y, wbox, hbox, angle_deg, cnt, box = meas
            est = kf.correct(np.array([[cx],[cy]], dtype=np.float32))
            fx, fy = float(est[0,0]), float(est[1,0])

            # --- 覆蓋需求：紅色輪廓框 + 綠色中心點 ---
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 4)
            if np.isfinite(fx) and np.isfinite(fy):
                cv2.circle(frame, (int(round(fx)), int(round(fy))), 6, (0,255,0), -1)
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
        # 記錄 box（如果有偵測到）
        if meas is not None:
            box_rec.append((frame_idx, t_s, fx, fy, box))
        else:
            box_rec.append((frame_idx, t_s, fx, fy, None))
        frame_idx += 1

    cap.release()
    writer.release()

    # ---- 匯總為 DataFrame ----
    df = pd.DataFrame(rec, columns=[
        "frame","t_s","x_px_filt","y_px_filt","x_mm","y_mm","angle_deg_raw","mm_per_px"
    ])

    # === 將座標平移，使起點為 (0, 0) ===
    x_abs = df["x_mm"].to_numpy()
    y_abs = df["y_mm"].to_numpy()
    valid = np.isfinite(x_abs) & np.isfinite(y_abs)
    if valid.any():
        x0, y0 = x_abs[valid][0], y_abs[valid][0]
        df["x_mm"] = x_abs - x0
        df["y_mm"] = y_abs - y0
    else:
        df["x_mm"] = x_abs
        df["y_mm"] = y_abs


    # 角度展開 + 平滑
    angle_raw = df["angle_deg_raw"].to_numpy()
    mask_ang  = np.isfinite(angle_raw)
    angle_unwrapped = np.full_like(angle_raw, np.nan, dtype=float)
    if mask_ang.any():
        # 傳入實際 fps 以正確計算最大允許角速度
        angle_unwrapped[mask_ang] = unwrap_angles_deg(angle_raw[mask_ang], max_angular_vel_dps=300.0, fps=fps)
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

    # B2) Theta (Angle) vs Time
    figTheta, ax_theta = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ax_theta.plot(df["t_s"], df["angle_deg_unwrapped"], lw=2, color='blue', label="θ (deg)")

    theta_all = df["angle_deg_unwrapped"].to_numpy()
    finite_theta = np.isfinite(theta_all)
    if np.any(finite_theta):
        # 標記起點和終點的角度
        t_valid = t_all[finite_theta]
        theta_valid = theta_all[finite_theta]
        ax_theta.scatter([t_valid[0]], [theta_valid[0]], s=80, c="green", marker="o", 
                        label=f"Start: {theta_valid[0]:.2f}°", zorder=5)
        ax_theta.scatter([t_valid[-1]], [theta_valid[-1]], s=80, c="red", marker="o", 
                        label=f"End: {theta_valid[-1]:.2f}°", zorder=5)
        
        # 計算總旋轉角度
        total_rotation = theta_valid[-1] - theta_valid[0]
        ax_theta.set_title(f"Theta (Angle) vs Time (Total rotation: {total_rotation:.2f}°)")
    else:
        ax_theta.set_title("Theta (Angle) vs Time")

    ax_theta.set_xlabel("Time (s)")
    ax_theta.set_ylabel("θ (deg)")
    ax_theta.grid(True, linestyle="--", alpha=0.4)
    ax_theta.legend(loc="best")

    plot_theta_path = os.path.join(output_dir, f"{OUT_PREFIX}_theta_vs_time.png")
    figTheta.savefig(plot_theta_path, dpi=220)
    plt.close(figTheta)
    print(f"[輸出] 圖片：{plot_theta_path}")

    # C) 軌跡圖（統一虛線、同個顏色）
    figC, ax_contour = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    
    # 繪製整體路徑
    ax_contour.plot(x_plot, y_plot, lw=2, color='blue', label="Trajectory", alpha=0.7)
    
    # 找出有效的幀（有box數據的）
    valid_boxes = [(idx, ts, fx_val, fy_val, box) for idx, ts, fx_val, fy_val, box in box_rec 
                   if box is not None and np.isfinite(fx_val) and np.isfinite(fy_val)]
    
    if len(valid_boxes) >= 8:
        # 選擇8個等間隔的時間點，包括起點和終點
        n_segments = 8
        indices = np.linspace(0, len(valid_boxes) - 1, n_segments, dtype=int)
        
        # 使用統一顏色虛線繪製輪廓
        for i, idx in enumerate(indices):
            frame_idx_val, t_s_val, fx_px_val, fy_px_val, box = valid_boxes[idx]
            
            # 找出對應的 mm 座標（相對起點）
            df_row = df[df['frame'] == frame_idx_val]
            if len(df_row) > 0:
                # 將box的四個頂點轉換為相對mm座標
                box_mm = []
                for vertex in box:
                    vx_px, vy_px = vertex[0], vertex[1]
                    vx_mm = vx_px * mm_per_px - x0
                    vy_mm = vy_px * mm_per_px - y0
                    box_mm.append([vx_mm, vy_mm])
                
                box_mm = np.array(box_mm)
                
                # 繪製輪廓（統一淺藍色虛線）
                color = (0.5, 0.7, 1.0)  # 淺藍色
                linestyle = '--'
                linewidth = 1.5
                alpha = 0.7
                
                box_closed = np.vstack([box_mm, box_mm[0:1]])
                ax_contour.plot(box_closed[:, 0], box_closed[:, 1], 
                              linestyle=linestyle, linewidth=linewidth, 
                              color=color, alpha=alpha)
    
    # 標記起點和終點
    valid_pos = np.vstack([x_plot, y_plot]).T
    valid_pos = valid_pos[~np.isnan(valid_pos).any(axis=1)]
    if len(valid_pos) > 0:
        x_start, y_start = valid_pos[0, 0], valid_pos[0, 1]
        x_end, y_end = valid_pos[-1, 0], valid_pos[-1, 1]
        ax_contour.scatter([x_start], [y_start], s=100, c="green", 
                          marker="o", label="Start", zorder=5)
        ax_contour.scatter([x_end], [y_end], s=100, c="red", 
                          marker="o", label="End", zorder=5)
    
    # 設置座標軸範圍：自動偵測，確保所有數據可見，最小範圍 ±15mm
    MIN_AXIS_RANGE = 15  # 最小軸範圍（mm）
    
    # 計算實際數據範圍（包含軌跡和輪廓box）
    all_x = [x_plot[~np.isnan(x_plot)]]
    all_y = [y_plot[~np.isnan(y_plot)]]
    
    # 加入box頂點的座標
    if len(valid_boxes) >= 8:
        for i, idx in enumerate(indices):
            frame_idx_val, t_s_val, fx_px_val, fy_px_val, box = valid_boxes[idx]
            if box is not None:
                for vertex in box:
                    vx_mm = vertex[0] * mm_per_px - x0
                    vy_mm = vertex[1] * mm_per_px - y0
                    all_x.append([vx_mm])
                    all_y.append([vy_mm])
    
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    
    # 計算範圍
    data_xmin, data_xmax = np.nanmin(all_x), np.nanmax(all_x)
    data_ymin, data_ymax = np.nanmin(all_y), np.nanmax(all_y)
    
    # 計算中心和半範圍
    xc = 0.5 * (data_xmin + data_xmax)
    yc = 0.5 * (data_ymin + data_ymax)
    half_x = 0.5 * (data_xmax - data_xmin) * 1.2  # 加 20% 邊距
    half_y = 0.5 * (data_ymax - data_ymin) * 1.2
    half_range = max(half_x, half_y, MIN_AXIS_RANGE)  # 確保最小為 MIN_AXIS_RANGE
    
    ax_contour.set_xlim(xc - half_range, xc + half_range)
    ax_contour.set_ylim(yc - half_range, yc + half_range)
    
    if INVERT_Y_AXIS:
        ax_contour.invert_yaxis()
    
    ax_contour.set_aspect("equal", adjustable="box")
    ax_contour.set_xlabel("x (mm)")
    ax_contour.set_ylabel("y (mm)")
    ax_contour.set_title("Center Trajectory with 8 Time Points")
    ax_contour.grid(True, linestyle="--", alpha=0.4)
    ax_contour.legend(loc="best")
    
    plot_contour_path = os.path.join(output_dir, f"{OUT_PREFIX}_trajectory_center_only.png")
    figC.savefig(plot_contour_path, dpi=220)
    plt.close(figC)
    print(f"[輸出] 圖片（八等分輪廓）：{plot_contour_path}")

    # ====== 新增：從原始影片提取 8 個時間點的畫面並合併（帶中心路徑和累積輪廓）======
    if len(valid_boxes) >= 8:
        print(f"\n[資訊] 正在提取 8 個時間點的影片畫面...")
        
        # 重新開啟原始影片
        cap_composite = cv2.VideoCapture(VIDEO_PATH)
        
        # 使用與輪廓圖相同的 8 個時間點
        n_segments = 8
        indices = np.linspace(0, len(valid_boxes) - 1, n_segments, dtype=int)
        
        # 提取對應的 frame 編號和時間
        snapshot_frames = []
        snapshot_times = []
        for idx in indices:
            frame_idx_val, t_s_val, _, _, _ = valid_boxes[idx]
            snapshot_frames.append(frame_idx_val)
            # 計算相對時間（從第一個有效幀開始為 0s）
            first_valid_time = valid_boxes[0][1]
            relative_time = t_s_val - first_valid_time
            snapshot_times.append(relative_time)
        
        # 用表格逐幀疊字
        row_map = {int(r.frame): i for i, r in df.iterrows()}
        
        # 提取畫面並繪製中心路徑和累積輪廓
        extracted_frames = []
        
        for snap_idx, frame_num in enumerate(snapshot_frames):
            cap_composite.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame = cap_composite.read()
            if not ok:
                continue
            
            # 繪製該時間點之前的所有輪廓（淺藍色虛線）
            light_blue = (255, 200, 150)  # BGR 淺藍色
            for i in range(snap_idx + 1):  # 包含當前時間點
                box_idx = indices[i]
                _, _, _, _, box = valid_boxes[box_idx]
                if box is not None:
                    box_int = np.int32(box)
                    # 繪製虛線輪廓
                    for j in range(4):
                        pt1 = tuple(box_int[j])
                        pt2 = tuple(box_int[(j + 1) % 4])
                        # 簡單虛線實現
                        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
                        num_segments = int(dist / 10)
                        if num_segments > 0:
                            for k in range(num_segments):
                                if k % 2 == 0:  # 只畫偶數段
                                    start = (
                                        int(pt1[0] + (pt2[0] - pt1[0]) * k / num_segments),
                                        int(pt1[1] + (pt2[1] - pt1[1]) * k / num_segments)
                                    )
                                    end = (
                                        int(pt1[0] + (pt2[0] - pt1[0]) * (k + 1) / num_segments),
                                        int(pt1[1] + (pt2[1] - pt1[1]) * (k + 1) / num_segments)
                                    )
                                    cv2.line(frame, start, end, light_blue, 2)
            
            # 繪製完整的中心路徑（從起點到當前時間點的所有位置）
            center_color_composite = (255, 0, 0)  # BGR 藍色
            current_frame_num = snapshot_frames[snap_idx]
            
            # 收集所有從起點到當前幀的位置
            path_points = []
            for idx in range(len(df)):
                if df.iloc[idx]['frame'] <= current_frame_num:
                    fx_val = df.iloc[idx]['x_px_filt']
                    fy_val = df.iloc[idx]['y_px_filt']
                    if np.isfinite(fx_val) and np.isfinite(fy_val):
                        path_points.append((int(round(fx_val)), int(round(fy_val))))
            
            # 繪製完整路徑（連續的線）
            if len(path_points) > 1:
                for i in range(1, len(path_points)):
                    cv2.line(frame, path_points[i-1], path_points[i], 
                           center_color_composite, 2)
            
            # 在8個snapshot點上畫圓標記（純藍色，無外框）
            for i in range(snap_idx + 1):
                df_idx = snapshot_frames[i]
                if df_idx in row_map:
                    fx_px = df[df['frame'] == df_idx]['x_px_filt'].values
                    fy_px = df[df['frame'] == df_idx]['y_px_filt'].values
                    if len(fx_px) > 0 and len(fy_px) > 0:
                        if np.isfinite(fx_px[0]) and np.isfinite(fy_px[0]):
                            cx_i = int(round(fx_px[0]))
                            cy_i = int(round(fy_px[0]))
                            cv2.circle(frame, (cx_i, cy_i), 7, center_color_composite, -1)
            
            # 在右下角添加時間標註
            height, width = frame.shape[:2]
            time_text = f"t={snapshot_times[snap_idx]:.2f}s"
            
            # 設定文字參數
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 2
            text_color = (0, 0, 255)  # 紅色 (BGR)
            
            # 獲取文字大小
            (text_width, text_height), baseline = cv2.getTextSize(
                time_text, font, font_scale, font_thickness
            )
            
            # 計算文字位置（右下角，留一些邊距）
            margin = 10
            text_x = width - text_width - margin
            text_y = height - margin
            
            # 添加文字背景（黑色半透明矩形）
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (text_x - 5, text_y - text_height - 5),
                (text_x + text_width + 5, text_y + baseline + 5),
                (0, 0, 0),
                -1
            )
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # 添加文字
            cv2.putText(
                frame,
                time_text,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA
            )
            
            extracted_frames.append(frame)
        
        cap_composite.release()
        
        # 將 8 張圖水平拼接
        if len(extracted_frames) == 8:
            # 調整每張圖的大小（可選：縮小以便組圖不會太大）
            target_height = 300  # 可以調整這個值
            resized_frames = []
            for frame in extracted_frames:
                h, w = frame.shape[:2]
                scale = target_height / h
                new_width = int(w * scale)
                resized = cv2.resize(frame, (new_width, target_height))
                resized_frames.append(resized)
            
            # 水平拼接
            combined_frame = np.hstack(resized_frames)
            
            # 保存組圖
            composite_path = os.path.join(output_dir, f"{OUT_PREFIX}_8_timepoints_composite.png")
            cv2.imwrite(composite_path, combined_frame)
            print(f"[輸出] 8 個時間點組圖：{composite_path}")
        else:
            print(f"[警告] 只提取到 {len(extracted_frames)} 個畫面，無法生成組圖")

    print(f"\n所有輸出檔案已整理至資料夾：{output_dir}")

if __name__ == "__main__":
    main()
