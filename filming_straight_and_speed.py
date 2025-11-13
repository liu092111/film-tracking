# -*- coding: utf-8 -*-
"""
離線版：直走 + 旋轉 分析（位置、角度、速度）
------------------------------------------------
本版針對需求調整：
1) Position 軸標籤移除 "aligned" 字樣（但仍可選擇對齊到 +Y）
2) 位置全改為相對起點：第一個有效點視為 (0,0)，影片疊字與 CSV 皆使用相對座標
3) Orientation 圖新增「偏移量（相對起始角）」曲線與 legend
4) Speed 圖新增平均速度（水平虛線）與最大速度（★標記），並顯示數值
輸出檔名維持：
  *_tracked.mp4、*_pos_angle_speed.csv、*_position.png、*_speed_orientation.png
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 使用者設定 =========
GRID_SPACING_MM = 5.0
VIDEO_PATH = "IMG_7129.mov"            # ← 換成你的影片
OUT_PREFIX = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
SPEED_THRESH_ACTIVE = 0.1  # mm/s，低於此速度視為靜止

# 色彩遮罩
HSV_YELLOW_LO = np.array([15, 60, 120], dtype=np.uint8)
HSV_YELLOW_HI = np.array([35, 255, 255], dtype=np.uint8)
HSV_WHITE_LO  = np.array([0,  0, 200], dtype=np.uint8)
HSV_WHITE_HI  = np.array([180, 60, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 50
PROCESS_EVERY_N  = 1                   # 保持逐幀處理（你要 120FPS）
SMOOTH_WIN_ANGLE = 5
SMOOTH_WIN_VEL   = 5                   # 速度平滑視窗（移動平均），1~3 更即時

# 比例（mm/px）
AUTO_GRID_MM_PER_PX = True
MANUAL_MM_PER_PX    = None

# 位置圖與朝向圖的視覺化選項
INVERT_Y_AXIS       = True             # 位置圖是否反轉 Y 軸（多數影像座標慣例）
ALIGN_TRAJ_TO_Y     = True             # Position 圖自動旋轉，使主要移動方向貼齊 +Y
ORIENT_PLOT_WRAPPED = True             # Orientation 圖用 wrapped(-90~90)
ORIENT_YLIM_DEG     = 60               # Orientation 圖 Y 軸 ±限制（None=自動）

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

def wrap_angles_deg(angles_deg):
    """包到 (-90,90] 以利視覺化正常化"""
    a = ((angles_deg + 90) % 180) - 90
    return a

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

    if rw >= rh:
        angle_deg = rect_angle + 90.0
    else:
        angle_deg = rect_angle

    while angle_deg >= 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0

    x, y, w, h = cv2.boundingRect(cnt)
    return (cx, cy, x, y, w, h, angle_deg, cnt)

def create_output_directory(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"{base_name}_analysis_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def finite_diff(values, t):
    """對齊於原序列的簡單微分：v[i] 來自 (i)-(i-1) 差分，首元素為 NaN。"""
    v = np.full_like(values, np.nan, dtype=float)
    for i in range(1, len(values)):
        if np.isfinite(values[i]) and np.isfinite(values[i-1]) and (t[i] > t[i-1]):
            v[i] = (values[i] - values[i-1]) / (t[i] - t[i-1])
    return v

def principal_direction_xy(x, y):
    """用 PCA 找主要移動方向（以 mm 座標），回傳旋轉角度 phi（弧度），
       使得旋轉後的主方向對齊 +Y。"""
    pts = np.vstack([x, y]).T
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 2:
        return 0.0
    pts_center = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts_center.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argmax(eigvals)
    v = eigvecs[:, idx]  # 主軸向量
    # v 指向的角度相對 +X 的夾角：
    ang = np.arctan2(v[1], v[0])  # 弧度
    # 想讓主軸旋到 +Y（角度為 +90 度 = pi/2）
    phi = (np.pi/2) - ang
    return phi

def rotate_xy(x, y, phi):
    """將座標旋轉 phi（弧度），回傳 x', y'。"""
    c, s = np.cos(phi), np.sin(phi)
    xr = c * x - s * y
    yr = s * x + c * y
    return xr, yr

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
    out_path = os.path.join(output_dir, f"{OUT_PREFIX}_tracked.mp4")  # 檔名維持（無 rotation）
    writer = cv2.VideoWriter(out_path, fourcc, max(5.0, fps/PROCESS_EVERY_N), (W, H))

    # Kalman init
    kf = make_kalman()
    init_meas = find_target_and_angle(first)
    if init_meas is not None:
        cx0, cy0 = float(init_meas[0]), float(init_meas[1])
    else:
        cx0, cy0 = W/2.0, H/2.0
    kf.statePost = np.array([[cx0], [cy0], [0.0], [0.0]], dtype=np.float32)

    # 逐幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rec = []  # frame, t_s, x_px_filt, y_px_filt, x_mm, y_mm, angle_deg_raw, mm_per_px
    frame_idx = 0

    # 影片疊字的即時速度估計
    last_fx_mm = None
    last_fy_mm = None
    last_t     = None
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
        if meas is not None:
            cx, cy, x, y, wbox, hbox, angle_deg, cnt = meas
            est = kf.correct(np.array([[cx],[cy]], dtype=np.float32))
            fx, fy = float(est[0,0]), float(est[1,0])

            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 4)
            if np.isfinite(fx) and np.isfinite(fy):
                cv2.circle(frame, (int(round(fx)), int(round(fy))), 6, (0, 255, 0), -1)
        else:
            fx = fy = np.nan

        fx_mm = fx * mm_per_px if np.isfinite(fx) else np.nan
        fy_mm = fy * mm_per_px if np.isfinite(fy) else np.nan

        # 即時速度（用上一幀）
        t_s = frame_idx / fps
        if last_fx_mm is not None and last_fy_mm is not None and last_t is not None and \
           np.isfinite(fx_mm) and np.isfinite(fy_mm) and (t_s > last_t):
            dx = fx_mm - last_fx_mm
            dy = fy_mm - last_fy_mm
            inst_speed = np.hypot(dx, dy) / (t_s - last_t)  # mm/s

        last_fx_mm, last_fy_mm, last_t = fx_mm, fy_mm, t_s

        # 暫存（先存絕對 mm，稍後做「相對起點」轉換）
        rec.append((frame_idx, t_s, fx, fy, fx_mm, fy_mm, angle_deg, mm_per_px))
        frame_idx += 1

    cap.release()

    # ---- 匯總為 DataFrame ----
    df = pd.DataFrame(rec, columns=[
        "frame","t_s","x_px_filt","y_px_filt","x_mm_abs","y_mm_abs","angle_deg_raw","mm_per_px"
    ])

    # === 相對起點（0,0）座標（CSV 與視覺化、影片疊字都使用相對座標） ===  # NEW
    x_abs = df["x_mm_abs"].to_numpy()
    y_abs = df["y_mm_abs"].to_numpy()
    valid = ~np.isnan(x_abs) & ~np.isnan(y_abs)
    if valid.any():
        x0 = x_abs[valid][0]
        y0 = y_abs[valid][0]
    else:
        x0 = y0 = 0.0
    x_mm = x_abs - x0
    y_mm = y_abs - y0
    df["x_mm"] = x_mm
    df["y_mm"] = y_mm

    # 角度處理：保存 unwrapped 供數值、wrapped 供視覺化
    angle_raw = df["angle_deg_raw"].to_numpy()
    mask_ang  = np.isfinite(angle_raw)
    angle_unwrapped = np.full_like(angle_raw, np.nan, dtype=float)
    if mask_ang.any():
        angle_unwrapped[mask_ang] = unwrap_angles_deg(angle_raw[mask_ang])
        angle_unwrapped = moving_average(angle_unwrapped, SMOOTH_WIN_ANGLE)
    df["angle_deg_unwrapped"] = angle_unwrapped

    # === 速度（基於相對座標） ===
    t   = df["t_s"].to_numpy()
    vx = finite_diff(x_mm, t)  # mm/s
    vy = finite_diff(y_mm, t)  # mm/s
    vx = moving_average(vx, SMOOTH_WIN_VEL)
    vy = moving_average(vy, SMOOTH_WIN_VEL)
    speed = np.sqrt(vx**2 + vy**2)
    df["vx_mm_s"] = vx
    df["vy_mm_s"] = vy
    df["speed_mm_s"] = speed

    # === 以時間加權、且扣除靜止期的平均速度 ===
    t = df["t_s"].to_numpy()
    sp = df["speed_mm_s"].to_numpy()

    # 建立 dt（區間長度），對齊到索引 i（代表 i-1 -> i 的區間）
    dt = np.zeros_like(t, dtype=float)
    dt[1:] = np.maximum(0.0, t[1:] - t[:-1])

    active = np.isfinite(sp) & (sp >= SPEED_THRESH_ACTIVE) & (dt > 0)
    if np.any(active):
        avg_speed_active = float(np.nansum(sp[active] * dt[active]) / np.nansum(dt[active]))
    else:
        avg_speed_active = float("nan")

    # === 平均偏移角（相對起始角），只要一個數值 ===
    # 用視覺化同款的 wrapped 角度來定義 offset
    if ORIENT_PLOT_WRAPPED:
        ang_vis = wrap_angles_deg(df["angle_deg_unwrapped"].to_numpy())
    else:
        ang_vis = df["angle_deg_unwrapped"].to_numpy()

    if np.isfinite(ang_vis).any():
        first_idx = int(np.where(np.isfinite(ang_vis))[0][0])
        ang0 = float(ang_vis[first_idx])
        offset_series = wrap_angles_deg(ang_vis - ang0)

        # 用時間加權平均會更穩健（和上面的速度一致），也可改成簡單平均
        off_valid = np.isfinite(offset_series) & (dt >= 0)
        if np.any(off_valid):
            avg_offset_deg = float(np.nansum(offset_series[off_valid] * dt[off_valid]) /
                                np.nansum(dt[off_valid]))
        else:
            avg_offset_deg = float("nan")
    else:
        avg_offset_deg = float("nan")

    # 重新輸出 CSV（仍為相同檔名）
    csv_path = os.path.join(output_dir, f"{OUT_PREFIX}_pos_angle_speed.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # === 重新寫回影片（用相對座標疊字 + 即時速度） ===   # CHG：影片疊字也顯示相對座標
    cap = cv2.VideoCapture(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, max(5.0, fps/PROCESS_EVERY_N), (W, H))

    # 用表格逐幀疊字
    row_map = {int(r.frame): i for i, r in df.iterrows()}
    inst_speed = np.nan
    last_rel_x = None
    last_rel_y = None
    last_t = None
    fidx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % PROCESS_EVERY_N != 0:
            fidx += 1
            continue

        if fidx in row_map:
            i = row_map[fidx]
            relx = df.loc[i, "x_mm"]
            rely = df.loc[i, "y_mm"]
            ang  = df.loc[i, "angle_deg_raw"]
            t_s  = df.loc[i, "t_s"]

            meas2 = find_target_and_angle(frame)
            if meas2 is not None:
                _, _, _, _, _, _, _, cnt2 = meas2
                cv2.drawContours(frame, [cnt2], -1, (0, 0, 255), 4)

            fx_px = df.loc[i, "x_px_filt"]
            fy_px = df.loc[i, "y_px_filt"]
            if np.isfinite(fx_px) and np.isfinite(fy_px):
                cv2.circle(frame, (int(round(fx_px)), int(round(fy_px))), 6, (0, 255, 0), -1)


            # 即時速度（相對座標）
            if last_rel_x is not None and last_rel_y is not None and last_t is not None and \
               np.isfinite(relx) and np.isfinite(rely) and (t_s > last_t):
                dx = relx - last_rel_x
                dy = rely - last_rel_y
                inst_speed = np.hypot(dx, dy) / (t_s - last_t)
            last_rel_x, last_rel_y, last_t = relx, rely, t_s

            cv2.putText(frame, f"Pos(mm): ({relx:.2f}, {rely:.2f})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            if np.isnan(ang):
                cv2.putText(frame, f"Angle: NaN",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                cv2.putText(frame, f"Angle: {ang:+.2f} deg",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            if np.isnan(inst_speed):
                cv2.putText(frame, f"Speed: NaN mm/s",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                cv2.putText(frame, f"Speed: {inst_speed:.2f} mm/s",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        writer.write(frame)
        fidx += 1

    cap.release()
    writer.release()

    print(f"[輸出] 追蹤影片：{out_path}")
    print(f"[輸出] 數據 CSV：{csv_path}")
    try:
        mmpp_preview = df['mm_per_px'].dropna().iloc[0]
        print(f"[資訊] 使用 mm/px ≈ {mmpp_preview:.6f}")
    except Exception:
        pass

    # ====== 視覺化 ======
    # 圖A：Position（可選對齊 +Y；軸標籤不再顯示 aligned 字樣）   # CHG
    figA, ax_pos = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)

    x_plot = df["x_mm"].to_numpy()
    y_plot = df["y_mm"].to_numpy()
    if ALIGN_TRAJ_TO_Y:
        phi = principal_direction_xy(x_plot, y_plot)   # 旋轉角（弧度），把主方向對齊 +Y
        x_plot, y_plot = rotate_xy(x_plot, y_plot, phi)

    ax_pos.plot(x_plot, y_plot, lw=2, label="Trajectory")
    valid_pos = np.vstack([x_plot, y_plot]).T
    valid_pos = valid_pos[~np.isnan(valid_pos).any(axis=1)]
    if len(valid_pos) > 0:
        x_start, y_start = valid_pos[0,0], valid_pos[0,1]
        x_end,   y_end   = valid_pos[-1,0], valid_pos[-1,1]
        ax_pos.scatter([x_start], [y_start], s=100, c="green", marker="o", label="Start", zorder=5)
        ax_pos.scatter([x_end],   [y_end],   s=100, c="red",   marker="o", label="End",   zorder=5)

        # 自動設限讓畫面集中
        xmin, xmax = np.nanmin(x_plot), np.nanmax(x_plot)
        ymin, ymax = np.nanmin(y_plot), np.nanmax(y_plot)
        xc = 0.5 * (xmin + xmax)
        yc = 0.5 * (ymin + ymax)
        half_range = 0.5 * max(xmax - xmin, ymax - ymin)
        half_range = max(half_range, 1e-6) * 1.35
        ax_pos.set_xlim(xc - half_range, xc + half_range)
        ax_pos.set_ylim(yc - half_range, yc + half_range)

    if INVERT_Y_AXIS:
        ax_pos.invert_yaxis()
    ax_pos.set_aspect("equal", adjustable="box")
    ax_pos.set_xlabel("x (mm)")   # CHG: 移除 aligned
    ax_pos.set_ylabel("y (mm)")   # CHG: 移除 aligned
    ax_pos.set_title("Position (Trajectory)")
    ax_pos.grid(True, linestyle="--", alpha=0.4)
    ax_pos.legend(loc="best")

    plot_pos_path = os.path.join(output_dir, f"{OUT_PREFIX}_position.png")
    figA.savefig(plot_pos_path, dpi=220)
    plt.close(figA)
    print(f"[輸出] 圖片：{plot_pos_path}")

    # 圖B：Speed（含平均/最大） + Orientation（含偏移量）  # NEW
    figB, (ax_s, ax_a) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # 左：Speed
    ax_s.plot(df["t_s"], df["speed_mm_s"], lw=2, label="Speed (mm/s)")

    # 最大速度以「紅點」標出（取全域最大，若想只在 active 範圍內取最大，可把條件改成 active）
    sp_all = df["speed_mm_s"].to_numpy()
    t_all  = df["t_s"].to_numpy()
    finite_mask = np.isfinite(sp_all)
    if np.any(finite_mask):
        i_max = int(np.nanargmax(sp_all[finite_mask]))
        # 對應到原始索引
        idxs = np.where(finite_mask)[0]
        i_max_abs = idxs[i_max]
        max_sp = float(sp_all[i_max_abs])
        t_at_max = float(t_all[i_max_abs])

        ax_s.plot([t_at_max], [max_sp], marker="o", markersize=8, color="red",
                label=f"Max: {max_sp:.2f} mm/s @ {t_at_max:.2f}s")

    ax_s.set_xlabel("Time (s)")
    ax_s.set_ylabel("Speed (mm/s)")
    ax_s.set_title("Speed vs Time")
    ax_s.grid(True, linestyle="--", alpha=0.4)
    ax_s.legend(loc="best")


    # 右：Orientation（視覺化用 wrapped）
    if ORIENT_PLOT_WRAPPED:
        ang_vis_plot = wrap_angles_deg(df["angle_deg_unwrapped"].to_numpy())
    else:
        ang_vis_plot = df["angle_deg_unwrapped"].to_numpy()
    if ORIENT_PLOT_WRAPPED:
        ang_vis_plot = wrap_angles_deg(df["angle_deg_unwrapped"].to_numpy())
    else:
        ang_vis_plot = df["angle_deg_unwrapped"].to_numpy()

    # 在 legend 中直接附註平均偏移角
    if np.isfinite(avg_offset_deg):
        label_orient = f"Orientation (deg)\nAvg offset: {avg_offset_deg:.2f}°"
    else:
        label_orient = "Orientation (deg)"

    ax_a.plot(df["t_s"], ang_vis_plot, lw=2, label=label_orient)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Angle (deg)")
    ax_a.set_title("Orientation vs Time")
    if ORIENT_YLIM_DEG is not None and np.isfinite(ORIENT_YLIM_DEG):
        ylim = float(ORIENT_YLIM_DEG)
        ax_a.set_ylim(-ylim, +ylim)
    ax_a.grid(True, linestyle="--", alpha=0.4)
    ax_a.legend(loc="best")


    plot_so_path = os.path.join(output_dir, f"{OUT_PREFIX}_speed_orientation.png")
    figB.savefig(plot_so_path, dpi=220)
    plt.close(figB)
    print(f"[輸出] 圖片：{plot_so_path}")

    print(f"\n所有輸出檔案已整理至資料夾：{output_dir}")

if __name__ == "__main__":
    main()
