# -*- coding: utf-8 -*-
"""
離線版：直走 + 旋轉 分析（位置、角度、速度）+ 四頂點軌跡
------------------------------------------------
新增功能：
1) 針對 device 的 minAreaRect 四個頂點，繪製虛線路徑（顏色較淡）
2) 每幀畫出 device 外框（實線），搭配原本中心點路徑與文字疊字
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 使用者設定 =========
GRID_SPACING_MM = 5.0
VIDEO_PATH = "film/IMG_7129.mov"            # ← 換成你的影片
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

def draw_dashed_polyline(img, points, color, thickness=1, dash_len=10, gap_len=5):
    """
    在影像上畫虛線 polyline
    points: list of (x, y)
    """
    if len(points) < 2:
        return
    pts = np.array(points, dtype=np.float32)

    for i in range(1, len(pts)):
        p1 = pts[i - 1]
        p2 = pts[i]
        dist = np.linalg.norm(p2 - p1)
        if dist == 0:
            continue
        direction = (p2 - p1) / dist
        num_dashes = int(dist // (dash_len + gap_len)) + 1

        for d in range(num_dashes):
            start_dist = d * (dash_len + gap_len)
            end_dist = start_dist + dash_len
            if start_dist > dist:
                break
            if end_dist > dist:
                end_dist = dist

            start_point = p1 + direction * start_dist
            end_point   = p1 + direction * end_dist
            cv2.line(
                img,
                (int(start_point[0]), int(start_point[1])),
                (int(end_point[0]), int(end_point[1])),
                color,
                thickness,
            )

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

def make_kalman(init_x=0.0, init_y=0.0):
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
    kf.statePost = np.array([[init_x], [init_y], [0.0], [0.0]], dtype=np.float32)
    return kf

def find_target_and_angle(frame_bgr):
    """
    回傳：
      (cx, cy, x, y, w, h, angle_deg, cnt, box)
    其中 box 為 minAreaRect 的四個頂點 (4,2)
    """
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

    # 角度處理
    if rw >= rh:
        angle_deg = rect_angle + 90.0
    else:
        angle_deg = rect_angle

    while angle_deg >= 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0

    x, y, w, h = cv2.boundingRect(cnt)
    return (cx, cy, x, y, w, h, angle_deg, cnt, box)

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

    # Kalman init for center
    kf_center = make_kalman()
    init_meas = find_target_and_angle(first)
    if init_meas is not None:
        cx0, cy0 = float(init_meas[0]), float(init_meas[1])
    else:
        cx0, cy0 = W/2.0, H/2.0
    kf_center.statePost = np.array([[cx0], [cy0], [0.0], [0.0]], dtype=np.float32)

    # 逐幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rec = []  # frame, t_s, x_px_filt, y_px_filt, x_mm, y_mm, angle_deg_raw, mm_per_px
    box_rec = []  # 記錄每一幀的 box (四個頂點) 用於後續繪製
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

        # Center point Kalman prediction and correction
        pred_center = kf_center.predict()
        meas = find_target_and_angle(frame)

        angle_deg = np.nan
        fx = fy = np.nan

        if meas is not None:
            cx, cy, x, y, wbox, hbox, angle_deg, cnt, box = meas
            est_center = kf_center.correct(np.array([[cx],[cy]], dtype=np.float32))
            fx, fy = float(est_center[0,0]), float(est_center[1,0])

            # 第一輪只簡單畫輪廓（可省略，因為真正輸出的影片在第二輪）
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)

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
        
        # 記錄 box（如果有偵測到）
        if meas is not None:
            box_rec.append((frame_idx, t_s, fx, fy, meas[8]))  # meas[8] 是 box
        else:
            box_rec.append((frame_idx, t_s, fx, fy, None))
        
        frame_idx += 1

    cap.release()

    # ---- 匯總為 DataFrame ----
    df = pd.DataFrame(rec, columns=[
        "frame","t_s","x_px_filt","y_px_filt","x_mm_abs","y_mm_abs","angle_deg_raw","mm_per_px"
    ])

    # === 相對起點（0,0）座標（CSV 與視覺化、影片疊字都使用相對座標） ===
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

    dt = np.zeros_like(t, dtype=float)
    dt[1:] = np.maximum(0.0, t[1:] - t[:-1])

    active = np.isfinite(sp) & (sp >= SPEED_THRESH_ACTIVE) & (dt > 0)
    if np.any(active):
        avg_speed_active = float(np.nansum(sp[active] * dt[active]) / np.nansum(dt[active]))
    else:
        avg_speed_active = float("nan")

    # === 平均偏移角（相對起始角） ===
    if ORIENT_PLOT_WRAPPED:
        ang_vis = wrap_angles_deg(df["angle_deg_unwrapped"].to_numpy())
    else:
        ang_vis = df["angle_deg_unwrapped"].to_numpy()

    if np.isfinite(ang_vis).any():
        first_idx = int(np.where(np.isfinite(ang_vis))[0][0])
        ang0 = float(ang_vis[first_idx])
        offset_series = wrap_angles_deg(ang_vis - ang0)

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

    # === 重新寫回影片（用相對座標疊字 + 即時速度 + 五條路徑） ===
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

    # === 路徑紀錄：中心 + 四頂點 ===
    center_path = []   # [(x,y), ...]
    v0_path, v1_path, v2_path, v3_path = [], [], [], []

    light_color = (200, 200, 0)   # 四頂點虛線顏色較淡
    center_color = (255, 0, 0)    # 中心路徑：藍紅系實線

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % PROCESS_EVERY_N != 0:
            fidx += 1
            continue

        # 嘗試偵測一次，拿到輪廓 & 四頂點
        meas2 = find_target_and_angle(frame)

        box_int = None
        if meas2 is not None:
            cx2, cy2, _, _, _, _, _, cnt2, box = meas2
            box_int = np.int32(box)

            # 畫當前外框（實線）
            cv2.polylines(frame, [box_int], isClosed=True, color=(0, 0, 255), thickness=2)

            # 畫當前輪廓線（可留可不留）
            cv2.drawContours(frame, [cnt2], -1, (0, 0, 255), 1)

        # 取出這一幀在 df 的資料（位置/角度/時間）
        if fidx in row_map:
            i = row_map[fidx]
            relx = df.loc[i, "x_mm"]
            rely = df.loc[i, "y_mm"]
            ang  = df.loc[i, "angle_deg_raw"]
            t_s  = df.loc[i, "t_s"]
            fx_px = df.loc[i, "x_px_filt"]
            fy_px = df.loc[i, "y_px_filt"]

            # 中心點：畫在影像 + 累積路徑
            if np.isfinite(fx_px) and np.isfinite(fy_px):
                cx_i = int(round(fx_px))
                cy_i = int(round(fy_px))
                cv2.circle(frame, (cx_i, cy_i), 6, (0, 255, 0), -1)
                center_path.append((cx_i, cy_i))

            # 四個頂點：若有偵測到 box，累積路徑
            if box_int is not None and len(box_int) == 4:
                v0_path.append(tuple(box_int[0]))
                v1_path.append(tuple(box_int[1]))
                v2_path.append(tuple(box_int[2]))
                v3_path.append(tuple(box_int[3]))

            # 即時速度（相對座標）
            if last_rel_x is not None and last_rel_y is not None and last_t is not None and \
               np.isfinite(relx) and np.isfinite(rely) and (t_s > last_t):
                dx = relx - last_rel_x
                dy = rely - last_rel_y
                inst_speed = np.hypot(dx, dy) / (t_s - last_t)
            last_rel_x, last_rel_y, last_t = relx, rely, t_s

            # 疊字：位置 / 角度 / 當下速度
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

        # === 畫歷史路徑（已移除）===
        # 注意：已移除中心點路徑和四頂點路徑的繪製功能

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
    # 圖A：Position
    figA, ax_pos = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)

    x_plot = df["x_mm"].to_numpy()
    y_plot = df["y_mm"].to_numpy()
    if ALIGN_TRAJ_TO_Y:
        phi = principal_direction_xy(x_plot, y_plot)
        x_plot, y_plot = rotate_xy(x_plot, y_plot, phi)

    ax_pos.plot(x_plot, y_plot, lw=2, label="Trajectory")
    valid_pos = np.vstack([x_plot, y_plot]).T
    valid_pos = valid_pos[~np.isnan(valid_pos).any(axis=1)]
    if len(valid_pos) > 0:
        x_start, y_start = valid_pos[0,0], valid_pos[0,1]
        x_end,   y_end   = valid_pos[-1,0], valid_pos[-1,1]
        ax_pos.scatter([x_start], [y_start], s=100, c="green", marker="o", label="Start", zorder=5)
        ax_pos.scatter([x_end],   [y_end],   s=100, c="red",   marker="o", label="End",   zorder=5)

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
    ax_pos.set_xlabel("x (mm)")
    ax_pos.set_ylabel("y (mm)")
    ax_pos.set_title("Position (Trajectory)")
    ax_pos.grid(True, linestyle="--", alpha=0.4)
    ax_pos.legend(loc="best")

    plot_pos_path = os.path.join(output_dir, f"{OUT_PREFIX}_position.png")
    figA.savefig(plot_pos_path, dpi=220)
    plt.close(figA)
    print(f"[輸出] 圖片：{plot_pos_path}")

    # 圖B：Speed + Orientation
    figB, (ax_s, ax_a) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # 左：Speed
    ax_s.plot(df["t_s"], df["speed_mm_s"], lw=2, label="Speed (mm/s)")

    sp_all = df["speed_mm_s"].to_numpy()
    t_all  = df["t_s"].to_numpy()
    finite_mask = np.isfinite(sp_all)
    if np.any(finite_mask):
        i_max = int(np.nanargmax(sp_all[finite_mask]))
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

    # 右：Orientation
    if ORIENT_PLOT_WRAPPED:
        ang_vis_plot = wrap_angles_deg(df["angle_deg_unwrapped"].to_numpy())
    else:
        ang_vis_plot = df["angle_deg_unwrapped"].to_numpy()

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

    # ====== 新增：軌跡圖（統一虛線、同個顏色）======
    figC, ax_contour = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    
    # 繪製整體路徑（與圖A相同的處理）
    x_plot = df["x_mm"].to_numpy()
    y_plot = df["y_mm"].to_numpy()
    
    if ALIGN_TRAJ_TO_Y:
        phi = principal_direction_xy(x_plot, y_plot)
        x_plot_rot, y_plot_rot = rotate_xy(x_plot, y_plot, phi)
    else:
        phi = 0.0
        x_plot_rot, y_plot_rot = x_plot, y_plot
    
    # 畫整體路徑
    ax_contour.plot(x_plot_rot, y_plot_rot, lw=2, color='blue', label="Trajectory", alpha=0.7)
    
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
                
                # 如果有旋轉，對box頂點也做旋轉
                if ALIGN_TRAJ_TO_Y:
                    box_mm_rot = np.zeros_like(box_mm)
                    for j in range(len(box_mm)):
                        box_mm_rot[j, 0], box_mm_rot[j, 1] = rotate_xy(
                            box_mm[j, 0], box_mm[j, 1], phi
                        )
                else:
                    box_mm_rot = box_mm
                
                # 繪製輪廓（統一淺藍色虛線）
                color = (0.5, 0.7, 1.0)  # 淺藍色
                linestyle = '--'
                linewidth = 1.5
                alpha = 0.7
                
                box_closed = np.vstack([box_mm_rot, box_mm_rot[0:1]])
                ax_contour.plot(box_closed[:, 0], box_closed[:, 1], 
                              linestyle=linestyle, linewidth=linewidth, 
                              color=color, alpha=alpha)
    
    # 標記起點和終點
    valid_pos = np.vstack([x_plot_rot, y_plot_rot]).T
    valid_pos = valid_pos[~np.isnan(valid_pos).any(axis=1)]
    if len(valid_pos) > 0:
        x_start, y_start = valid_pos[0, 0], valid_pos[0, 1]
        x_end, y_end = valid_pos[-1, 0], valid_pos[-1, 1]
        ax_contour.scatter([x_start], [y_start], s=100, c="green", 
                          marker="o", label="Start", zorder=5)
        ax_contour.scatter([x_end], [y_end], s=100, c="red", 
                          marker="o", label="End", zorder=5)
        
        # 設置座標軸範圍
        xmin, xmax = np.nanmin(x_plot_rot), np.nanmax(x_plot_rot)
        ymin, ymax = np.nanmin(y_plot_rot), np.nanmax(y_plot_rot)
        xc = 0.5 * (xmin + xmax)
        yc = 0.5 * (ymin + ymax)
        half_range = 0.5 * max(xmax - xmin, ymax - ymin)
        half_range = max(half_range, 1e-6) * 1.35
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
