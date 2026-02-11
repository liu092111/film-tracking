# -*- coding: utf-8 -*-
"""
Real-time Camera Tracking (Preview→Record→Finish) with Robust Smoothing
----------------------------------------------------------------------
改動重點：
- 開啟先 PREVIEW，按一次空白鍵開始錄；再按一次空白鍵直接結束並輸出
- 輸出資料夾在本地：YYYYMMDD_HHMMSS_<MODE>/
- 更穩追蹤：Kalman(強化)、EMA 平滑、離群保護、旋轉矩形紅框
- 錄影 1:1 實速（真實時間戳估 FPS）

操作：
  [Space]：PREVIEW→開始錄影；RECORDING→結束與輸出
  [q] 或 [ESC]：隨時結束與輸出
"""

import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
try:
    from scipy.signal import savgol_filter
    HAVE_SG = True
except Exception:
    HAVE_SG = False

# ========== 使用者設定 ==========
MODE              = "straight"   # "straight" 或 "rotation"
CAMERA_INDEX      = 1
CAM_WIDTH         = 1280
CAM_HEIGHT        = 720
CAM_FPS_REQ       = 120
RECORD_OUTPUT     = True
WINDOW_TITLE      = "Live Tracking (Preview→Space to Start)"

# 尺度（mm/px）：若畫面有 5 mm 方格可自動估
GRID_SPACING_MM   = 5.0
AUTO_GRID_MM_PER_PX = True
MANUAL_MM_PER_PX     = None   # 例：0.1

# 顏色遮罩（黃 + 白）
HSV_YELLOW_LO = np.array([15, 60, 120], dtype=np.uint8)
HSV_YELLOW_HI = np.array([35, 255, 255], dtype=np.uint8)
HSV_WHITE_LO  = np.array([0,  0, 200], dtype=np.uint8)
HSV_WHITE_HI  = np.array([180, 60, 255], dtype=np.uint8)

MIN_CONTOUR_AREA   = 50
PROCESS_EVERY_N    = 1          # 請保持 1
INVERT_Y_AXIS      = True
ORIENT_PLOT_WRAPPED= True       # straight 模式用
ORIENT_YLIM_DEG    = 60
PLOT_RANGE_SCALE   = 1.35

# —— 追蹤穩定化參數 —— #
# Kalman 噪聲：調整為更穩定的設定
KF_PROCESS_NOISE   = 1e-3   # 降低過程噪聲，增加穩定性
KF_MEASURE_NOISE   = 1e-2   # 適度測量噪聲

# 指數平滑（EMA）：調整為更平滑的參數
EMA_ALPHA_POS      = 0.25   # 中心點平滑
EMA_ALPHA_ANGLE    = 0.20   # 角度平滑

# Savitzky–Golay（可選）：短窗波形平滑
USE_SG_POS         = False
SG_WIN             = 7
SG_POLY            = 2

# 離群保護（px）：適當放寬以減少過濾
MAX_MEAS_JUMP_PX   = 80

# 暖身估 FPS 幀數
WARMUP_FRAMES_FOR_FPS = 30
# ==============================

def estimate_mm_per_px_single_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=60, maxLineGap=10)
    if lines is None or len(lines) < 8:
        return None

    horiz, vert = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx, dy = x2 - x1, y2 - y1
        ang = np.degrees(np.arctan2(dy, dx))
        if ang < -90: ang += 180
        if ang >  90: ang -= 180
        if abs(ang) < 10:
            horiz.extend([y1, y2])
        elif abs(abs(ang) - 90) < 10:
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

    sp_h = spacing(horiz); sp_v = spacing(vert)
    if sp_h and sp_v: px_per_cell = (sp_h + sp_v)/2.0
    elif sp_h:        px_per_cell = sp_h
    elif sp_v:        px_per_cell = sp_v
    else:             return None
    return (GRID_SPACING_MM / px_per_cell) if (px_per_cell and px_per_cell > 0) else None

def unwrap_angles_deg(a_deg):
    if len(a_deg) == 0: return a_deg
    return np.rad2deg(np.unwrap(np.deg2rad(a_deg)))

def wrap_angles_deg(a_deg):
    return ((a_deg + 90) % 180) - 90

def moving_average(a, w):
    if w is None or w <= 1: return a
    a = np.asarray(a, dtype=float)
    if len(a) < w: return a
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
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * KF_PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KF_MEASURE_NOISE
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
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_CONTOUR_AREA: return None

    rect = cv2.minAreaRect(cnt)  # (center(x,y), (w,h), angle)
    (cx, cy), (rw, rh), rect_angle = rect
    if rw >= rh: angle_deg = rect_angle + 90.0
    else:        angle_deg = rect_angle
    while angle_deg >= 90.0: angle_deg -= 180.0
    while angle_deg <  -90.0: angle_deg += 180.0

    # 用旋轉矩形四點取代完整輪廓（較穩定）
    box = cv2.boxPoints(rect).astype(int)
    return (cx, cy, angle_deg, box)

def finite_diff(values, t, smooth_win=1):
    v = np.full_like(values, np.nan, dtype=float)
    for i in range(1, len(values)):
        if np.isfinite(values[i]) and np.isfinite(values[i-1]) and (t[i] > t[i-1]):
            v[i] = (values[i] - values[i-1]) / (t[i] - t[i-1])
    if smooth_win and smooth_win > 1:
        v = moving_average(v, smooth_win)
    return v

def ema(prev, cur, alpha):
    if prev is None or not np.isfinite(prev): return cur
    if not np.isfinite(cur): return prev
    return alpha*prev + (1.0-alpha)*cur

def main():
    # === 輸出資料夾（本地） ===
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{run_tag}_{MODE}"
    os.makedirs(output_dir, exist_ok=True)
    OUT_PREFIX = f"camera_{MODE}"

    # === 相機 ===
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS_REQ)
    for _ in range(5): cap.read()  # 暖機

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("攝影機開啟失敗或抓不到影像")

    fps_device = cap.get(cv2.CAP_PROP_FPS) or float(CAM_FPS_REQ)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # === mm/px ===
    if AUTO_GRID_MM_PER_PX:
        mm_per_px = estimate_mm_per_px_single_frame(first)
    else:
        mm_per_px = MANUAL_MM_PER_PX
    if mm_per_px is None: mm_per_px = 0.1

    # === 狀態機：PREVIEW → RECORDING ===
    STATE_PREVIEW  = 0
    STATE_RECORD   = 1
    state = STATE_PREVIEW

    # === 錄影（延後建立，估到實 FPS 後） ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_dir, f"{OUT_PREFIX}_tracked.mp4")
    writer = None
    fps_out = None
    frame_buf = []
    ts_list   = []

    # === Kalman / 平滑器 ===
    kf = make_kalman()
    init = find_target_and_angle(first)
    if init is not None:
        cx0, cy0, ang0, box0 = init
    else:
        cx0, cy0, ang0 = W/2.0, H/2.0, 0.0
    kf.statePost = np.array([[cx0],[cy0],[0.0],[0.0]], dtype=np.float32)

    # EMA 狀態
    ema_x = None; ema_y = None; ema_ang = None

    # 資料紀錄（只在 RECORDING 時累積）
    rec = []  # (frame, t_s, x_px_filt,y_px_filt, x_mm_abs,y_mm_abs, angle_deg_raw, mm_per_px)
    frame_idx = 0
    origin_x = None; origin_y = None

    # 疊字速度（僅用於顯示）
    last_x_mm_abs = None; last_y_mm_abs = None; last_t = None
    inst_speed = np.nan

    # 顯示窗
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, CAM_WIDTH//2, CAM_HEIGHT//2)

    # 短歷史（去抖參考）
    last_pred = None

    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame_idx % PROCESS_EVERY_N != 0:
            frame_idx += 1; continue

        # 預測
        pred = kf.predict()
        px_pred, py_pred = float(pred[0,0]), float(pred[1,0])
        last_pred = (px_pred, py_pred)

        # 量測
        m = find_target_and_angle(frame)
        use_meas = True
        if m is not None:
            cx, cy, angle_deg, box = m
            # 離群保護：量測跳太遠就忽略（用預測）
            dist = np.hypot(cx - px_pred, cy - py_pred)
            if dist > MAX_MEAS_JUMP_PX:
                use_meas = False
        else:
            use_meas = False

        if use_meas:
            est = kf.correct(np.array([[cx],[cy]], dtype=np.float32))
            fx_raw, fy_raw = float(est[0,0]), float(est[1,0])
        else:
            # 沒量測或離群 → 用預測值
            fx_raw, fy_raw = px_pred, py_pred
            angle_deg = np.nan
            box = None

        # EMA 平滑中心與角度（畫面顯示與輸出都用平滑後）
        ema_x = ema(ema_x, fx_raw, EMA_ALPHA_POS)
        ema_y = ema(ema_y, fy_raw, EMA_ALPHA_POS)
        ema_ang = ema(ema_ang, angle_deg, EMA_ALPHA_ANGLE) if np.isfinite(angle_deg) else ema_ang

        fx, fy = float(ema_x), float(ema_y)
        ang_for_draw = float(ema_ang) if ema_ang is not None else (float(angle_deg) if m is not None else np.nan)

        # 旋轉矩形（若有量測才畫；否則畫預測附近的小框）
        if box is None:
            # 用預測值畫固定小框，避免閃爍
            sz = 20
            bx = np.array([[fx-sz, fy-sz],[fx+sz, fy-sz],[fx+sz, fy+sz],[fx-sz, fy+sz]], dtype=int)
            box_draw = bx
            # 綠點位置：預測中心
            green_center_x, green_center_y = fx, fy
        else:
            box_draw = box
            # 綠點位置：紅框的實際中心（與檢測中心一致）
            green_center_x, green_center_y = cx, cy

        # —— 疊圖 —— #
        # 紅框（旋轉矩形）
        cv2.polylines(frame, [box_draw], isClosed=True, color=(0,0,255), thickness=3)
        # 綠點（固定在紅框中心，消除 shifting）
        if np.isfinite(green_center_x) and np.isfinite(green_center_y):
            cv2.circle(frame, (int(round(green_center_x)), int(round(green_center_y))), 6, (0,255,0), -1)

        # 絕對座標（mm）
        fx_mm_abs = fx * mm_per_px if np.isfinite(fx) else np.nan
        fy_mm_abs = fy * mm_per_px if np.isfinite(fy) else np.nan

        # 疊字時間（顯示用）
        t_s = frame_idx / (fps_device if fps_device > 0 else 30.0)

        # 疊字速度（用絕對 mm）
        if (last_x_mm_abs is not None and last_y_mm_abs is not None and last_t is not None and
            np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs) and (t_s > last_t)):
            inst_speed = np.hypot(fx_mm_abs-last_x_mm_abs, fy_mm_abs-last_y_mm_abs) / (t_s - last_t)
        last_x_mm_abs, last_y_mm_abs, last_t = fx_mm_abs, fy_mm_abs, t_s

        # 相對零點（第一個有效點）
        if origin_x is None and np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs):
            origin_x, origin_y = fx_mm_abs, fy_mm_abs

        # 疊字
        if origin_x is not None and np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs):
            rx = fx_mm_abs - origin_x
            ry = fy_mm_abs - origin_y
            pos_text = f"Pos(mm): ({rx:.2f}, {ry:.2f})"
        else:
            pos_text = f"Pos(mm): (NaN, NaN)"
        cv2.putText(frame, pos_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        if np.isfinite(ang_for_draw):
            cv2.putText(frame, f"Angle: {ang_for_draw:+.2f} deg", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(frame, "Angle: NaN", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        if np.isfinite(inst_speed):
            cv2.putText(frame, f"Speed: {inst_speed:.2f} mm/s", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(frame, "Speed: NaN mm/s", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # PREVIEW 提示
        if state == STATE_PREVIEW:
            cv2.putText(frame, "Preview - Press [SPACE] to START recording",
                        (20, H-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # 顯示
        cv2.imshow(WINDOW_TITLE, frame)

        # 錄影流程：只有 RECORDING 時才寫檔與記錄資料
        if state == STATE_RECORD and RECORD_OUTPUT:
            ts_now = time.perf_counter()
            ts_list.append(ts_now)
            if writer is None:
                frame_buf.append(frame.copy())
                if len(frame_buf) >= WARMUP_FRAMES_FOR_FPS and len(ts_list) >= WARMUP_FRAMES_FOR_FPS:
                    dt = np.diff(np.array(ts_list[-WARMUP_FRAMES_FOR_FPS:], dtype=float))
                    dt = dt[dt > 0]
                    fps_out = float(1.0 / np.median(dt)) if dt.size > 0 else (fps_device or 30.0)
                    writer = cv2.VideoWriter(out_path, fourcc, fps_out, (W, H))
                    for f in frame_buf:
                        writer.write(f)
                    frame_buf.clear()
            else:
                writer.write(frame)

        # RECORDING：記錄資料
        if state == STATE_RECORD:
            rec.append((
                frame_idx, t_s,
                fx, fy,
                fx_mm_abs, fy_mm_abs,
                float(ang_for_draw) if np.isfinite(ang_for_draw) else np.nan,
                mm_per_px
            ))

        # 按鍵
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC 或 q 隨時結束
            break
        if key == 32:  # Space
            if state == STATE_PREVIEW:
                state = STATE_RECORD
            else:
                # 第二次空白鍵：結束並輸出
                break

        frame_idx += 1

    # 收尾
    cap.release()
    # 若在 RECORDING 但 writer 尚未建立，補建、估 FPS 後寫出緩衝
    if state == STATE_RECORD and RECORD_OUTPUT and writer is None and len(frame_buf) > 0:
        dt_all = np.diff(np.array(ts_list, dtype=float))
        dt_all = dt_all[dt_all > 0]
        fps_out = float(1.0 / np.median(dt_all)) if dt_all.size > 0 else (fps_device or 30.0)
        writer = cv2.VideoWriter(out_path, fourcc, fps_out, (W, H))
        for f in frame_buf:
            writer.write(f)
        frame_buf.clear()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # 如果沒有真的錄（停在 PREVIEW 就離開），就不做輸出
    if state != STATE_RECORD or len(rec) == 0:
        print("未開始錄影，沒有輸出。")
        return

    # === 匯出 CSV（整理資料）===
    df = pd.DataFrame(rec, columns=[
        "frame","t_s","x_px_filt","y_px_filt","x_mm_abs","y_mm_abs","angle_deg_raw","mm_per_px"
    ])

    # 相對起點（0,0）
    x_abs = df["x_mm_abs"].to_numpy(); y_abs = df["y_mm_abs"].to_numpy()
    valid = np.isfinite(x_abs) & np.isfinite(y_abs)
    if valid.any():
        x0, y0 = x_abs[valid][0], y_abs[valid][0]
        df["x_mm"] = x_abs - x0
        df["y_mm"] = y_abs - y0
    else:
        df["x_mm"] = x_abs
        df["y_mm"] = y_abs

    # 可選：Savitzky–Golay 再平滑座標（減抖；太快時可開）
    if USE_SG_POS and HAVE_SG and len(df) >= SG_WIN:
        df["x_mm"] = savgol_filter(df["x_mm"].to_numpy(), SG_WIN, SG_POLY, mode="interp")
        df["y_mm"] = savgol_filter(df["y_mm"].to_numpy(), SG_WIN, SG_POLY, mode="interp")

    # 角度展開 + 平滑
    ang_raw = df["angle_deg_raw"].to_numpy()
    mask_ang = np.isfinite(ang_raw)
    ang_unwrap = np.full_like(ang_raw, np.nan, dtype=float)
    if mask_ang.any():
        ang_unwrap[mask_ang] = unwrap_angles_deg(ang_raw[mask_ang])
        ang_unwrap = moving_average(ang_unwrap, 5)
    df["angle_deg_unwrapped"] = ang_unwrap

    # 角速度（deg/s）
    t = df["t_s"].to_numpy()
    ang_vel = np.full_like(ang_unwrap, np.nan, dtype=float)
    idx = np.where(mask_ang)[0]
    if len(idx) >= 2:
        for i0, i1 in zip(idx[:-1], idx[1:]):
            dt = t[i1] - t[i0]
            if dt > 0:
                ang_vel[i1] = (ang_unwrap[i1] - ang_unwrap[i0]) / dt
        ang_vel = moving_average(ang_vel, 5)
    df["angular_vel_dps"] = ang_vel

    # 線速度（mm/s，基於相對座標）
    vx = finite_diff(df["x_mm"].to_numpy(), t, smooth_win=5)
    vy = finite_diff(df["y_mm"].to_numpy(), t, smooth_win=5)
    speed = np.sqrt(vx**2 + vy**2)
    df["vx_mm_s"] = vx; df["vy_mm_s"] = vy; df["speed_mm_s"] = speed

    # 輸出 CSV
    csv_path = os.path.join(output_dir, f"{OUT_PREFIX}_pos_angle_speed.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ===== 圖檔輸出 =====
    # A) Position
    figA, ax_pos = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    x_plot = df["x_mm"].to_numpy(); y_plot = df["y_mm"].to_numpy()
    ax_pos.plot(x_plot, y_plot, lw=2, label="Trajectory")
    valid_pos = np.vstack([x_plot, y_plot]).T
    valid_pos = valid_pos[~np.isnan(valid_pos).any(axis=1)]
    if len(valid_pos) > 0:
        ax_pos.scatter([valid_pos[0,0]],[valid_pos[0,1]], s=100, c="green", marker="o", label="Start", zorder=5)
        ax_pos.scatter([valid_pos[-1,0]],[valid_pos[-1,1]], s=100, c="red",   marker="o", label="End",   zorder=5)
        xmin, xmax = np.nanmin(x_plot), np.nanmax(x_plot)
        ymin, ymax = np.nanmin(y_plot), np.nanmax(y_plot)
        xc = 0.5*(xmin+xmax); yc = 0.5*(ymin+ymax)
        half = 0.5*max(xmax-xmin, ymax-ymin); half = max(half, 1e-6)*PLOT_RANGE_SCALE
        ax_pos.set_xlim(xc-half, xc+half); ax_pos.set_ylim(yc-half, yc+half)
    if INVERT_Y_AXIS: ax_pos.invert_yaxis()
    ax_pos.set_aspect("equal", adjustable="box")
    ax_pos.set_xlabel("x (mm)", fontsize=24); ax_pos.set_ylabel("y (mm)", fontsize=24)
    ax_pos.set_title("Position (Trajectory)", fontsize=22)
    #ax_pos.grid(True, linestyle="--", alpha=0.4); ax_pos.legend(loc="best")
    ax_pos.legend(loc="best", fontsize=14)
    plot_pos_path = os.path.join(output_dir, f"{OUT_PREFIX}_position.png")
    figA.savefig(plot_pos_path, dpi=1200, bbox_inches='tight', pad_inches=0.3); plt.close(figA)

    # B) 第二張圖
    if MODE.lower() == "straight":
        figB, (ax_s, ax_a) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        # 左：Speed
        ax_s.plot(df["t_s"], df["speed_mm_s"], lw=2, label="Speed (mm/s)")
        sp_all = df["speed_mm_s"].to_numpy(); tt = df["t_s"].to_numpy()
        finite = np.isfinite(sp_all)
        if np.any(finite):
            i_max = np.nanargmax(sp_all)
            ax_s.plot([tt[i_max]],[sp_all[i_max]], marker="o", markersize=8, color="red",
                      label=f"Max: {sp_all[i_max]:.2f} mm/s @ {tt[i_max]:.2f}s")
        ax_s.set_xlabel("Time (s)", fontsize=24); ax_s.set_ylabel("Speed (mm/s)", fontsize=24)
        ax_s.set_title("Speed vs Time", fontsize=22); #ax_s.grid(True, linestyle="--", alpha=0.4); ax_s.legend(loc="best")
        ax_s.legend(loc="best", fontsize=14)

        # 右：Orientation + Avg offset in legend
        if ORIENT_PLOT_WRAPPED: ang_vis = wrap_angles_deg(df["angle_deg_unwrapped"].to_numpy())
        else:                    ang_vis = df["angle_deg_unwrapped"].to_numpy()
        if np.isfinite(ang_vis).any():
            first_idx = np.where(np.isfinite(ang_vis))[0][0]
            ang0 = float(ang_vis[first_idx])
            offset_series = wrap_angles_deg(ang_vis - ang0)
            avg_offset_deg = float(np.nanmean(offset_series))
        else:
            avg_offset_deg = float("nan")
        label_orient = f"Orientation (deg)\nAvg offset: {avg_offset_deg:.2f}°" if np.isfinite(avg_offset_deg) else "Orientation (deg)"
        ax_a.plot(df["t_s"], ang_vis, lw=2, label=label_orient)
        ax_a.set_xlabel("Time (s)", fontsize=24); ax_a.set_ylabel("Angle (deg)", fontsize=24)
        ax_a.set_title("Orientation vs Time", fontsize=22)
        if ORIENT_YLIM_DEG is not None and np.isfinite(ORIENT_YLIM_DEG):
            ylim = float(ORIENT_YLIM_DEG); ax_a.set_ylim(-ylim, +ylim)
        #ax_a.grid(True, linestyle="--", alpha=0.4); ax_a.legend(loc="best")
        ax_a.legend(loc="best", fontsize=14)

        plot_so_path = os.path.join(output_dir, f"{OUT_PREFIX}_speed_orientation.png")
        figB.savefig(plot_so_path, dpi=1200, bbox_inches='tight', pad_inches=0.3); plt.close(figB)
    else:
        figW, ax_w = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        ax_w.plot(df["t_s"], df["angular_vel_dps"], lw=2, label="Angular speed (deg/s)")
        w_all = df["angular_vel_dps"].to_numpy(); tt = df["t_s"].to_numpy()
        finite = np.isfinite(w_all)
        if np.any(finite):
            idx_rel = int(np.nanargmax(np.abs(w_all[finite]))); idxs = np.where(finite)[0]
            i_max = idxs[idx_rel]
            ax_w.plot([tt[i_max]],[w_all[i_max]], marker="o", markersize=8, color="red",
                      label=f"Max: {w_all[i_max]:.2f} deg/s @ {tt[i_max]:.2f}s")
        ax_w.set_xlabel("Time (s)", fontsize=24); ax_w.set_ylabel("Angular speed (deg/s)", fontsize=24)
        ax_w.set_title("Angular Speed vs Time", fontsize=22)
        #ax_w.grid(True, linestyle="--", alpha=0.4); ax_w.legend(loc="best")
        ax_w.legend(loc="best", fontsize=14)
        plot_w_path = os.path.join(output_dir, f"{OUT_PREFIX}_angular_speed.png")
        figW.savefig(plot_w_path, dpi=1200, bbox_inches='tight', pad_inches=0.3); plt.close(figW)

    # 訊息
    print(f"[輸出] 目錄：{output_dir}")
    print(f"[輸出] CSV：{csv_path}")
    print(f"[輸出] Position 圖：{plot_pos_path}")
    if MODE.lower() == "straight":
        print(f"[輸出] Speed/Orientation 圖：{os.path.join(output_dir, OUT_PREFIX + '_speed_orientation.png')}")
    else:
        print(f"[輸出] Angular Speed 圖：{os.path.join(output_dir, OUT_PREFIX + '_angular_speed.png')}")
    if RECORD_OUTPUT and os.path.exists(out_path):
        print(f"[輸出] 追蹤影片：{out_path}")

if __name__ == "__main__":
    main()
