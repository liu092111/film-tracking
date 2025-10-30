# -*- coding: utf-8 -*-
"""
離線版：壓電致動器 位置/速度 分析（含紅框、即時疊字、CSV、subplot 圖）
新增：
1) 位置圖加起點(綠)與終點(紅) + legend
2) 位置與速度放在同一張圖的左右 subplot
3) 位置圖自動放大視野與等比例顯示，降低抖動視覺感
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 使用者設定 =========
VIDEO_PATH = "IMG_7110.mov"          # ← 換成你的影片
OUT_PREFIX = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

# 色彩遮罩（可視實況微調；這裡是黃＋白兩段合併）
HSV_YELLOW_LO = np.array([15, 60, 120], dtype=np.uint8)
HSV_YELLOW_HI = np.array([35, 255, 255], dtype=np.uint8)
HSV_WHITE_LO  = np.array([0,  0, 200], dtype=np.uint8)
HSV_WHITE_HI  = np.array([180, 60, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 50    # 去除雜訊的最小面積（px^2）
PROCESS_EVERY_N  = 1     # 每 N 幀處理一次（=1 逐幀；=2 每兩幀）
SMOOTH_WIN       = 5     # 速度移動平均視窗（若想更穩可加大）

# 比例（mm/px）設定
AUTO_GRID_MM_PER_PX = True   # True：嘗試用 1cm 格線自動估；False：使用 MANUAL_MM_PER_PX
MANUAL_MM_PER_PX    = None   # 例如 0.172（若 AUTO_GRID_MM_PER_PX=False 就填這裡）

# 位置圖視覺化參數
PLOT_RANGE_SCALE    = 1.35   # 視野放大倍率（>1 代表放大；建議 1.2~1.6）
INVERT_Y_AXIS       = True   # 位置圖是否反轉 Y 軸（較符合一般幾何直覺）

# ============================

def estimate_mm_per_px_single_frame(frame):
    """單幀嘗試以 1cm 格線估 mm/px，失敗回傳 None。"""
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
        if abs(ang) < 10:             # 水平
            horiz.extend([y1, y2])
        elif abs(abs(ang)-90) < 10:   # 垂直
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
        px_per_cm = (sp_h + sp_v) / 2.0
    elif sp_h:
        px_per_cm = sp_h
    elif sp_v:
        px_per_cm = sp_v
    else:
        return None

    return 10.0 / px_per_cm if px_per_cm and px_per_cm > 0 else None  # mm/px

def make_kalman():
    """狀態: [x, y, vx, vy]；量測: [x, y]"""
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

def find_target(frame_bgr):
    """
    色彩遮罩（黃＋白）→ 形態學 → 取最大連通域 → 回傳：
    (cx, cy, x, y, w, h) ；若找不到回傳 None
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

    x, y, w, h = cv2.boundingRect(cnt)
    cx, cy = x + w/2.0, y + h/2.0
    return (cx, cy, x, y, w, h)

def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"找不到影片：{VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("讀取影片第一幀失敗")

    # 估 mm/px
    if AUTO_GRID_MM_PER_PX:
        mm_per_px = estimate_mm_per_px_single_frame(first)
    else:
        mm_per_px = MANUAL_MM_PER_PX
    if mm_per_px is None:
        mm_per_px = 0.1   # 偵測不到格線時的暫定值（可換成你的校正值）

    # 覆寫輸出影片
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = f"{OUT_PREFIX}_tracked.mp4"
    writer = cv2.VideoWriter(out_path, fourcc, max(5.0, fps/PROCESS_EVERY_N), (W, H))

    # Kalman init
    kf = make_kalman()
    init_meas = find_target(first)
    if init_meas is not None:
        cx0, cy0 = init_meas[0], init_meas[1]
    else:
        cx0, cy0 = W/2.0, H/2.0
    kf.statePost = np.array([[cx0], [cy0], [0.0], [0.0]], dtype=np.float32)

    # 逐幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rec = []
    frame_idx = 0
    last_px = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % PROCESS_EVERY_N != 0:
            frame_idx += 1
            continue

        pred = kf.predict()
        meas = find_target(frame)

        if meas is not None:
            cx, cy, x, y, wbox, hbox = meas
            est = kf.correct(np.array([[cx],[cy]], dtype=np.float32))
            fx, fy = float(est[0,0]), float(est[1,0])
            # 像素速度（以濾波後位置）
            if last_px is None:
                vx_px, vy_px = 0.0, 0.0
            else:
                dt = (PROCESS_EVERY_N / fps)
                vx_px = (fx - last_px[0]) / dt
                vy_px = (fy - last_px[1]) / dt
            last_px = (fx, fy)

            # 轉 mm 與 mm/s
            fx_mm, fy_mm = fx * mm_per_px, fy * mm_per_px
            vx_mm_s, vy_mm_s = vx_px * mm_per_px, vy_px * mm_per_px

            # 疊圖：紅框＋中心＋文字
            cv2.rectangle(frame, (x, y), (x + int(wbox), y + int(hbox)), (0,0,255), 2)
            cv2.circle(frame, (int(fx), int(fy)), 5, (0,255,0), -1)
            cv2.putText(frame, f"Position: ({fx_mm:.2f}, {fy_mm:.2f}) mm",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, f"Velocity: ({vx_mm_s:.2f}, {vy_mm_s:.2f}) mm/s",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            fx = fy = np.nan
            fx_mm = fy_mm = np.nan
            vx_mm_s = vy_mm_s = np.nan

        t_s = frame_idx / fps
        rec.append((frame_idx, t_s, fx, fy, fx_mm, fy_mm, vx_mm_s, vy_mm_s, mm_per_px))
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # 匯出 CSV
    df = pd.DataFrame(rec, columns=[
        "frame","t_s",
        "x_px_filt","y_px_filt",
        "x_mm","y_mm",
        "vx_mm_s","vy_mm_s",
        "mm_per_px"
    ])
    csv_path = f"{OUT_PREFIX}_pos_speed.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[輸出] 疊圖影片：{out_path}")
    print(f"[輸出] CSV：{csv_path}")
    print(f"[資訊] 使用 mm/px ≈ {df['mm_per_px'].dropna().iloc[0]:.6f}")

    # =============== 視覺化（新版重點） ===============
    # 先計算總速 |v|
    speed = np.sqrt(df["vx_mm_s"]**2 + df["vy_mm_s"]**2)

    # 建立 1×2 subplot：左=位置，右=速度
    fig, (ax_pos, ax_v) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # ---- 左：位置軌跡 ----
    ax_pos.plot(df["x_mm"], df["y_mm"], lw=2, label="Trajectory")
    # 起點、終點標記
    # 有效點（非 NaN）
    valid = df[["x_mm", "y_mm"]].dropna()
    if len(valid) > 0:
        x_start, y_start = valid.iloc[0]["x_mm"], valid.iloc[0]["y_mm"]
        x_end,   y_end   = valid.iloc[-1]["x_mm"], valid.iloc[-1]["y_mm"]
        ax_pos.scatter([x_start], [y_start], s=80, c="green", marker="o", label="Start")
        ax_pos.scatter([x_end],   [y_end],   s=80, c="red",   marker="o", label="End")

    # 等比例顯示 + 自動放大視野（降低抖動視覺感）
    ax_pos.set_aspect("equal", adjustable="box")
    if len(valid) > 0:
        xmin, xmax = valid["x_mm"].min(), valid["x_mm"].max()
        ymin, ymax = valid["y_mm"].min(), valid["y_mm"].max()
        xc = 0.5 * (xmin + xmax)
        yc = 0.5 * (ymin + ymax)
        half_range = 0.5 * max(xmax - xmin, ymax - ymin)
        half_range = max(half_range, 1e-6)  # 避免零寬
        half_range *= PLOT_RANGE_SCALE      # 放大視野
        ax_pos.set_xlim(xc - half_range, xc + half_range)
        ax_pos.set_ylim(yc - half_range, yc + half_range)

    if INVERT_Y_AXIS:
        ax_pos.invert_yaxis()

    ax_pos.set_xlabel("x (mm)")
    ax_pos.set_ylabel("y (mm)")
    ax_pos.set_title("Position (Trajectory)")
    ax_pos.grid(True, linestyle="--", alpha=0.4)
    ax_pos.legend(loc="best")

    # ---- 右：速度-時間 ----
    ax_v.plot(df["t_s"], speed, lw=2)
    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("Speed (mm/s)")
    ax_v.set_title("Speed vs Time")
    ax_v.grid(True, linestyle="--", alpha=0.4)

    fig.savefig(f"{OUT_PREFIX}_pos_speed_subplot.png", dpi=220)
    plt.close(fig)
    print(f"[輸出] 圖片：{OUT_PREFIX}_pos_speed_subplot.png")

if __name__ == "__main__":
    main()
