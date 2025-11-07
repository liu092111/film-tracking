# -*- coding: utf-8 -*-
"""
離線版：直走 + 旋轉 分析（位置、角度、角速度）
------------------------------------------------
相較於基準檔（僅位置/速度），本版本新增：
1) 以 minAreaRect / PCA 估計目標朝向 angle（deg），並做角度展開（unwrap）
2) 計算角速度 angular_velocity（deg/s）
3) 疊圖改為：左=Position(trajectory)，右=Angular velocity vs Time
4) 疊加固定實體尺寸的「旋轉紅框」：9mm × 6mm（會隨 angle 轉動；以 mm/px 轉換成像素）
5) CSV 另存 angle_deg、angle_deg_unwrapped、angular_vel_dps，供後續分析

使用方法：
- 將 VIDEO_PATH 指向你的影片（例如 IMG_7110.mov）。
- 如能偵測到固定網格（預設 `GRID_SPACING_MM = 5.0 mm`），會自動估 mm/px；否則 fallback 使用 MANUAL_MM_PER_PX 或 0.1。
- 若要強制使用手動比例，將 AUTO_GRID_MM_PER_PX=False 並設定 MANUAL_MM_PER_PX。

注意：
- 角度定義以「minAreaRect 的方塊長邊方向」為主，經矯正到 [-90, 90) 再做展開。
- 當輪廓過小或偵測失敗時，本幀 angle 記為 NaN；角速度依相鄰有效角度計算。
- OpenCV 的盒框角度定義較特別，本程式以幾何方式將其轉為較直觀的朝向角。
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 使用者設定（新增） =========
USE_CAMERA      = True         # True=用攝影機；False=讀影片
CAMERA_INDEX    = 1            # 你的 B0332 在系統中的 index（0/1/2…）
CAM_WIDTH       = 1280         # 想要的取樣解析度
CAM_HEIGHT      = 720
CAM_FPS_REQ     = 12           # 目標幀率（實際會依設備而定）
RECORD_OUTPUT   = True         # 是否同時把即時畫面錄成 MP4
WINDOW_TITLE    = "B0332 Live + Detection"
# 影片中標線的網格間距（mm）。你現在使用 0.5 cm → 5.0 mm
GRID_SPACING_MM = 5.0
VIDEO_PATH = "IMG_7131.mov"            # ← 換成你的影片
OUT_PREFIX = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

# 色彩遮罩（可視實況微調；這裡是黃＋白兩段合併）
HSV_YELLOW_LO = np.array([15, 60, 120], dtype=np.uint8)
HSV_YELLOW_HI = np.array([35, 255, 255], dtype=np.uint8)
HSV_WHITE_LO  = np.array([0,  0, 200], dtype=np.uint8)
HSV_WHITE_HI  = np.array([180, 60, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 50    # 去除雜訊的最小面積（px^2）
PROCESS_EVERY_N  = 1     # 每 N 幀處理一次（=1 逐幀；=2 每兩幀）
SMOOTH_WIN_ANGLE = 5     # 角度/角速度移動平均視窗

# 比例（mm/px）設定
AUTO_GRID_MM_PER_PX = True   # True：嘗試用固定網格自動估（以 GRID_SPACING_MM 為單位）；False：使用 MANUAL_MM_PER_PX
MANUAL_MM_PER_PX    = None   # 例如 0.172（若 AUTO_GRID_MM_PER_PX=False 就填這裡）

# 固定紅框的實體尺寸（mm）
BOX_W_MM = 9.0
BOX_H_MM = 6.0

# 位置圖視覺化參數
PLOT_RANGE_SCALE    = 1.35   # 視野放大倍率（>1 代表放大；建議 1.2~1.6）
INVERT_Y_AXIS       = True   # 位置圖是否反轉 Y 軸（較符合一般幾何直覺）

# ============================

def estimate_mm_per_px_single_frame(frame):
    """單幀嘗試以固定網格（以 GRID_SPACING_MM 為單位）估 mm/px，失敗回傳 None。"""
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
        px_per_cell = (sp_h + sp_v) / 2.0
    elif sp_h:
        px_per_cell = sp_h
    elif sp_v:
        px_per_cell = sp_v
    else:
        return None

    # 由「每個網格的像素寬度」換算為 mm/px。若你的網格是 0.5 cm（= 5 mm），則 mm/px = 5.0 / px_per_cell。
    return (GRID_SPACING_MM / px_per_cell) if (px_per_cell and px_per_cell > 0) else None  # mm/px


def unwrap_angles_deg(angles_deg):
    """將角度序列（度）做展開，避免跨 ±180 度造成跳變。回傳展開後序列。"""
    if len(angles_deg) == 0:
        return angles_deg
    # 先轉為弧度做 unwrap，再轉回度
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


def find_target_and_angle(frame_bgr):
    """
    色彩遮罩（黃＋白）→ 形態學 → 取最大連通域 → 回傳：
    (cx, cy, x, y, w, h, angle_deg, contour) ；若找不到回傳 None

    angle_deg 以 minAreaRect 的長邊方向為主，轉換到 [-90, 90)。
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

    # 以長邊方向為朝向；rect_angle 在 OpenCV 中：
    #   當 w < h 時，angle 是與水平的角度；當 w >= h，朝向需加 90 度作調整
    if rw >= rh:
        angle_deg = rect_angle + 90.0
    else:
        angle_deg = rect_angle

    # 將角度限制到 [-90, 90)
    while angle_deg >= 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0

    x, y, w, h = cv2.boundingRect(cnt)
    return (cx, cy, x, y, w, h, angle_deg, cnt)


def create_output_directory(video_path):
    """建立輸出資料夾"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"{base_name}_analysis_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def main():
    if USE_CAMERA:
        output_dir = "camera_live_output"
        os.makedirs(output_dir, exist_ok=True)
        print(f"[資訊] 輸出資料夾：{output_dir}")

        # 開相機
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # Windows 建議 CAP_DSHOW
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAM_FPS_REQ)

        # 丟幾幀暖機，避免第一幀曝光怪異
        for _ in range(5):
            cap.read()

        ok, first = cap.read()
        if not ok:
            raise RuntimeError("攝影機開啟失敗或抓不到影像")

        # 若相機不回報 FPS，就用請求值；回報有值就用回報值
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else float(CAM_FPS_REQ)

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
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
        
    # 估 mm/px
    if AUTO_GRID_MM_PER_PX:
        mm_per_px = estimate_mm_per_px_single_frame(first)
    else:
        mm_per_px = MANUAL_MM_PER_PX
    if mm_per_px is None:
        mm_per_px = 0.1   # 偵測不到格線時的暫定值（可換成你的校正值）

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if USE_CAMERA:
        OUT_PREFIX = "camera_live"
        out_path = os.path.join(output_dir, f"{OUT_PREFIX}_rot_tracked.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, max(5.0, fps/PROCESS_EVERY_N), (W, H)) if RECORD_OUTPUT else None
    else:
        out_path = os.path.join(output_dir, f"{OUT_PREFIX}_rot_tracked.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, max(5.0, fps/PROCESS_EVERY_N), (W, H))

    # Kalman init（追蹤中心點）
    kf = make_kalman()
    init_meas = find_target_and_angle(first)
    if init_meas is not None:
        cx0, cy0 = float(init_meas[0]), float(init_meas[1])
    else:
        cx0, cy0 = W/2.0, H/2.0
    kf.statePost = np.array([[cx0], [cy0], [0.0], [0.0]], dtype=np.float32)

    # 逐幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rec = []  # frame, t_s, x_px_filt, y_px_filt, x_mm, y_mm, angle_deg_raw
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
        meas = find_target_and_angle(frame)

        angle_deg = np.nan
        if meas is not None:
            cx, cy, x, y, wbox, hbox, angle_deg, cnt = meas
            est = kf.correct(np.array([[cx],[cy]], dtype=np.float32))
            fx, fy = float(est[0,0]), float(est[1,0])

            # 顯示偵測到的原始輪廓（紅色線條）- 可調整粗細
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 4)  # 改成3讓紅框更粗
            
            # 中心點（綠色）- 可調整大小
            cv2.circle(frame, (int(round(fx)), int(round(fy))), 6, (0,255,0), -1)  # 改成6讓綠點更大
        else:
            fx = fy = np.nan

        # 疊字：位置 & 角度
        fx_mm = fx * mm_per_px if not np.isnan(fx) else np.nan
        fy_mm = fy * mm_per_px if not np.isnan(fy) else np.nan
        cv2.putText(frame, f"Pos(mm): ({fx_mm:.2f}, {fy_mm:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        if not np.isnan(angle_deg):
            cv2.putText(frame, f"Angle: {angle_deg:+.2f} deg", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(frame, f"Angle: NaN", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        t_s = frame_idx / fps
        rec.append((frame_idx, t_s, fx, fy, fx_mm, fy_mm, angle_deg, mm_per_px))
        if writer is not None:
            writer.write(frame)
        # 即時顯示
        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        # q 或 ESC 離開；空白鍵暫停/繼續
        if key == 27 or key == ord('q'):
            break
        elif key == 32:
            # 暫停模式
            while True:
                key2 = cv2.waitKey(10) & 0xFF
                if key2 == 27 or key2 == ord('q') or key2 == 32:
                    if key2 == 32:
                        break
                    else:
                        # 直接離開整個程式
                        cap.release()
                        if writer is not None: writer.release()
                        cv2.destroyAllWindows()
                        return
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # 匯出 CSV（第一版，尚未展開角度/角速度）
    df = pd.DataFrame(rec, columns=[
        "frame","t_s",
        "x_px_filt","y_px_filt",
        "x_mm","y_mm",
        "angle_deg_raw",
        "mm_per_px"
    ])

    # 角度展開與平滑
    angle_series = df["angle_deg_raw"].to_numpy()
    mask_valid = ~np.isnan(angle_series)
    angle_unwrapped = np.full_like(angle_series, np.nan, dtype=float)
    if mask_valid.any():
        angle_unwrapped[mask_valid] = unwrap_angles_deg(angle_series[mask_valid])
        angle_unwrapped = moving_average(angle_unwrapped, SMOOTH_WIN_ANGLE)
    df["angle_deg_unwrapped"] = angle_unwrapped

    # 角速度（deg/s）
    dt = np.gradient(df["t_s"].to_numpy())
    ang_vel = np.full_like(angle_unwrapped, np.nan, dtype=float)
    valid_idx = np.where(mask_valid)[0]
    if len(valid_idx) >= 2:
        # 僅在相鄰有效角度間做微分
        for i0, i1 in zip(valid_idx[:-1], valid_idx[1:]):
            dtheta = angle_unwrapped[i1] - angle_unwrapped[i0]
            dt_seg = df.loc[i1, "t_s"] - df.loc[i0, "t_s"]
            if dt_seg > 0:
                ang_vel[i1] = dtheta / dt_seg
        # 再做平滑
        ang_vel = moving_average(ang_vel, SMOOTH_WIN_ANGLE)
    df["angular_vel_dps"] = ang_vel

    # 輸出 CSV 到資料夾
    csv_path = os.path.join(output_dir, f"{OUT_PREFIX}_pos_angle.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    print(f"[輸出] 追蹤影片：{out_path}")
    print(f"[輸出] 數據 CSV：{csv_path}")
    print(f"[資訊] 使用 mm/px ≈ {df['mm_per_px'].dropna().iloc[0]:.6f}")

    # =============== 視覺化（新版） ===============
    # 建立 1×2 subplot：左=位置，右=角速度
    fig, (ax_pos, ax_w) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # ---- 左：位置軌跡 ----
    ax_pos.plot(df["x_mm"], df["y_mm"], lw=2, label="Trajectory", color='blue')
    valid_pos = df[["x_mm", "y_mm"]].dropna()
    if len(valid_pos) > 0:
        x_start, y_start = valid_pos.iloc[0]["x_mm"], valid_pos.iloc[0]["y_mm"]
        x_end,   y_end   = valid_pos.iloc[-1]["x_mm"], valid_pos.iloc[-1]["y_mm"]
        ax_pos.scatter([x_start], [y_start], s=100, c="green", marker="o", label="Start", zorder=5)
        ax_pos.scatter([x_end],   [y_end],   s=100, c="red",   marker="o", label="End", zorder=5)

    ax_pos.set_aspect("equal", adjustable="box")
    if len(valid_pos) > 0:
        xmin, xmax = valid_pos["x_mm"].min(), valid_pos["x_mm"].max()
        ymin, ymax = valid_pos["y_mm"].min(), valid_pos["y_mm"].max()
        xc = 0.5 * (xmin + xmax)
        yc = 0.5 * (ymin + ymax)
        half_range = 0.5 * max(xmax - xmin, ymax - ymin)
        half_range = max(half_range, 1e-6)
        half_range *= PLOT_RANGE_SCALE
        ax_pos.set_xlim(xc - half_range, xc + half_range)
        ax_pos.set_ylim(yc - half_range, yc + half_range)

    if INVERT_Y_AXIS:
        ax_pos.invert_yaxis()

    ax_pos.set_xlabel("x (mm)")
    ax_pos.set_ylabel("y (mm)")
    ax_pos.set_title("Position (Trajectory)")
    ax_pos.grid(True, linestyle="--", alpha=0.4)
    ax_pos.legend(loc="best")

    # ---- 右：角速度-時間 ----
    ax_w.plot(df["t_s"], df["angular_vel_dps"], lw=2, color='red')
    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("Angular Velocity (deg/s)")
    ax_w.set_title("Angular Velocity vs Time")
    ax_w.grid(True, linestyle="--", alpha=0.4)

    # 保存圖片到資料夾
    plot_path = os.path.join(output_dir, f"{OUT_PREFIX}_pos_angvel_subplot.png")
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    print(f"[輸出] 分析圖片：{plot_path}")
    
    print(f"\n所有輸出檔案已整理至資料夾：{output_dir}")


if __name__ == "__main__":
    main()