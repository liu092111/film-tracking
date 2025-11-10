# -*- coding: utf-8 -*-
"""
整合式攝影機追蹤與函數產生器控制系統
========================================
功能：
1. 即時攝影機追蹤（基於改良版 real_time_camera.py）
2. Keysight 33600A 函數產生器控制（基於 dual_modal_4ms.py）
3. 同時運行，支援實時切換函數產生器模式

操作說明：
- 攝影機控制：[Space] 開始/停止錄製，[ESC/Q] 退出
- 函數產生器：[1-4] 切換模式，[0] 關閉輸出
- [H] 顯示幫助，[M] 切換操作模式

作者：整合版本
"""

import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import queue
from collections import deque

# Function Generator 相關
try:
    import pyvisa as visa
    HAVE_VISA = True
except ImportError:
    HAVE_VISA = False
    print("警告：未安裝 pyvisa，函數產生器功能將被禁用")

# Scipy 相關
try:
    from scipy.signal import savgol_filter
    HAVE_SG = True
except Exception:
    HAVE_SG = False

# ========== 攝影機設定 ==========
MODE              = "straight"
CAMERA_INDEX      = 1
CAM_WIDTH         = 1280
CAM_HEIGHT        = 720
CAM_FPS_REQ       = 120
RECORD_OUTPUT     = True
WINDOW_TITLE      = "Integrated Camera & Function Generator Control"

# 尺度設定
GRID_SPACING_MM   = 5.0
AUTO_GRID_MM_PER_PX = True
MANUAL_MM_PER_PX     = None

# 顏色遮罩
HSV_YELLOW_LO = np.array([15, 60, 120], dtype=np.uint8)
HSV_YELLOW_HI = np.array([35, 255, 255], dtype=np.uint8)
HSV_WHITE_LO  = np.array([0,  0, 200], dtype=np.uint8)
HSV_WHITE_HI  = np.array([180, 60, 255], dtype=np.uint8)

MIN_CONTOUR_AREA   = 50
PROCESS_EVERY_N    = 1
INVERT_Y_AXIS      = True
ORIENT_PLOT_WRAPPED= True
ORIENT_YLIM_DEG    = 60
PLOT_RANGE_SCALE   = 1.35

# 追蹤穩定化參數
KF_PROCESS_NOISE   = 1e-3
KF_MEASURE_NOISE   = 1e-2
EMA_ALPHA_POS      = 0.25
EMA_ALPHA_ANGLE    = 0.20
MAX_MEAS_JUMP_PX   = 80
WARMUP_FRAMES_FOR_FPS = 30

# ========== 函數產生器設定 ==========
FG_RESOURCE_STRING = 'USB0::0x0957::0x5707::MY59001615::0::INSTR'

class FunctionGeneratorController:
    """函數產生器控制器"""
    
    def __init__(self):
        self.inst = None
        self.sampling_rates = {}
        self.current_mode = None
        self.connected = False
        
    def connect(self):
        """連接函數產生器"""
        if not HAVE_VISA:
            print("PyVISA 未安裝，函數產生器功能已禁用")
            return False
            
        try:
            rm = visa.ResourceManager()
            self.inst = rm.open_resource(FG_RESOURCE_STRING)
            try:
                self.inst.control_ren(6)
            except:
                pass
            
            # 重置和初始化
            self.reset_instrument()
            self.sampling_rates = self.preload_waveforms()
            self.connected = True
            print("✓ 函數產生器已連接並初始化")
            return True
            
        except Exception as e:
            print(f"✗ 函數產生器連接失敗：{e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """斷開函數產生器"""
        if self.inst and self.connected:
            try:
                self.reset_instrument()
                self.inst.close()
                print("✓ 函數產生器已斷開")
            except:
                pass
        self.connected = False
    
    def reset_instrument(self):
        """重置儀器"""
        if not self.inst:
            return
            
        self.inst.write('OUTP1 OFF')
        self.inst.write('OUTP2 OFF')
        self.inst.write('SOUR1:TRACK OFF')
        self.inst.write('SOUR2:TRACK OFF')
        self.inst.write('*WAI')
        
        self.inst.write('SOUR1:FUNC SIN')
        self.inst.write('SOUR2:FUNC SIN')
        self.inst.write('SOUR1:FREQ 1000')
        self.inst.write('SOUR2:FREQ 1000')
        self.inst.write('SOUR1:VOLT 0.1')
        self.inst.write('SOUR2:VOLT 0.1')
        self.inst.write('*WAI')
        
        # 清理記憶體
        self.inst.write('SOUR1:DATA:VOL:CLE')
        self.inst.write('SOUR2:DATA:VOL:CLE')
        self.inst.write('*WAI')
    
    def load_waveform_csv(self, filename):
        """載入 CSV 波形檔案"""
        times = []
        values = []
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if ',' in line:
                        parts = line.split(',')
                    else:
                        parts = line.split()
                    
                    if len(parts) >= 2:
                        try:
                            t = float(parts[0])
                            v = float(parts[1])
                            times.append(t)
                            values.append(v)
                        except ValueError:
                            continue
        except FileNotFoundError:
            print(f"警告：找不到波形檔案 {filename}")
            return np.array([]), np.array([])
        
        return np.array(times), np.array(values)
    
    def align_waveforms(self, file1, file2):
        """對齊波形"""
        times1, values1 = self.load_waveform_csv(file1)
        times2, values2 = self.load_waveform_csv(file2)
        
        if len(times1) == 0 or len(times2) == 0:
            # 如果找不到檔案，返回預設正弦波
            print("使用預設正弦波")
            t = np.linspace(0, 1, 2000)
            sin1 = np.sin(2 * np.pi * t)
            sin2 = np.cos(2 * np.pi * t)
            return sin1.astype('f4'), sin2.astype('f4'), 2000.0, 2000
        
        t_start = max(times1[0], times2[0])
        t_end = min(times1[-1], times2[-1])
        
        dt1 = np.mean(np.diff(times1))
        dt2 = np.mean(np.diff(times2))
        dt_unified = min(dt1, dt2)
        
        unified_times = np.arange(t_start, t_end + dt_unified, dt_unified)
        
        aligned_values1 = np.interp(unified_times, times1, values1)
        aligned_values2 = np.interp(unified_times, times2, values2)
        
        sRate = 1 / dt_unified
        
        return (aligned_values1.astype('f4'), aligned_values2.astype('f4'), 
                sRate, len(unified_times))
    
    def preload_waveforms(self):
        """預載入所有波形"""
        if not self.inst:
            return {}
            
        print("載入函數產生器波形...")
        
        # 模式配置
        modes_config = {
            1: {
                'file1': 'modal/ONEPERIOD_A_25k_50k_84p88deg_2000pts.csv',
                'file2': 'modal/ONEPERIOD_B_25k_50k_264p88deg_2000pts.csv',
                'name1': 'WF_25K_84',
                'name2': 'WF_25K_264'
            },
            2: {
                'file1': 'modal/ONEPERIOD_C_47k_94k_57p32deg_2000pts.csv',
                'file2': 'modal/ONEPERIOD_D_47k_94k_237p32deg_2000pts.csv',
                'name1': 'WF_47K_57',
                'name2': 'WF_47K_237'
            },
            3: {
                'file1': 'modal/ONEPERIOD_A_25k_50k_84p88deg_2000pts.csv',
                'file2': 'modal/ONEPERIOD_B_25k_50k_264p88deg_2000pts.csv',
                'name1': 'WF_25K_84',
                'name2': 'WF_25K_264'
            },
            4: {
                'file1': 'modal/ONEPERIOD_C_47k_94k_57p32deg_2000pts.csv',
                'file2': 'modal/ONEPERIOD_D_47k_94k_237p32deg_2000pts.csv',
                'name1': 'WF_47K_57',
                'name2': 'WF_47K_237'
            }
        }
        
        self.inst.write('OUTP1 OFF')
        self.inst.write('OUTP2 OFF')
        self.inst.write('*WAI')
        
        sampling_rates = {}
        
        for mode_num, config in modes_config.items():
            try:
                sig1, sig2, sRate, points = self.align_waveforms(
                    config['file1'], config['file2']
                )
                
                freq = sRate / points
                
                sampling_rates[mode_num] = {
                    'sRate': sRate,
                    'points': points,
                    'freq': freq,
                    'name1': config['name1'],
                    'name2': config['name2']
                }
                
                # 上傳波形
                self.inst.write_binary_values(f'SOUR1:DATA:ARB {config["name1"]},', sig1, datatype='f', is_big_endian=False)
                self.inst.write('*WAI')
                
                self.inst.write_binary_values(f'SOUR2:DATA:ARB {config["name2"]},', sig2, datatype='f', is_big_endian=False)
                self.inst.write('*WAI')
                
                print(f"  Mode {mode_num} 已載入 - 頻率: {freq:.1f} Hz")
                
            except Exception as e:
                print(f"  Mode {mode_num} 載入失敗: {e}")
        
        return sampling_rates
    
    def switch_mode(self, mode_num):
        """切換函數產生器模式"""
        if not self.connected or mode_num not in self.sampling_rates:
            return False
        
        start_time = time.time()
        
        # 模式配置
        mode_configs = {
            1: {'ch1_pol': 'NORM', 'ch2_pol': 'INV', 'desc': '25k Hz, CH1=NORM, CH2=INV'},
            2: {'ch1_pol': 'NORM', 'ch2_pol': 'INV', 'desc': '47k Hz, CH1=NORM, CH2=INV'},
            3: {'ch1_pol': 'INV', 'ch2_pol': 'NORM', 'desc': '25k Hz, CH1=INV, CH2=NORM'},
            4: {'ch1_pol': 'INV', 'ch2_pol': 'NORM', 'desc': '47k Hz, CH1=INV, CH2=NORM'}
        }
        
        config = mode_configs[mode_num]
        mode_data = self.sampling_rates[mode_num]
        
        try:
            # 關閉輸出
            self.inst.write('OUTP1 OFF; OUTP2 OFF')
            
            # 設定波形
            self.inst.write('SOUR1:FUNC ARB')
            self.inst.write('SOUR2:FUNC ARB')
            self.inst.write(f'SOUR1:FUNC:ARB {mode_data["name1"]}')
            self.inst.write(f'SOUR2:FUNC:ARB {mode_data["name2"]}')
            self.inst.write(f'SOUR1:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
            self.inst.write(f'SOUR2:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
            self.inst.write(f'SOUR1:FREQ {mode_data["freq"]}')
            self.inst.write(f'SOUR2:FREQ {mode_data["freq"]}')
            self.inst.write('SOUR1:VOLT 1.2')
            self.inst.write('SOUR2:VOLT 1.2')
            self.inst.write('SOUR1:VOLT:OFFS 0')
            self.inst.write('SOUR2:VOLT:OFFS 0')
            self.inst.write('SOUR1:PHAS 0')
            self.inst.write('SOUR2:PHAS 0')
            self.inst.write('*WAI')
            
            # 設定同步
            self.inst.write('SOUR1:TRACK OFF')
            self.inst.write('SOUR2:TRACK OFF')
            self.inst.write('SOUR2:TRACK ON')
            self.inst.write('SOUR2:PHAS:SYNC')
            self.inst.write('*WAI')
            
            # 設定極性
            self.inst.write(f'OUTP1:POL {config["ch1_pol"]}')
            self.inst.write(f'OUTP2:POL {config["ch2_pol"]}')
            self.inst.write('*WAI')
            
            # 開啟輸出
            self.inst.write('OUTP1 ON; OUTP2 ON')
            self.inst.write('*WAI')
            
            switch_time = (time.time() - start_time) * 1000
            self.current_mode = mode_num
            
            print(f"FG Mode {mode_num} | {switch_time:.1f}ms | {config['desc']}")
            return True
            
        except Exception as e:
            print(f"函數產生器模式切換失敗: {e}")
            return False
    
    def turn_off(self):
        """關閉函數產生器輸出"""
        if not self.connected:
            return
        try:
            self.inst.write('OUTP1 OFF; OUTP2 OFF')
            self.inst.write('*WAI')
            self.current_mode = None
            print("函數產生器輸出已關閉")
        except Exception as e:
            print(f"關閉函數產生器失敗: {e}")

class CameraTracker:
    """攝影機追蹤器"""
    
    def __init__(self, fg_controller):
        self.fg_controller = fg_controller
        self.cap = None
        self.writer = None
        self.frame_buf = []
        self.ts_list = []
        self.kf = None
        self.ema_x = None
        self.ema_y = None
        self.ema_ang = None
        self.rec = []
        self.frame_idx = 0
        self.origin_x = None
        self.origin_y = None
        self.last_x_mm_abs = None
        self.last_y_mm_abs = None
        self.last_t = None
        self.inst_speed = np.nan
        self.mm_per_px = 0.1
        self.state = 0  # 0=PREVIEW, 1=RECORD
        
    def initialize_camera(self):
        """初始化攝影機"""
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS_REQ)
        
        # 暖機
        for _ in range(5):
            self.cap.read()
        
        ok, first = self.cap.read()
        if not ok:
            raise RuntimeError("攝影機開啟失敗")
        
        # 估算 mm/px
        if AUTO_GRID_MM_PER_PX:
            self.mm_per_px = self.estimate_mm_per_px_single_frame(first)
        else:
            self.mm_per_px = MANUAL_MM_PER_PX
        if self.mm_per_px is None:
            self.mm_per_px = 0.1
        
        # 初始化 Kalman 濾波器
        self.kf = self.make_kalman()
        init = self.find_target_and_angle(first)
        if init is not None:
            cx0, cy0, ang0, box0 = init
        else:
            cx0, cy0, ang0 = CAM_WIDTH/2.0, CAM_HEIGHT/2.0, 0.0
        self.kf.statePost = np.array([[cx0],[cy0],[0.0],[0.0]], dtype=np.float32)
        
        print("✓ 攝影機已初始化")
        return True
    
    def cleanup_camera(self):
        """清理攝影機資源"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
    
    def estimate_mm_per_px_single_frame(self, frame):
        """估算 mm/px 比例"""
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

    def make_kalman(self):
        """創建 Kalman 濾波器"""
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

    def find_target_and_angle(self, frame_bgr):
        """尋找目標並計算角度"""
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

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), rect_angle = rect
        if rw >= rh: angle_deg = rect_angle + 90.0
        else:        angle_deg = rect_angle
        while angle_deg >= 90.0: angle_deg -= 180.0
        while angle_deg <  -90.0: angle_deg += 180.0

        box = cv2.boxPoints(rect).astype(int)
        return (cx, cy, angle_deg, box)

    def ema(self, prev, cur, alpha):
        """指數移動平均"""
        if prev is None or not np.isfinite(prev): return cur
        if not np.isfinite(cur): return prev
        return alpha*prev + (1.0-alpha)*cur

    def process_frame(self, frame):
        """處理單一幀"""
        if self.frame_idx % PROCESS_EVERY_N != 0:
            self.frame_idx += 1
            return frame

        # Kalman 預測
        pred = self.kf.predict()
        px_pred, py_pred = float(pred[0,0]), float(pred[1,0])

        # 目標檢測
        m = self.find_target_and_angle(frame)
        use_meas = True
        if m is not None:
            cx, cy, angle_deg, box = m
            # 離群保護
            dist = np.hypot(cx - px_pred, cy - py_pred)
            if dist > MAX_MEAS_JUMP_PX:
                use_meas = False
        else:
            use_meas = False

        if use_meas:
            est = self.kf.correct(np.array([[cx],[cy]], dtype=np.float32))
            fx_raw, fy_raw = float(est[0,0]), float(est[1,0])
        else:
            fx_raw, fy_raw = px_pred, py_pred
            angle_deg = np.nan
            box = None

        # EMA 平滑
        self.ema_x = self.ema(self.ema_x, fx_raw, EMA_ALPHA_POS)
        self.ema_y = self.ema(self.ema_y, fy_raw, EMA_ALPHA_POS)
        self.ema_ang = self.ema(self.ema_ang, angle_deg, EMA_ALPHA_ANGLE) if np.isfinite(angle_deg) else self.ema_ang

        fx, fy = float(self.ema_x), float(self.ema_y)
        ang_for_draw = float(self.ema_ang) if self.ema_ang is not None else (float(angle_deg) if m is not None else np.nan)

        # 繪製結果
        if box is None:
            sz = 20
            bx = np.array([[fx-sz, fy-sz],[fx+sz, fy-sz],[fx+sz, fy+sz],[fx-sz, fy+sz]], dtype=int)
            box_draw = bx
            green_center_x, green_center_y = fx, fy
        else:
            box_draw = box
            green_center_x, green_center_y = cx, cy

        # 繪製追蹤結果
        cv2.polylines(frame, [box_draw], isClosed=True, color=(0,0,255), thickness=3)
        if np.isfinite(green_center_x) and np.isfinite(green_center_y):
            cv2.circle(frame, (int(round(green_center_x)), int(round(green_center_y))), 6, (0,255,0), -1)

        # 計算座標和速度
        fx_mm_abs = fx * self.mm_per_px if np.isfinite(fx) else np.nan
        fy_mm_abs = fy * self.mm_per_px if np.isfinite(fy) else np.nan

        fps_device = self.cap.get(cv2.CAP_PROP_FPS) or float(CAM_FPS_REQ)
        t_s = self.frame_idx / (fps_device if fps_device > 0 else 30.0)

        if (self.last_x_mm_abs is not None and self.last_y_mm_abs is not None and self.last_t is not None and
            np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs) and (t_s > self.last_t)):
            self.inst_speed = np.hypot(fx_mm_abs-self.last_x_mm_abs, fy_mm_abs-self.last_y_mm_abs) / (t_s - self.last_t)
        self.last_x_mm_abs, self.last_y_mm_abs, self.last_t = fx_mm_abs, fy_mm_abs, t_s

        if self.origin_x is None and np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs):
            self.origin_x, self.origin_y = fx_mm_abs, fy_mm_abs

        # 疊字顯示
        if self.origin_x is not None and np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs):
            rx = fx_mm_abs - self.origin_x
            ry = fy_mm_abs - self.origin_y
            pos_text = f"Pos(mm): ({rx:.2f}, {ry:.2f})"
        else:
            pos_text = f"Pos(mm): (NaN, NaN)"
        cv2.putText(frame, pos_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        if np.isfinite(ang_for_draw):
            cv2.putText(frame, f"Angle: {ang_for_draw:+.2f} deg", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(frame, "Angle: NaN", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
        if np.isfinite(self.inst_speed):
            cv2.putText(frame, f"Speed: {self.inst_speed:.2f} mm/s", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(frame, "Speed: NaN mm/s", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # 函數產生器狀態顯示
        fg_status = f"FG: Mode {self.fg_controller.current_mode}" if self.fg_controller.current_mode else "FG: OFF"
        cv2.putText(frame, fg_status, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # 狀態提示
        if self.state == 0:  # PREVIEW
            cv2.putText(frame, "INTEGRATED PREVIEW - [SPACE] Start Recording, [1-4] FG Mode, [0] FG Off",
                        (20, CAM_HEIGHT-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # 錄影處理
        if self.state == 1 and RECORD_OUTPUT:  # RECORD
            ts_now = time.perf_counter()
            self.ts_list.append(ts_now)
            if self.writer is None:
                self.frame_buf.append(frame.copy())
                if len(self.frame_buf) >= WARMUP_FRAMES_FOR_FPS and len(self.ts_list) >= WARMUP_FRAMES_FOR_FPS:
                    dt = np.diff(np.array(self.ts_list[-WARMUP_FRAMES_FOR_FPS:], dtype=float))
                    dt = dt[dt > 0]
                    fps_out = float(1.0 / np.median(dt)) if dt.size > 0 else (fps_device or 30.0)
                    
                    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = f"{run_tag}_{MODE}_integrated"
                    os.makedirs(output_dir, exist_ok=True)
                    out_path = os.path.join(output_dir, f"camera_{MODE}_tracked.mp4")
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.writer = cv2.VideoWriter(out_path, fourcc, fps_out, (W, H))
                    
                    for f in self.frame_buf:
                        self.writer.write(f)
                    self.frame_buf.clear()
            else:
                self.writer.write(frame)

        # 記錄資料
        if self.state == 1:  # RECORD
            self.rec.append((
                self.frame_idx, t_s,
                fx, fy,
                fx_mm_abs, fy_mm_abs,
                float(ang_for_draw) if np.isfinite(ang_for_draw) else np.nan,
                self.mm_per_px
            ))

        self.frame_idx += 1
        return frame

def show_help():
    """顯示幫助信息"""
    print("\n" + "="*60)
    print("整合式攝影機追蹤與函數產生器控制系統")
    print("="*60)
    print("攝影機控制：")
    print("  [Space]     - 開始/停止錄製")
    print("  [ESC/Q]     - 退出程式")
    print("")
    print("函數產生器控制：")
    print("  [1]         - Mode 1 (25k Hz, CH1=NORM, CH2=INV)")
    print("  [2]         - Mode 2 (47k Hz, CH1=NORM, CH2=INV)")
    print("  [3]         - Mode 3 (25k Hz, CH1=INV, CH2=NORM)")
    print("  [4]         - Mode 4 (47k Hz, CH1=INV, CH2=NORM)")
    print("  [0]         - 關閉函數產生器輸出")
    print("")
    print("其他：")
    print("  [H]         - 顯示此幫助")
    print("="*60)

def main():
    """主程式"""
    print("="*60)
    print("整合式攝影機追蹤與函數產生器控制系統")
    print("="*60)
    
    # 初始化函數產生器
    fg_controller = FunctionGeneratorController()
    fg_connected = fg_controller.connect()
    
    # 初始化攝影機追蹤器
    camera_tracker = CameraTracker(fg_controller)
    
    try:
        camera_tracker.initialize_camera()
        
        # 顯示幫助
        show_help()
        
        print(f"\n系統狀態：")
        print(f"✓ 攝影機：已連接")
        print(f"{'✓' if fg_connected else '✗'} 函數產生器：{'已連接' if fg_connected else '未連接'}")
        print(f"\n系統就緒！")
        
        # 主迴圈
        while True:
            ok, frame = camera_tracker.cap.read()
            if not ok:
                break
            
            # 處理幀
            frame = camera_tracker.process_frame(frame)
            
            # 顯示
            cv2.imshow(WINDOW_TITLE, frame)
            
            # 按鍵處理
            key = cv2.waitKey(1) & 0xFF
            
            if key in (27, ord('q')):  # ESC 或 Q 退出
                break
            elif key == 32:  # Space 開始/停止錄製
                if camera_tracker.state == 0:
                    camera_tracker.state = 1
                    print("開始錄製...")
                else:
                    camera_tracker.state = 0
                    print("錄製結束")
                    break
            elif key == ord('h'):  # 顯示幫助
                show_help()
            elif key == ord('0'):  # 關閉函數產生器
                if fg_connected:
                    fg_controller.turn_off()
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:  # 函數產生器模式切換
                if fg_connected:
                    mode_num = int(chr(key))
                    fg_controller.switch_mode(mode_num)
                else:
                    print("函數產生器未連接")
    
    except Exception as e:
        print(f"系統錯誤：{e}")
    
    finally:
        # 清理資源
        print("\n清理資源...")
        camera_tracker.cleanup_camera()
        fg_controller.disconnect()
        print("程式結束")

if __name__ == "__main__":
    main()
