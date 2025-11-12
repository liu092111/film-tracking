# -*- coding: utf-8 -*-
"""
整合式攝影機追蹤與函數產生器控制系統 - 直線運動專用版
==================================================
修正項目：
1. 使用實際時間戳記而非理論計算
2. 持續監控真實FPS
3. 修正CSV時間記錄
4. 修正orientation角度計算（直線偏差）
5. 自動輸出GIF檔案

功能：
1. 直線運動追蹤與偏差分析
2. Keysight 33600A 函數產生器控制
3. 自動輸出MP4影片、GIF動畫、分析圖表和CSV資料

操作說明：
- 攝影機控制：[Space] 開始/停止錄製，[ESC/Q] 退出
- 函數產生器：[1-4] 切換模式，[0] 關閉輸出
- [H] 顯示幫助

作者：直線運動專用版
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
CAMERA_INDEX      = 1
CAM_WIDTH         = 1280
CAM_HEIGHT        = 720
CAM_FPS_REQ       = 120
RECORD_OUTPUT     = True
EXPORT_GIF        = True   # 是否輸出GIF檔案
WINDOW_TITLE      = "Straight Motion Tracking & Function Generator Control"
DISPLAY_SCALE     = 0.5  # 顯示縮放比例

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
ORIENT_YLIM_DEG    = 60
PLOT_RANGE_SCALE   = 1.35

# 追蹤穩定化參數
KF_PROCESS_NOISE   = 1e-4
KF_MEASURE_NOISE   = 5e-3
EMA_ALPHA_POS      = 0.15
EMA_ALPHA_ANGLE    = 0.12
MAX_MEAS_JUMP_PX   = 80

# 初始檢測增強參數
INITIAL_DETECTION_RETRIES = 50
INITIAL_DETECTION_WARMUP = 20
STATIC_DETECTION_THRESHOLD = 2.0
STATIC_SMOOTHING_FRAMES = 10

# ========== 函數產生器設定 ==========
FG_RESOURCE_STRING = 'USB0::0x0957::0x5707::MY59001615::0::INSTR'

# ========== 資料處理函數 ==========
def moving_average(a, w):
    """移動平均"""
    if w is None or w <= 1: return a
    a = np.asarray(a, dtype=float)
    if len(a) < w: return a
    kernel = np.ones(w) / float(w)
    return np.convolve(a, kernel, mode='same')

def finite_diff(values, t, smooth_win=1):
    """有限差分計算導數"""
    v = np.full_like(values, np.nan, dtype=float)
    for i in range(1, len(values)):
        if np.isfinite(values[i]) and np.isfinite(values[i-1]) and (t[i] > t[i-1]):
            v[i] = (values[i] - values[i-1]) / (t[i] - t[i-1])
    if smooth_win and smooth_win > 1:
        v = moving_average(v, smooth_win)
    return v

class FunctionGeneratorController:
    """函數產生器控制器"""
    
    def __init__(self):
        self.inst = None
        self.sampling_rates = {}
        self.current_mode = None
        self.connected = False
        self.continuous_output_setup = False
        
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
                
                self.inst.write_binary_values(f'SOUR1:DATA:ARB {config["name1"]},', sig1, datatype='f', is_big_endian=False)
                self.inst.write('*WAI')
                
                self.inst.write_binary_values(f'SOUR2:DATA:ARB {config["name2"]},', sig2, datatype='f', is_big_endian=False)
                self.inst.write('*WAI')
                
                print(f"  Mode {mode_num} 已載入 - 頻率: {freq:.1f} Hz")
                
            except Exception as e:
                print(f"  Mode {mode_num} 載入失敗: {e}")
        
        return sampling_rates
    
    def setup_continuous_output(self):
        """設定連續輸出模式"""
        if not self.connected:
            return False
            
        try:
            print("設定連續輸出模式...")
            
            base_mode = self.sampling_rates[1]
            
            self.inst.write('SOUR1:FUNC ARB')
            self.inst.write('SOUR2:FUNC ARB')
            self.inst.write(f'SOUR1:FUNC:ARB {base_mode["name1"]}')
            self.inst.write(f'SOUR2:FUNC:ARB {base_mode["name2"]}')
            self.inst.write(f'SOUR1:FUNC:ARB:SRAT {base_mode["sRate"]:.0f}')
            self.inst.write(f'SOUR2:FUNC:ARB:SRAT {base_mode["sRate"]:.0f}')
            self.inst.write(f'SOUR1:FREQ {base_mode["freq"]}')
            self.inst.write(f'SOUR2:FREQ {base_mode["freq"]}')
            self.inst.write('SOUR1:PHAS 0')
            self.inst.write('SOUR2:PHAS 0')
            self.inst.write('SOUR1:VOLT:OFFS 0')
            self.inst.write('SOUR2:VOLT:OFFS 0')
            self.inst.write('*WAI')
            
            self.inst.write('SOUR1:TRACK OFF')
            self.inst.write('SOUR2:TRACK OFF')
            self.inst.write('SOUR2:TRACK ON')
            self.inst.write('SOUR2:PHAS:SYNC')
            self.inst.write('*WAI')
            
            self.inst.write('SOUR1:VOLT 0')
            self.inst.write('SOUR2:VOLT 0')
            self.inst.write('OUTP1:POL NORM')
            self.inst.write('OUTP2:POL INV')
            self.inst.write('*WAI')
            
            self.inst.write('OUTP1 ON')
            self.inst.write('OUTP2 ON')
            self.inst.write('*WAI')
            
            self.continuous_output_setup = True
            print("✓ 連續輸出模式已設定完成")
            return True
            
        except Exception as e:
            print(f"設定連續輸出模式失敗: {e}")
            return False
    
    def switch_mode(self, mode_num):
        """模式切換"""
        if not self.connected or mode_num not in self.sampling_rates:
            return False
        
        if not self.continuous_output_setup:
            if not self.setup_continuous_output():
                return False
        
        start_time = time.time()
        
        mode_configs = {
            1: {'ch1_volt': 1.2, 'ch2_volt': 1.2, 'ch1_pol': 'NORM', 'ch2_pol': 'INV', 'wave_type': '25k', 'desc': '25k Hz, CH1=NORM, CH2=INV'},
            2: {'ch1_volt': 1.2, 'ch2_volt': 1.2, 'ch1_pol': 'NORM', 'ch2_pol': 'INV', 'wave_type': '47k', 'desc': '47k Hz, CH1=NORM, CH2=INV'},
            3: {'ch1_volt': 1.2, 'ch2_volt': 1.2, 'ch1_pol': 'INV', 'ch2_pol': 'NORM', 'wave_type': '25k', 'desc': '25k Hz, CH1=INV, CH2=NORM'},
            4: {'ch1_volt': 1.2, 'ch2_volt': 1.2, 'ch1_pol': 'INV', 'ch2_pol': 'NORM', 'wave_type': '47k', 'desc': '47k Hz, CH1=INV, CH2=NORM'}
        }
        
        config = mode_configs[mode_num]
        mode_data = self.sampling_rates[mode_num]
        
        try:
            need_waveform_switch = False
            if self.current_mode is None:
                need_waveform_switch = config['wave_type'] != '25k'
            else:
                current_config = mode_configs[self.current_mode]
                need_waveform_switch = current_config['wave_type'] != config['wave_type']
            
            need_polarity_change = False
            if self.current_mode is None:
                need_polarity_change = True
            else:
                current_config = mode_configs[self.current_mode]
                need_polarity_change = (current_config['ch1_pol'] != config['ch1_pol'] or 
                                      current_config['ch2_pol'] != config['ch2_pol'])
            
            if need_waveform_switch or need_polarity_change:
                self.inst.write('OUTP1 OFF; OUTP2 OFF')
                
                if need_waveform_switch:
                    self.inst.write(f'SOUR1:FUNC:ARB {mode_data["name1"]}')
                    self.inst.write(f'SOUR2:FUNC:ARB {mode_data["name2"]}')
                    self.inst.write(f'SOUR1:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
                    self.inst.write(f'SOUR2:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
                    self.inst.write(f'SOUR1:FREQ {mode_data["freq"]}')
                    self.inst.write(f'SOUR2:FREQ {mode_data["freq"]}')
                    self.inst.write('SOUR2:PHAS:SYNC')
                    self.inst.write('*WAI')
                
                if need_polarity_change:
                    self.inst.write(f'OUTP1:POL {config["ch1_pol"]}')
                    self.inst.write(f'OUTP2:POL {config["ch2_pol"]}')
                    self.inst.write('*WAI')
                
                self.inst.write('OUTP1 ON; OUTP2 ON')
                self.inst.write('*WAI')
            
            self.inst.write(f'SOUR1:VOLT {config["ch1_volt"]}; SOUR2:VOLT {config["ch2_volt"]}')
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
            self.inst.write('SOUR1:VOLT 0; SOUR2:VOLT 0')
            self.inst.write('*WAI')
            self.current_mode = None
            print("函數產生器輸出已關閉 (電壓=0V)")
        except Exception as e:
            print(f"關閉函數產生器失敗: {e}")

class StraightMotionTracker:
    """直線運動追蹤器"""
    
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
        self.output_dir = None
        self.out_path = None
        
        # FPS 修正相關變數
        self.start_time = None
        self.actual_fps_buffer = deque(maxlen=60)
        self.last_frame_time = None
        
    def initialize_camera(self):
        """初始化攝影機"""
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS_REQ)
        
        print("⏳ 攝影機暖機中...")
        for _ in range(INITIAL_DETECTION_WARMUP):
            self.cap.read()
        
        ok, first = self.cap.read()
        if not ok:
            raise RuntimeError("攝影機開啟失敗")
        
        if AUTO_GRID_MM_PER_PX:
            self.mm_per_px = self.estimate_mm_per_px_single_frame(first)
        else:
            self.mm_per_px = MANUAL_MM_PER_PX
        if self.mm_per_px is None:
            self.mm_per_px = 0.1
        
        print("🔍 正在尋找追蹤目標...")
        initial_detection = None
        for attempt in range(INITIAL_DETECTION_RETRIES):
            ok, frame = self.cap.read()
            if not ok:
                continue
                
            frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            detection = self.find_target_and_angle(frame_rotated)
            
            if detection is not None:
                initial_detection = detection
                print(f"✓ 目標檢測成功 (嘗試 {attempt + 1}/{INITIAL_DETECTION_RETRIES})")
                break
            
            if (attempt + 1) % 10 == 0:
                print(f"⏳ 持續尋找目標... ({attempt + 1}/{INITIAL_DETECTION_RETRIES})")
        
        self.kf = self.make_kalman()
        if initial_detection is not None:
            cx0, cy0, ang0, box0 = initial_detection
            print(f"✓ 初始位置: ({cx0:.1f}, {cy0:.1f}), 角度: {ang0:.1f}°")
        else:
            cx0, cy0, ang0 = CAM_WIDTH/2.0, CAM_HEIGHT/2.0, 0.0
            print("⚠️  未檢測到目標，使用預設位置。請確保目標在視野中且光線充足。")
        
        self.kf.statePost = np.array([[cx0],[cy0],[0.0],[0.0]], dtype=np.float32)
        
        self.position_history = deque(maxlen=STATIC_SMOOTHING_FRAMES)
        self.is_static = False
        
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
        """處理單一幀 - 直線運動專用版"""
        current_time = time.perf_counter()
        
        # 計算實際 FPS（僅在錄製時）
        if self.state == 1:  # RECORDING
            if self.start_time is None:
                self.start_time = current_time
                self.last_frame_time = current_time
                actual_fps = CAM_FPS_REQ
            else:
                if self.last_frame_time is not None:
                    frame_dt = current_time - self.last_frame_time
                    if frame_dt > 0:
                        instant_fps = 1.0 / frame_dt
                        self.actual_fps_buffer.append(instant_fps)
                        actual_fps = np.mean(self.actual_fps_buffer)
                    else:
                        actual_fps = CAM_FPS_REQ
                else:
                    actual_fps = CAM_FPS_REQ
                self.last_frame_time = current_time
        else:
            actual_fps = CAM_FPS_REQ

        if self.frame_idx % PROCESS_EVERY_N != 0:
            self.frame_idx += 1
            return frame

        # 順時針旋轉90度
        frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Kalman 預測
        pred = self.kf.predict()
        px_pred, py_pred = float(pred[0,0]), float(pred[1,0])

        # 目標檢測
        m = self.find_target_and_angle(frame_rotated)
        use_meas = True
        if m is not None:
            cx, cy, angle_deg, box = m
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

        cv2.polylines(frame_rotated, [box_draw], isClosed=True, color=(0,0,255), thickness=3)
        if np.isfinite(green_center_x) and np.isfinite(green_center_y):
            cv2.circle(frame_rotated, (int(round(green_center_x)), int(round(green_center_y))), 6, (0,255,0), -1)

        # 計算座標和速度 - 使用實際時間
        fx_mm_abs = fx * self.mm_per_px if np.isfinite(fx) else np.nan
        fy_mm_abs = fy * self.mm_per_px if np.isfinite(fy) else np.nan

        # 使用實際經過時間
        if self.state == 1 and self.start_time is not None:  # RECORDING
            t_s = current_time - self.start_time
        else:
            t_s = self.frame_idx / float(actual_fps)

        # 靜止檢測和平滑
        if np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs):
            current_pos = (fx_mm_abs, fy_mm_abs)
            self.position_history.append(current_pos)
            
            if len(self.position_history) >= STATIC_SMOOTHING_FRAMES:
                positions = np.array(self.position_history)
                max_movement = np.max(np.sqrt(np.sum((positions - positions[0])**2, axis=1)))
                self.is_static = max_movement < STATIC_DETECTION_THRESHOLD
                
                if self.is_static:
                    avg_pos = np.mean(positions, axis=0)
                    fx_mm_abs, fy_mm_abs = avg_pos[0], avg_pos[1]
                    fx = fx_mm_abs / self.mm_per_px
                    fy = fy_mm_abs / self.mm_per_px

        # 計算速度
        if (self.last_x_mm_abs is not None and self.last_y_mm_abs is not None and self.last_t is not None and
            np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs) and (t_s > self.last_t)):
            raw_speed = np.hypot(fx_mm_abs-self.last_x_mm_abs, fy_mm_abs-self.last_y_mm_abs) / (t_s - self.last_t)
            if hasattr(self, 'is_static') and self.is_static:
                self.inst_speed = 0.0
            else:
                self.inst_speed = raw_speed
        self.last_x_mm_abs, self.last_y_mm_abs, self.last_t = fx_mm_abs, fy_mm_abs, t_s

        if self.origin_x is None and np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs):
            self.origin_x, self.origin_y = fx_mm_abs, fy_mm_abs

        # 顯示資訊
        display_frame = frame_rotated.copy()
        
        if self.origin_x is not None and np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs):
            rx = fx_mm_abs - self.origin_x
            ry = fy_mm_abs - self.origin_y
            pos_text = f"Pos(mm): ({rx:.2f}, {ry:.2f})"
        else:
            pos_text = f"Pos(mm): (NaN, NaN)"
        cv2.putText(display_frame, pos_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        if np.isfinite(ang_for_draw):
            # 直線偏差顯示
            deviation = ang_for_draw  # 相對於理想直線的偏差
            cv2.putText(display_frame, f"Deviation: {deviation:+.2f} deg", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(display_frame, "Deviation: NaN", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
        if np.isfinite(self.inst_speed):
            cv2.putText(display_frame, f"Speed: {self.inst_speed:.2f} mm/s", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(display_frame, "Speed: NaN mm/s", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText(display_frame, f"FPS: {actual_fps:.1f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        
        fg_status = f"FG: Mode {self.fg_controller.current_mode}" if self.fg_controller.current_mode else "FG: OFF"
        cv2.putText(display_frame, fg_status, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        if self.state == 1:  # RECORDING
            cv2.putText(display_frame, "REC", (CAM_HEIGHT-100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
            cv2.circle(display_frame, (CAM_HEIGHT-130, 25), 8, (0,0,255), -1)
        
        if self.state == 0:  # PREVIEW
            cv2.putText(display_frame, "STRAIGHT MOTION - [SPACE] Start Recording, [1-4] FG Mode",
                        (20, CAM_WIDTH-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # 錄影處理
        if self.state == 1 and RECORD_OUTPUT:  # RECORD
            ts_now = time.perf_counter()
            self.ts_list.append(ts_now)
            if self.writer is None:
                self.frame_buf.append(display_frame.copy())
                if len(self.frame_buf) >= 10:
                    fps_out = actual_fps
                    
                    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.output_dir = f"{run_tag}_straight_integrated"
                    os.makedirs(self.output_dir, exist_ok=True)
                    self.out_path = os.path.join(self.output_dir, f"camera_straight_tracked.mp4")
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    W = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    H = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.writer = cv2.VideoWriter(self.out_path, fourcc, fps_out, (W, H))
                    
                    for f in self.frame_buf:
                        self.writer.write(f)
                    self.frame_buf.clear()
                    
                    print(f"✓ 開始錄影，使用實際FPS: {fps_out:.1f}")
            else:
                self.writer.write(display_frame)

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
        
        return display_frame

def create_gif_from_mp4(mp4_path, gif_path, fps=10, scale=0.5):
    """從MP4檔案創建GIF"""
    try:
        import imageio
        
        reader = imageio.get_reader(mp4_path)
        original_fps = reader.get_meta_data()['fps']
        frame_skip = max(1, int(original_fps / fps))
        
        frames = []
        for i, frame in enumerate(reader):
            if i % frame_skip == 0:
                if scale != 1.0:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    frame = cv2.resize(frame, (new_w, new_h))
                frames.append(frame)
        
        reader.close()
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"✓ GIF已建立：{gif_path}")
        return True
        
    except ImportError:
        print("⚠️  缺少 imageio 模組，無法建立GIF。請執行: pip install imageio")
        return False
    except Exception as e:
        print(f"✗ 建立GIF失敗：{e}")
        return False

def process_and_export_data(tracker):
    """處理並輸出追蹤資料 - 直線運動專用版"""
    if len(tracker.rec) == 0:
        print("沒有追蹤資料可輸出")
        return
    
    if tracker.output_dir and os.path.exists(tracker.output_dir):
        output_dir = tracker.output_dir
    else:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{run_tag}_straight_integrated"
        os.makedirs(output_dir, exist_ok=True)
    OUT_PREFIX = "camera_straight"
    
    # 整理資料為 DataFrame
    df = pd.DataFrame(tracker.rec, columns=[
        "frame","t_s","x_px_filt","y_px_filt","x_mm_abs","y_mm_abs","angle_deg_raw","mm_per_px"
    ])
    
    # 計算相對起點座標
    x_abs = df["x_mm_abs"].to_numpy()
    y_abs = df["y_mm_abs"].to_numpy()
    valid = np.isfinite(x_abs) & np.isfinite(y_abs)
    if valid.any():
        x0, y0 = x_abs[valid][0], y_abs[valid][0]
        df["x_mm"] = x_abs - x0
        df["y_mm"] = y_abs - y0
    else:
        df["x_mm"] = x_abs
        df["y_mm"] = y_abs
    
    # 角度處理 - 直線偏差計算
    ang_raw = df["angle_deg_raw"].to_numpy()
    mask_ang = np.isfinite(ang_raw)
    ang_deviation = np.full_like(ang_raw, np.nan, dtype=float)
    
    if mask_ang.any():
        valid_angles = ang_raw[mask_ang]
        ideal_angle = 0.0  # 理想直線方向
        angle_deviations = []
        
        for angle in valid_angles:
            dev = angle - ideal_angle
            while dev > 90: dev -= 180
            while dev <= -90: dev += 180
            angle_deviations.append(dev)
        
        ang_deviation[mask_ang] = angle_deviations
        ang_deviation = moving_average(ang_deviation, 5)
    
    df["angle_deviation_deg"] = ang_deviation
    
    # 計算線速度
    t = df["t_s"].to_numpy()
    vx = finite_diff(df["x_mm"].to_numpy(), t, smooth_win=5)
    vy = finite_diff(df["y_mm"].to_numpy(), t, smooth_win=5)
    speed = np.sqrt(vx**2 + vy**2)
    df["vx_mm_s"] = vx
    df["vy_mm_s"] = vy
    df["speed_mm_s"] = speed
    
    # 輸出 CSV
    csv_path = os.path.join(output_dir, f"{OUT_PREFIX}_pos_angle_speed.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # 生成圖表
    plot_paths = generate_straight_plots(df, output_dir, OUT_PREFIX)
    
    # 建立GIF檔案
    gif_path = None
    if EXPORT_GIF and tracker.out_path and os.path.exists(tracker.out_path):
        gif_path = os.path.join(output_dir, f"camera_straight_tracked.gif")
        create_gif_from_mp4(tracker.out_path, gif_path)
    
    # 輸出訊息
    print(f"[輸出] 目錄：{output_dir}")
    print(f"[輸出] CSV：{csv_path}")
    print(f"[輸出] Position 圖：{plot_paths['position']}")
    print(f"[輸出] Speed/Deviation 圖：{plot_paths['speed_deviation']}")
    if RECORD_OUTPUT and tracker.out_path and os.path.exists(tracker.out_path):
        print(f"[輸出] 追蹤影片：{tracker.out_path}")
    if gif_path and os.path.exists(gif_path):
        print(f"[輸出] GIF動畫：{gif_path}")
    
    file_count = 4 if not gif_path else 5
    print(f"\n✓ 已成功輸出{file_count}個檔案（直線運動追蹤）")

def generate_straight_plots(df, output_dir, OUT_PREFIX):
    """生成直線運動追蹤結果圖表"""
    
    plot_paths = {}
    
    # A) 位置軌跡圖
    figA, ax_pos = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    x_plot = df["x_mm"].to_numpy()
    y_plot = df["y_mm"].to_numpy()
    ax_pos.plot(x_plot, y_plot, lw=2, label="Trajectory")
    
    valid_pos = np.vstack([x_plot, y_plot]).T
    valid_pos = valid_pos[~np.isnan(valid_pos).any(axis=1)]
    if len(valid_pos) > 0:
        ax_pos.scatter([valid_pos[0,0]], [valid_pos[0,1]], s=100, c="green", marker="o", label="Start", zorder=5)
        ax_pos.scatter([valid_pos[-1,0]], [valid_pos[-1,1]], s=100, c="red", marker="o", label="End", zorder=5)
        
        xmin, xmax = np.nanmin(x_plot), np.nanmax(x_plot)
        ymin, ymax = np.nanmin(y_plot), np.nanmax(y_plot)
        xc = 0.5*(xmin+xmax)
        yc = 0.5*(ymin+ymax)
        half = 0.5*max(xmax-xmin, ymax-ymin)
        half = max(half, 1e-6)*PLOT_RANGE_SCALE
        ax_pos.set_xlim(xc-half, xc+half)
        ax_pos.set_ylim(yc-half, yc+half)
    
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
    plot_paths['position'] = plot_pos_path
    
    # B) 速度和角度偏差圖
    figB, (ax_s, ax_a) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    
    # 左圖：速度
    ax_s.plot(df["t_s"], df["speed_mm_s"], lw=2, label="Speed (mm/s)")
    sp_all = df["speed_mm_s"].to_numpy()
    tt = df["t_s"].to_numpy()
    finite = np.isfinite(sp_all)
    if np.any(finite):
        i_max = np.nanargmax(sp_all)
        ax_s.plot([tt[i_max]], [sp_all[i_max]], marker="o", markersize=8, color="red",
                  label=f"Max: {sp_all[i_max]:.2f} mm/s @ {tt[i_max]:.2f}s")
    ax_s.set_xlabel("Time (s)")
    ax_s.set_ylabel("Speed (mm/s)")
    ax_s.set_title("Speed vs Time")
    ax_s.grid(True, linestyle="--", alpha=0.4)
    ax_s.legend(loc="best")
    
    # 右圖：角度偏差
    ang_vis = df["angle_deviation_deg"].to_numpy()
    
    if np.isfinite(ang_vis).any():
        avg_deviation = float(np.nanmean(np.abs(ang_vis)))
        max_deviation = float(np.nanmax(np.abs(ang_vis)))
    else:
        avg_deviation = float("nan")
        max_deviation = float("nan")
    
    label_orient = f"Orientation Deviation (deg)\nAvg: {avg_deviation:.2f}°, Max: {max_deviation:.2f}°" if np.isfinite(avg_deviation) else "Orientation Deviation (deg)"
    ax_a.plot(df["t_s"], ang_vis, lw=2, label=label_orient)
    ax_a.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Ideal (0°)')
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Angle Deviation (deg)")
    ax_a.set_title("Orientation Deviation vs Time")
    if ORIENT_YLIM_DEG is not None and np.isfinite(ORIENT_YLIM_DEG):
        ylim = float(ORIENT_YLIM_DEG)
        ax_a.set_ylim(-ylim, +ylim)
    ax_a.grid(True, linestyle="--", alpha=0.4)
    ax_a.legend(loc="best")
    
    plot_so_path = os.path.join(output_dir, f"{OUT_PREFIX}_speed_deviation.png")
    figB.savefig(plot_so_path, dpi=220)
    plt.close(figB)
    plot_paths['speed_deviation'] = plot_so_path
    
    return plot_paths

def show_help():
    """顯示幫助信息"""
    print("\n" + "="*60)
    print("整合式攝影機追蹤與函數產生器控制系統 - 直線運動專用版")
    print("="*60)
    print("修正項目：")
    print("  - 使用實際時間戳記而非理論計算")
    print("  - 持續監控真實FPS")
    print("  - 修正orientation角度計算（顯示偏差）")
    print("  - 自動輸出GIF檔案")
    print("")
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
    print("整合式攝影機追蹤與函數產生器控制系統 - 直線運動專用版")
    print("="*60)
    
    # 初始化函數產生器
    fg_controller = FunctionGeneratorController()
    fg_connected = fg_controller.connect()
    
    # 初始化直線運動追蹤器
    tracker = StraightMotionTracker(fg_controller)
    
    try:
        tracker.initialize_camera()
        
        show_help()
        
        print(f"\n系統狀態：")
        print(f"✓ 攝影機：已連接")
        print(f"{'✓' if fg_connected else '✗'} 函數產生器：{'已連接' if fg_connected else '未連接'}")
        print(f"✓ 直線運動追蹤：已啟用")
        print(f"✓ FPS修正：已啟用")
        print(f"\n系統就緒！")
        
        # 主迴圈
        while True:
            ok, frame = tracker.cap.read()
            if not ok:
                break
            
            frame = tracker.process_frame(frame)
            
            if DISPLAY_SCALE != 1.0:
                display_height, display_width = int(frame.shape[0] * DISPLAY_SCALE), int(frame.shape[1] * DISPLAY_SCALE)
                frame_display = cv2.resize(frame, (display_width, display_height))
            else:
                frame_display = frame
            
            cv2.imshow(WINDOW_TITLE, frame_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key in (27, ord('q')):  # ESC 或 Q 退出
                break
            elif key == 32:  # Space 開始/停止錄製
                if tracker.state == 0:
                    tracker.state = 1
                    print("開始錄製...")
                else:
                    tracker.state = 0
                    print("錄製結束")
                    if fg_connected:
                        fg_controller.turn_off()
                        print("函數產生器已立即關閉")
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
        tracker.cleanup_camera()
        fg_controller.disconnect()
        
        if len(tracker.rec) > 0:
            print("\n正在處理和輸出資料...")
            process_and_export_data(tracker)
        else:
            print("未開始錄影，沒有輸出。")
        
        print("程式結束")

if __name__ == "__main__":
    main()
