# -*- coding: utf-8 -*-
"""
整合式攝影機追蹤與函數產生器控制系統 - 多執行緒版本
========================================
改進：
1. 使用多執行緒架構提升 FPS
2. Thread 1: 專門快速讀取相機幀
3. Thread 2: 影像處理（旋轉、檢測、追蹤、繪圖、計算）
4. Main Thread: 顯示和錄影
5. 解析度優化：640x480 @ 120 FPS (MJPG)

作者：多執行緒優化版本
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
CAM_WIDTH         = 640    # 降低解析度以提高 FPS
CAM_HEIGHT        = 480    # 降低解析度以提高 FPS
CAM_FPS_REQ       = 120
RECORD_OUTPUT     = True
WINDOW_TITLE      = "Integrated Camera & Function Generator Control (Multithreaded)"
DISPLAY_SCALE     = 0.6    # 縮小顯示視窗（60%）

# 多執行緒設定
FRAME_QUEUE_SIZE = 120      # 幀佇列大小（可緩存 1 秒的幀）
RESULT_QUEUE_SIZE = 30      # 結果佇列大小

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
INVERT_Y_AXIS      = False
ORIENT_PLOT_WRAPPED= True
ORIENT_YLIM_DEG    = 60
PLOT_RANGE_SCALE   = 1.35

# 追蹤穩定化參數（增強濾波以減少靜止時的抖動）
KF_PROCESS_NOISE   = 1e-5  # 降低以減少抖動
KF_MEASURE_NOISE   = 1e-2  # 增加以更信任預測
EMA_ALPHA_POS      = 0.25  # 增加以更平滑
EMA_ALPHA_ANGLE    = 0.20  # 增加以更平滑
MAX_MEAS_JUMP_PX   = 80

# 靜止檢測閾值（用於穩定靜止時的抖動）
STATIC_THRESHOLD_PX = 2.0  # 小於此值視為靜止
STATIC_SPEED_THRESHOLD = 1.0  # mm/s，小於此值視為靜止

# ========== 函數產生器設定 ==========
FG_RESOURCE_STRING = 'USB0::0x0957::0x5707::MY59001615::0::INSTR'

# ========== 統計類別 ==========
class Stats:
    """執行緒安全的統計資訊類別"""
    def __init__(self):
        self.lock = threading.Lock()
        self.frames_captured = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.capture_fps = 0.0
        self.process_fps = 0.0
        self.last_capture_time = time.perf_counter()
        self.last_process_time = time.perf_counter()
    
    def update_capture(self):
        with self.lock:
            self.frames_captured += 1
            now = time.perf_counter()
            if self.frames_captured % 30 == 0:
                elapsed = now - self.last_capture_time
                if elapsed > 0:
                    self.capture_fps = 30.0 / elapsed
                self.last_capture_time = now
    
    def update_process(self):
        with self.lock:
            self.frames_processed += 1
            now = time.perf_counter()
            if self.frames_processed % 30 == 0:
                elapsed = now - self.last_process_time
                if elapsed > 0:
                    self.process_fps = 30.0 / elapsed
                self.last_process_time = now
    
    def drop_frame(self):
        with self.lock:
            self.frames_dropped += 1
    
    def get_info(self):
        with self.lock:
            return {
                'captured': self.frames_captured,
                'processed': self.frames_processed,
                'dropped': self.frames_dropped,
                'capture_fps': self.capture_fps,
                'process_fps': self.process_fps
            }

# ========== 函數產生器控制器 ==========
class FunctionGeneratorController:
    """函數產生器控制器（與原版相同）"""
    
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
            print("✓ 函數產生器已連接")
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
        """快速模式切換"""
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

# ========== 輔助函數 ==========
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

def finite_diff(values, t, smooth_win=1):
    v = np.full_like(values, np.nan, dtype=float)
    for i in range(1, len(values)):
        if np.isfinite(values[i]) and np.isfinite(values[i-1]) and (t[i] > t[i-1]):
            v[i] = (values[i] - values[i-1]) / (t[i] - t[i-1])
    if smooth_win and smooth_win > 1:
        v = moving_average(v, smooth_win)
    return v

def make_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * KF_PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KF_MEASURE_NOISE
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf

def find_target_and_angle(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask_y = cv2.inRange(hsv, HSV_YELLOW_LO, HSV_YELLOW_HI)
    mask_w = cv2.inRange(hsv, HSV_WHITE_LO, HSV_WHITE_HI)
    mask = cv2.bitwise_or(mask_y, mask_w)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_CONTOUR_AREA: return None

    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), rect_angle = rect
    
    if rw >= rh: 
        angle_deg = rect_angle
    else:        
        angle_deg = rect_angle - 90.0
    
    while angle_deg >= 90.0: angle_deg -= 180.0
    while angle_deg < -90.0: angle_deg += 180.0

    box = cv2.boxPoints(rect).astype(int)
    return (cx, cy, angle_deg, box)

def ema(prev, cur, alpha):
    if prev is None or not np.isfinite(prev): return cur
    if not np.isfinite(cur): return prev
    return alpha*prev + (1.0-alpha)*cur

def calibrate_scale(frame):
    """
    校準尺度 (mm/pixel)
    方法 1: 檢測 5mm 網格
    方法 2: 使用 device 尺寸 (9mm × 6mm)
    """
    DEVICE_WIDTH_MM = 9.0
    DEVICE_HEIGHT_MM = 6.0
    GRID_MM = 5.0
    
    # 方法 1: 嘗試檢測網格
    print("  嘗試方法 1: 檢測背景網格...")
    grid_scale = detect_grid_scale(frame, GRID_MM)
    
    if grid_scale is not None and 0.05 < grid_scale < 0.5:  # 合理範圍檢查
        print(f"    ✓ 網格檢測成功")
        return grid_scale
    else:
        print(f"    ✗ 網格檢測失敗或不可靠")
    
    # 方法 2: 使用 device 尺寸
    print("  嘗試方法 2: 使用 device 尺寸 (9mm × 6mm)...")
    device_scale = detect_device_scale(frame, DEVICE_WIDTH_MM, DEVICE_HEIGHT_MM)
    
    if device_scale is not None and 0.05 < device_scale < 0.5:
        print(f"    ✓ Device 檢測成功")
        return device_scale
    else:
        print(f"    ✗ Device 檢測失敗")
    
    # 預設值
    print("  ⚠ 使用預設值: 0.1 mm/pixel")
    return 0.1

def detect_grid_scale(frame, grid_spacing_mm):
    """檢測網格間距來估算尺度"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=40, maxLineGap=15)
    if lines is None or len(lines) < 6:
        return None
    
    horiz, vert = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx, dy = x2 - x1, y2 - y1
        ang = np.degrees(np.arctan2(dy, dx))
        if ang < -90: ang += 180
        if ang >  90: ang -= 180
        
        if abs(ang) < 8:
            horiz.extend([y1, y2])
        elif abs(abs(ang) - 90) < 8:
            vert.extend([x1, x2])
    
    def get_spacing(positions):
        if len(positions) < 4:
            return None
        positions = sorted(positions)
        unique = [positions[0]]
        for p in positions[1:]:
            if abs(p - unique[-1]) > 1.5:
                unique.append(p)
        
        if len(unique) < 3:
            return None
        
        diffs = np.diff(unique)
        diffs = diffs[(diffs > 2) & (diffs < 300)]
        if len(diffs) == 0:
            return None
        
        return float(np.min(diffs))  # 最小間距應該是 5mm
    
    sp_h = get_spacing(horiz)
    sp_v = get_spacing(vert)
    
    if sp_h and sp_v:
        px_per_cell = (sp_h + sp_v) / 2.0
    elif sp_h:
        px_per_cell = sp_h
    elif sp_v:
        px_per_cell = sp_v
    else:
        return None
    
    if px_per_cell > 0:
        return grid_spacing_mm / px_per_cell
    return None

def detect_device_scale(frame, device_width_mm, device_height_mm):
    """使用 device 尺寸 (9mm × 6mm) 來估算尺度"""
    # 先旋轉畫面（因為主程式會旋轉）
    frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    # 找到 device
    result = find_target_and_angle(frame_rotated)
    if result is None:
        return None
    
    cx, cy, angle_deg, box = result
    
    # 計算 bounding box 的寬和高（pixels）
    rect = cv2.minAreaRect(box)
    (center_x, center_y), (width_px, height_px), angle = rect
    
    # 確保寬是長邊（9mm），高是短邊（6mm）
    if width_px < height_px:
        width_px, height_px = height_px, width_px
    
    # 使用兩個維度的平均值來估算 mm/pixel
    scale_from_width = device_width_mm / width_px if width_px > 0 else None
    scale_from_height = device_height_mm / height_px if height_px > 0 else None
    
    if scale_from_width and scale_from_height:
        # 使用平均值
        mm_per_px = (scale_from_width + scale_from_height) / 2.0
        print(f"    Device 尺寸: {width_px:.1f} × {height_px:.1f} pixels")
        print(f"    從寬度: {scale_from_width:.4f} mm/px")
        print(f"    從高度: {scale_from_height:.4f} mm/px")
        return mm_per_px
    elif scale_from_width:
        return scale_from_width
    elif scale_from_height:
        return scale_from_height
    
    return None

# ========== Thread 1: 相機讀取執行緒 ==========
def capture_thread(cap, frame_queue, running, stats):
    """
    專門負責快速讀取相機幀
    不做任何處理，只讀取並放入佇列
    """
    print("[Capture Thread] 啟動")
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 10
    
    while running[0]:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[Capture Thread] 連續讀取失敗 {consecutive_failures} 次，停止")
                break
            # 暫時讀取失敗，等待一下再試
            time.sleep(0.001)
            continue
        
        # 讀取成功，重置失敗計數
        consecutive_failures = 0
        timestamp = time.perf_counter()
        stats.update_capture()
        
        # 嘗試放入佇列
        try:
            frame_queue.put_nowait((frame, timestamp))
        except queue.Full:
            # 佇列滿了，丟棄此幀
            stats.drop_frame()
    
    print("[Capture Thread] 結束")

# ========== Thread 2: 影像處理執行緒 ==========
def process_thread(frame_queue, result_queue, running, stats, kf, tracker_state, mm_per_px, fg_controller):
    """
    從佇列取出幀並進行所有影像處理
    包括：旋轉、檢測、追蹤、繪圖、計算速度等
    """
    print("[Process Thread] 啟動")
    
    # EMA 狀態
    ema_x = None
    ema_y = None
    ema_ang = None
    
    # 速度計算
    last_x_mm_abs = None
    last_y_mm_abs = None
    last_t = None
    inst_speed = np.nan
    
    # 原點
    origin_x = [None]  # 使用 list 讓主執行緒也能存取
    origin_y = [None]
    
    # 幀計數
    frame_idx = [0]
    
    while running[0] or not frame_queue.empty():
        try:
            frame, timestamp = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # 旋轉畫面
        frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Kalman 預測
        pred = kf.predict()
        px_pred, py_pred = float(pred[0,0]), float(pred[1,0])
        
        # 目標檢測
        m = find_target_and_angle(frame_rotated)
        use_meas = True
        if m is not None:
            cx, cy, angle_deg, box = m
            dist = np.hypot(cx - px_pred, cy - py_pred)
            if dist > MAX_MEAS_JUMP_PX:
                use_meas = False
        else:
            use_meas = False
        
        if use_meas:
            est = kf.correct(np.array([[cx],[cy]], dtype=np.float32))
            fx_raw, fy_raw = float(est[0,0]), float(est[1,0])
        else:
            fx_raw, fy_raw = px_pred, py_pred
            angle_deg = np.nan
            box = None
        
        # EMA 平滑
        ema_x = ema(ema_x, fx_raw, EMA_ALPHA_POS)
        ema_y = ema(ema_y, fy_raw, EMA_ALPHA_POS)
        ema_ang = ema(ema_ang, angle_deg, EMA_ALPHA_ANGLE) if np.isfinite(angle_deg) else ema_ang
        
        fx, fy = float(ema_x), float(ema_y)
        ang_for_draw = float(ema_ang) if ema_ang is not None else (float(angle_deg) if m is not None else np.nan)
        
        # 繪製追蹤結果
        if box is None:
            sz = 20
            bx = np.array([[fx-sz, fy-sz],[fx+sz, fy-sz],[fx+sz, fy+sz],[fx-sz, fy+sz]], dtype=int)
            box_draw = bx
            green_center_x, green_center_y = fx, fy
        else:
            box_draw = box
            green_center_x, green_center_y = cx, cy
        
        display_frame = frame_rotated.copy()
        cv2.polylines(display_frame, [box_draw], isClosed=True, color=(0,0,255), thickness=3)
        if np.isfinite(green_center_x) and np.isfinite(green_center_y):
            cv2.circle(display_frame, (int(round(green_center_x)), int(round(green_center_y))), 6, (0,255,0), -1)
        
        # 計算座標和速度
        fx_mm_abs = fx * mm_per_px if np.isfinite(fx) else np.nan
        fy_mm_abs = fy * mm_per_px if np.isfinite(fy) else np.nan
        
        # 使用實際時間戳而不是假設的 FPS
        # 相對於第一幀的時間（秒）
        if frame_idx[0] == 0:
            tracker_state['start_time'] = timestamp
        t_s = timestamp - tracker_state.get('start_time', timestamp)
        
        if (last_x_mm_abs is not None and last_y_mm_abs is not None and last_t is not None and
            np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs) and (t_s > last_t)):
            inst_speed = np.hypot(fx_mm_abs-last_x_mm_abs, fy_mm_abs-last_y_mm_abs) / (t_s - last_t)
        last_x_mm_abs, last_y_mm_abs, last_t = fx_mm_abs, fy_mm_abs, t_s
        
        if origin_x[0] is None and np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs):
            origin_x[0], origin_y[0] = fx_mm_abs, fy_mm_abs
        
        # 疊字顯示
        if origin_x[0] is not None and np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs):
            rx = fx_mm_abs - origin_x[0]
            ry = -(fy_mm_abs - origin_y[0])  # 修正Y座標正負號
            pos_text = f"Pos(mm): ({rx:.2f}, {ry:.2f})"
        else:
            pos_text = f"Pos(mm): (NaN, NaN)"
        cv2.putText(display_frame, pos_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        if np.isfinite(ang_for_draw):
            cv2.putText(display_frame, f"Angle: {ang_for_draw:+.2f} deg", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            cv2.putText(display_frame, "Angle: NaN", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        if np.isfinite(inst_speed):
            cv2.putText(display_frame, f"Speed: {inst_speed:.2f} mm/s", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            cv2.putText(display_frame, "Speed: NaN mm/s", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        # 函數產生器狀態
        fg_status = f"FG: Mode {fg_controller.current_mode}" if fg_controller.current_mode else "FG: OFF"
        cv2.putText(display_frame, fg_status, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        
        # 錄影狀態
        if tracker_state['recording']:
            cv2.putText(display_frame, "REC", (CAM_HEIGHT-100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            cv2.circle(display_frame, (CAM_HEIGHT-130, 25), 8, (0,0,255), -1)
        
        # 顯示 FPS 資訊
        info = stats.get_info()
        cv2.putText(display_frame, f"Cap: {info['capture_fps']:.1f} FPS | Proc: {info['process_fps']:.1f} FPS", 
                   (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        
        # 放入結果佇列（包含原始幀和 box）
        result_data = {
            'frame': display_frame,
            'raw_frame': frame_rotated.copy(),  # 原始旋轉後的幀（無標註）
            'timestamp': timestamp,
            'frame_idx': frame_idx[0],
            'fx': fx,
            'fy': fy,
            'fx_mm_abs': fx_mm_abs,
            'fy_mm_abs': fy_mm_abs,
            'angle': ang_for_draw,
            'speed': inst_speed,
            't_s': t_s,
            'box': box  # 記錄 box 用於後續繪製
        }
        
        try:
            result_queue.put_nowait(result_data)
        except queue.Full:
            # 如果結果佇列滿了，丟棄舊的結果
            try:
                result_queue.get_nowait()
                result_queue.put_nowait(result_data)
            except:
                pass
        
        frame_idx[0] += 1
        stats.update_process()
    
    print("[Process Thread] 結束")

def show_help():
    """顯示幫助信息"""
    print("\n" + "="*60)
    print("整合式攝影機追蹤與函數產生器控制系統 (多執行緒)")
    print("="*60)
    print("攝影機控制：")
    print("  [Space]     - 開始/停止錄製")
    print("  [ESC/Q]     - 退出程式")
    print("")
    print("函數產生器控制：")
    print("  [1-4]       - Mode 1-4")
    print("  [0]         - 關閉函數產生器輸出")
    print("")
    print("其他：")
    print("  [H]         - 顯示此幫助")
    print("="*60)

def main():
    """主程式"""
    print("="*60)
    print("整合式攝影機追蹤與函數產生器控制系統 (多執行緒)")
    print("="*60)
    
    # 初始化函數產生器
    fg_controller = FunctionGeneratorController()
    fg_connected = fg_controller.connect()
    
    # 初始化攝影機
    print("\n初始化攝影機...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 設定 MJPG
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS_REQ)
    
    # 暖機
    for _ in range(10):
        cap.read()
    
    # 確認設定
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n相機設定:")
    print(f"  解析度: {actual_width}x{actual_height}")
    print(f"  目標 FPS: {actual_fps:.1f}")
    
    # 估算 mm/px - 使用網格或 device 尺寸校準
    ok, first = cap.read()
    if not ok:
        print("無法讀取影像")
        cap.release()
        return
    
    print("\n正在校準尺度 (mm/pixel)...")
    mm_per_px = calibrate_scale(first)
    print(f"✓ 尺度校準完成: {mm_per_px:.4f} mm/pixel")
    print(f"  驗證：5mm 網格 ≈ {5.0/mm_per_px:.1f} pixels")
    print(f"  驗證：9mm device 寬度 ≈ {9.0/mm_per_px:.1f} pixels")
    print(f"  驗證：6mm device 高度 ≈ {6.0/mm_per_px:.1f} pixels")
    
    # 初始化 Kalman 濾波器
    kf = make_kalman()
    first_rotated = cv2.rotate(first, cv2.ROTATE_90_CLOCKWISE)
    init = find_target_and_angle(first_rotated)
    if init is not None:
        cx0, cy0, ang0, box0 = init
        print(f"✓ 初始位置: ({cx0:.1f}, {cy0:.1f}), 角度: {ang0:.1f}°")
    else:
        cx0, cy0, ang0 = CAM_WIDTH/2.0, CAM_HEIGHT/2.0, 0.0
        print("未檢測到目標，使用預設位置")
    
    kf.statePost = np.array([[cx0],[cy0],[0.0],[0.0]], dtype=np.float32)
    
    # 共享狀態
    running = [True]
    stats = Stats()
    tracker_state = {'recording': False}
    
    # 建立佇列
    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    result_queue = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
    
    # 啟動執行緒
    capture_t = threading.Thread(target=capture_thread, args=(cap, frame_queue, running, stats), daemon=True)
    process_t = threading.Thread(target=process_thread, args=(frame_queue, result_queue, running, stats, kf, tracker_state, mm_per_px, fg_controller), daemon=True)
    
    capture_t.start()
    process_t.start()
    
    show_help()
    print(f"\n系統狀態：")
    print(f"✓ 攝影機：已連接 ({actual_width}x{actual_height} @ {actual_fps:.0f} FPS)")
    print(f"{'✓' if fg_connected else '✗'} 函數產生器：{'已連接' if fg_connected else '未連接'}")
    print(f"\n系統就緒！按 [H] 查看幫助\n")
    
    # 錄影相關
    writer = None
    writer_raw = None  # 原始影片（無標註）
    rec_data = []
    output_dir = None
    
    try:
        while running[0]:
            try:
                data = result_queue.get(timeout=0.1)
                frame = data['frame']
                
                # 顯示
                cv2.imshow(WINDOW_TITLE, frame)
                
                # 錄影
                if tracker_state['recording'] and RECORD_OUTPUT:
                    if writer is None:
                        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_dir = f"{run_tag}_{MODE}_integrated_mt"
                        os.makedirs(output_dir, exist_ok=True)
                        out_path = os.path.join(output_dir, f"camera_{MODE}_tracked.mp4")
                        out_path_raw = os.path.join(output_dir, f"camera_{MODE}_raw.mp4")
                        
                        # 使用實際測量的 FPS（取讀取和處理 FPS 的較小值）
                        out_path = os.path.join(output_dir, f"camera_{MODE}_tracked.avi")
                        out_path_raw = os.path.join(output_dir, f"camera_{MODE}_raw.avi")

                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        writer = cv2.VideoWriter(out_path, fourcc, actual_fps, (CAM_HEIGHT, CAM_WIDTH))
                        writer_raw = cv2.VideoWriter(out_path_raw, fourcc, actual_fps, (CAM_HEIGHT, CAM_WIDTH))
                        tracker_state['actual_fps'] = actual_fps  # 儲存用於 GIF
                        print(f"開始錄影: {out_path} (FPS: {actual_fps:.1f})")
                        print(f"原始影片: {out_path_raw}")
                    
                    writer.write(frame)
                    
                    # 寫入原始影片（無標註）
                    if 'raw_frame' in data:
                        writer_raw.write(data['raw_frame'])
                    
                    # 記錄資料（包含 box）
                    rec_data.append((
                        data['frame_idx'], data['t_s'],
                        data['fx'], data['fy'],
                        data['fx_mm_abs'], data['fy_mm_abs'],
                        data['angle'], mm_per_px,
                        data.get('box')  # 記錄 box
                    ))
                
            except queue.Empty:
                pass
            
            # 按鍵處理
            key = cv2.waitKey(1) & 0xFF
            
            if key in (27, ord('q')):  # ESC 或 Q
                print("\n停止中...")
                running[0] = False
                break
            elif key == 32:  # Space
                if not tracker_state['recording']:
                    tracker_state['recording'] = True
                    print("開始錄製...")
                else:
                    tracker_state['recording'] = False
                    print("停止錄製...")
                    if fg_connected:
                        fg_controller.turn_off()
                    break
            elif key == ord('h'):
                show_help()
            elif key == ord('0'):
                if fg_connected:
                    fg_controller.turn_off()
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                if fg_connected:
                    mode_num = int(chr(key))
                    fg_controller.switch_mode(mode_num)
                else:
                    print("函數產生器未連接")
    
    except KeyboardInterrupt:
        print("\n接收到中斷信號...")
        running[0] = False
    
    finally:
        # 等待執行緒結束
        print("等待執行緒結束...")
        capture_t.join(timeout=2.0)
        process_t.join(timeout=2.0)
        
        # 清理
        cap.release()
        if writer:
            writer.release()
        if writer_raw:
            writer_raw.release()
        cv2.destroyAllWindows()
        
        # 輸出資料
        if len(rec_data) > 0 and output_dir:
            print("\n處理資料並輸出...")
            process_and_export_data(rec_data, output_dir, mm_per_px)
        
        # 斷開函數產生器
        fg_controller.disconnect()
        
        # 最終統計
        info = stats.get_info()
        print("\n最終統計:")
        print(f"  讀取幀數: {info['captured']}")
        print(f"  處理幀數: {info['processed']}")
        print(f"  丟棄幀數: {info['dropped']}")
        print(f"  讀取 FPS: {info['capture_fps']:.1f}")
        print(f"  處理 FPS: {info['process_fps']:.1f}")
        print("程式結束")

def process_and_export_data(rec_data, output_dir, mm_per_px):
    """處理並輸出追蹤資料 - 包含圖表生成"""
    OUT_PREFIX = f"camera_{MODE}"
    
    # 分離 box 資料
    box_rec = []
    data_without_box = []
    for row in rec_data:
        if len(row) >= 9:  # 包含 box
            frame_idx, t_s, fx, fy, fx_mm_abs, fy_mm_abs, angle, mmpx, box = row
            box_rec.append((frame_idx, t_s, fx, fy, box))
            data_without_box.append((frame_idx, t_s, fx, fy, fx_mm_abs, fy_mm_abs, angle, mmpx))
        else:
            data_without_box.append(row)
    
    # 整理資料
    df = pd.DataFrame(data_without_box, columns=[
        "frame","t_s","x_px_filt","y_px_filt","x_mm_abs","y_mm_abs","angle_deg_raw","mm_per_px"
    ])
    
    # 計算相對座標
    x_abs = df["x_mm_abs"].to_numpy()
    y_abs = df["y_mm_abs"].to_numpy()
    valid = np.isfinite(x_abs) & np.isfinite(y_abs)
    if valid.any():
        x0, y0 = x_abs[valid][0], y_abs[valid][0]
        df["x_mm"] = x_abs - x0
        df["y_mm"] = -(y_abs - y0)  # 修正Y座標
    else:
        df["x_mm"] = x_abs
        df["y_mm"] = -y_abs
    
    # 角度處理
    ang_raw = df["angle_deg_raw"].to_numpy()
    mask_ang = np.isfinite(ang_raw)
    ang_unwrap = np.full_like(ang_raw, np.nan, dtype=float)
    if mask_ang.any():
        ang_unwrap[mask_ang] = unwrap_angles_deg(ang_raw[mask_ang])
        ang_unwrap = moving_average(ang_unwrap, 5)
    df["angle_deg_unwrapped"] = ang_unwrap
    
    # 速度計算
    t = df["t_s"].to_numpy()
    vx = finite_diff(df["x_mm"].to_numpy(), t, smooth_win=5)
    vy = finite_diff(df["y_mm"].to_numpy(), t, smooth_win=5)
    speed = np.sqrt(vx**2 + vy**2)
    df["vx_mm_s"] = vx
    df["vy_mm_s"] = vy
    df["speed_mm_s"] = speed
    
    # 角速度計算（for rotation mode）
    ang_vel = np.full_like(ang_unwrap, np.nan, dtype=float)
    idx = np.where(mask_ang)[0]
    if len(idx) >= 2:
        for i0, i1 in zip(idx[:-1], idx[1:]):
            dt = t[i1] - t[i0]
            if dt > 0:
                ang_vel[i1] = (ang_unwrap[i1] - ang_unwrap[i0]) / dt
        ang_vel = moving_average(ang_vel, 5)
    df["angular_vel_dps"] = ang_vel
    
    # 輸出 CSV
    csv_path = os.path.join(output_dir, f"{OUT_PREFIX}_pos_angle_speed.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # 生成圖表
    print("生成圖表...")
    plot_paths = generate_plots(df, output_dir, OUT_PREFIX)
    
    # 生成新的軌跡圖和組圖
    if len(box_rec) >= 8 and valid.any():
        print("生成軌跡輪廓圖和8時間點組圖...")
        try:
            additional_plots = generate_trajectory_and_composite(
                df, box_rec, output_dir, OUT_PREFIX, mm_per_px, x0, y0
            )
            plot_paths.update(additional_plots)
        except Exception as e:
            print(f"  警告：生成額外圖表時發生錯誤: {e}")
    
    print(f"\n✓ 資料已輸出到: {output_dir}")
    print(f"  - CSV: {os.path.basename(csv_path)}")
    print(f"  - Position 圖: {os.path.basename(plot_paths['position'])}")
    if MODE.lower() == "straight":
        print(f"  - Speed/Orientation 圖: {os.path.basename(plot_paths['speed_orientation'])}")
    else:
        print(f"  - Angular Speed 圖: {os.path.basename(plot_paths['angular_speed'])}")
    if 'trajectory_contour' in plot_paths:
        print(f"  - 軌跡輪廓圖: {os.path.basename(plot_paths['trajectory_contour'])}")
    if 'composite' in plot_paths:
        print(f"  - 8時間點組圖: {os.path.basename(plot_paths['composite'])}")

def generate_plots(df, output_dir, OUT_PREFIX):
    """生成追蹤結果圖表 - 參考 filming_straight_and_speed.py"""
    plot_paths = {}
    
    # A) 位置軌跡圖
    figA, ax_pos = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    x_plot = df["x_mm"].to_numpy()
    y_plot = df["y_mm"].to_numpy()
    ax_pos.plot(x_plot, y_plot, lw=2, label="Trajectory")
    
    # 標記起點和終點
    valid_pos = np.vstack([x_plot, y_plot]).T
    valid_pos = valid_pos[~np.isnan(valid_pos).any(axis=1)]
    if len(valid_pos) > 0:
        ax_pos.scatter([valid_pos[0,0]], [valid_pos[0,1]], s=100, c="green", marker="o", label="Start", zorder=5)
        ax_pos.scatter([valid_pos[-1,0]], [valid_pos[-1,1]], s=100, c="red", marker="o", label="End", zorder=5)
        
        # 設定座標軸範圍
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
    
    # B) 速度和角度圖 (straight mode) 或 角速度圖 (rotation mode)
    if MODE.lower() == "straight":
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
        
        # 右圖：角度
        if ORIENT_PLOT_WRAPPED:
            ang_vis = wrap_angles_deg(df["angle_deg_unwrapped"].to_numpy())
        else:
            ang_vis = df["angle_deg_unwrapped"].to_numpy()
        
        if np.isfinite(ang_vis).any():
            first_idx = np.where(np.isfinite(ang_vis))[0][0]
            ang0 = float(ang_vis[first_idx])
            offset_series = wrap_angles_deg(ang_vis - ang0)
            avg_offset_deg = float(np.nanmean(offset_series))
        else:
            avg_offset_deg = float("nan")
        
        label_orient = f"Orientation (deg)\nAvg offset: {avg_offset_deg:.2f}°" if np.isfinite(avg_offset_deg) else "Orientation (deg)"
        ax_a.plot(df["t_s"], ang_vis, lw=2, label=label_orient)
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
        plot_paths['speed_orientation'] = plot_so_path
        
    else:  # rotation mode
        figW, ax_w = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        ax_w.plot(df["t_s"], df["angular_vel_dps"], lw=2, label="Angular speed (deg/s)")
        
        w_all = df["angular_vel_dps"].to_numpy()
        tt = df["t_s"].to_numpy()
        finite = np.isfinite(w_all)
        if np.any(finite):
            idx_rel = int(np.nanargmax(np.abs(w_all[finite])))
            idxs = np.where(finite)[0]
            i_max = idxs[idx_rel]
            ax_w.plot([tt[i_max]], [w_all[i_max]], marker="o", markersize=8, color="red",
                      label=f"Max: {w_all[i_max]:.2f} deg/s @ {tt[i_max]:.2f}s")
        
        ax_w.set_xlabel("Time (s)")
        ax_w.set_ylabel("Angular speed (deg/s)")
        ax_w.set_title("Angular Speed vs Time")
        ax_w.grid(True, linestyle="--", alpha=0.4)
        ax_w.legend(loc="best")
        
        plot_w_path = os.path.join(output_dir, f"{OUT_PREFIX}_angular_speed.png")
        figW.savefig(plot_w_path, dpi=220)
        plt.close(figW)
        plot_paths['angular_speed'] = plot_w_path
    
    return plot_paths

def generate_trajectory_and_composite(df, box_rec, output_dir, OUT_PREFIX, mm_per_px, x0, y0):
    """生成軌跡輪廓圖和8時間點組圖"""
    plot_paths = {}
    
    # 從 box_rec 中提取有效的 boxes
    valid_boxes = [(idx, ts, fx, fy, box) for idx, ts, fx, fy, box in box_rec 
                   if box is not None and np.isfinite(fx) and np.isfinite(fy)]
    
    if len(valid_boxes) < 8:
        print(f"  警告：只有 {len(valid_boxes)} 個有效框，需要至少 8 個")
        return plot_paths
    
    # === 1. 生成軌跡輪廓圖 ===
    figC, ax_contour = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    
    # 繪製整體路徑
    x_plot = df["x_mm"].to_numpy()
    y_plot = df["y_mm"].to_numpy()
    ax_contour.plot(x_plot, y_plot, lw=2, color='blue', label="Trajectory", alpha=0.7)
    
    # 選擇8個等間隔的時間點
    n_segments = 8
    indices = np.linspace(0, len(valid_boxes) - 1, n_segments, dtype=int)
    
    # 使用統一顏色虛線繪製輪廓
    for i, idx in enumerate(indices):
        frame_idx_val, t_s_val, fx_px_val, fy_px_val, box = valid_boxes[idx]
        
        # 將box的四個頂點轉換為相對mm座標
        box_mm = []
        for vertex in box:
            vx_px, vy_px = vertex[0], vertex[1]
            vx_mm = vx_px * mm_per_px - x0
            vy_mm = -(vy_px * mm_per_px - y0)  # 修正Y座標
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
        
        # 設置座標軸範圍
        xmin, xmax = np.nanmin(x_plot), np.nanmax(x_plot)
        ymin, ymax = np.nanmin(y_plot), np.nanmax(y_plot)
        xc = 0.5 * (xmin + xmax)
        yc = 0.5 * (ymin + ymax)
        half_range = 0.5 * max(xmax - xmin, ymax - ymin)
        half_range = max(half_range, 1e-6) * PLOT_RANGE_SCALE
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
    plot_paths['trajectory_contour'] = plot_contour_path
    
    # === 2. 生成 8 個時間點組圖（需要原始影片）===
    # 檢查是否有原始影片
    raw_video_path = os.path.join(output_dir, f"camera_{MODE}_raw.mp4")
    if not os.path.exists(raw_video_path):
        print(f"  警告：找不到原始影片 {raw_video_path}，跳過組圖生成")
        return plot_paths
    
    try:
        # 重新開啟原始影片
        cap_composite = cv2.VideoCapture(raw_video_path)
        
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
        
        # 建立 frame 到 df row 的映射
        row_map = {int(r['frame']): i for i, r in df.iterrows()}
        
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
            
            # 在8個snapshot點上畫圓標記（純藍色）
            for i in range(snap_idx + 1):
                df_idx = snapshot_frames[i]
                if df_idx in row_map:
                    fx_px = df.iloc[row_map[df_idx]]['x_px_filt']
                    fy_px = df.iloc[row_map[df_idx]]['y_px_filt']
                    if np.isfinite(fx_px) and np.isfinite(fy_px):
                        cx_i = int(round(fx_px))
                        cy_i = int(round(fy_px))
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
            # 調整每張圖的大小
            target_height = 300
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
            plot_paths['composite'] = composite_path
        else:
            print(f"  警告：只提取到 {len(extracted_frames)} 個畫面")
    
    except Exception as e:
        print(f"  警告：生成組圖時發生錯誤: {e}")
    
    return plot_paths

if __name__ == "__main__":
    main()