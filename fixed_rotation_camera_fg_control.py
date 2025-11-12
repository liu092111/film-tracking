# -*- coding: utf-8 -*-
"""
整合式攝影機追蹤與函數產生器控制系統 - 旋轉運動專用版
==================================================
修正項目：
1. 使用實際時間戳記而非理論計算
2. 持續監控真實FPS
3. 修正CSV時間記錄
4. 角度展開和角速度分析
5. 自動輸出GIF檔案

功能：
1. 旋轉運動追蹤與角速度分析
2. Keysight 33600A 函數產生器控制
3. 自動輸出MP4影片、GIF動畫、分析圖表和CSV資料

操作說明：
- 攝影機控制：[Space] 開始/停止錄製，[ESC/Q] 退出
- 函數產生器：[1-4] 切換模式，[0] 關閉輸出
- [H] 顯示幫助

作者：旋轉運動專用版
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
WINDOW_TITLE      = "Rotation Motion Tracking & Function Generator Control"
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
def unwrap_angles_deg(a_deg):
    """角度展開 - 旋轉運動專用版"""
    if len(a_deg) == 0: 
        return a_deg
    
    # 轉換為數組並處理 NaN
    angles = np.array(a_deg, dtype=float)
    valid_mask = np.isfinite(angles)
    
    if not np.any(valid_mask):
        return angles
    
    # 只對有效角度進行unwrap
    valid_angles = angles[valid_mask]
    
    # 先將角度統一到 [-90, 90] 範圍
    valid_angles = ((valid_angles + 90) % 180) - 90
    
    # 進行角度展開
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(valid_angles)))
    
    # 將結果放回原數組
    result = angles.copy()
    result[valid_mask] = unwrapped
    
    return result

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

class RotationMotionTracker:
    """旋轉運動追蹤器"""
    
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
        mask = cv2.bitwise_or(mask_
