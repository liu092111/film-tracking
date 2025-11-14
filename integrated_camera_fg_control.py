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

修正重點：
- 實際FPS測量（修正假設120FPS問題）
- 使用實際時間戳記計算速度
- 準確的網格參考（5mm×5mm）
- 設備尺寸驗證（9mm×6mm）

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

# GIF 輸出相關
try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False
    print("警告：未安裝 PIL/Pillow，GIF 輸出功能將被禁用")

# ========== 攝影機設定 ==========
MODE              = "straight"
CAMERA_INDEX      = 1
CAM_WIDTH         = 1280
CAM_HEIGHT        = 720
CAM_FPS_REQ       = 120
RECORD_OUTPUT     = True
WINDOW_TITLE      = "Integrated Camera & Function Generator Control"
DISPLAY_SCALE     = 0.5  # 顯示縮放比例 - 縮小顯示畫面

# 尺度設定 - 修正網格參考
GRID_SPACING_MM   = 5.0  # 5mm × 5mm 網格
DEVICE_WIDTH_MM   = 9.0  # 設備寬度 9mm
DEVICE_HEIGHT_MM  = 6.0  # 設備高度 6mm
AUTO_GRID_MM_PER_PX = True
MANUAL_MM_PER_PX     = None

# 顏色遮罩
HSV_YELLOW_LO = np.array([15, 60, 120], dtype=np.uint8)
HSV_YELLOW_HI = np.array([35, 255, 255], dtype=np.uint8)
HSV_WHITE_LO  = np.array([0,  0, 200], dtype=np.uint8)
HSV_WHITE_HI  = np.array([180, 60, 255], dtype=np.uint8)

MIN_CONTOUR_AREA   = 50
PROCESS_EVERY_N    = 1
INVERT_Y_AXIS      = False
ORIENT_PLOT_WRAPPED= True
ORIENT_YLIM_DEG    = 60
PLOT_RANGE_SCALE   = 1.35

# 追蹤穩定化參數
KF_PROCESS_NOISE   = 1e-4  # 降低過程噪聲，增加穩定性
KF_MEASURE_NOISE   = 5e-3  # 稍微降低測量噪聲
EMA_ALPHA_POS      = 0.15  # 增加位置平滑度
EMA_ALPHA_ANGLE    = 0.12  # 增加角度平滑度
MAX_MEAS_JUMP_PX   = 80
WARMUP_FRAMES_FOR_FPS = 30

# 初始檢測增強參數
INITIAL_DETECTION_RETRIES = 50   # 初始檢測重試次數
INITIAL_DETECTION_WARMUP = 20    # 初始檢測前的暖機幀數
STATIC_DETECTION_THRESHOLD = 2.0 # 靜止檢測閾值 (mm)
STATIC_SMOOTHING_FRAMES = 10     # 靜止時的額外平滑幀數

# ========== 函數產生器設定 ==========
FG_RESOURCE_STRING = 'USB0::0x0957::0x5707::MY59001615::0::INSTR'

# ========== 資料處理函數 ==========
def unwrap_angles_deg(a_deg):
    """角度展開"""
    if len(a_deg) == 0: return a_deg
    return np.rad2deg(np.unwrap(np.deg2rad(a_deg)))

def wrap_angles_deg(a_deg):
    """角度包裝"""
    return ((a_deg + 90) % 180) - 90

def moving_average(a, w):
    """移動平均"""
    if w is None or w <= 1: return a
    a = np.asarray(a, dtype=float)
    if len(a) < w: return a
    kernel = np.ones(w) / float(w)
    return np.convolve(a, kernel, mode='same')

def finite_diff(values, t, smooth_win=1):
    """有限差分計算導數 - 使用實際時間戳記"""
    v = np.full_like(values, np.nan, dtype=float)
    for i in range(1, len(values)):
        if np.isfinite(values[i]) and np.isfinite(values[i-1]) and (t[i] > t[i-1]):
            dt = t[i] - t[i-1]
            if dt > 0:  # 確保時間差為正
                v[i] = (values[i] - values[i-1]) / dt
    if smooth_win and smooth_win > 1:
        v = moving_average(v, smooth_win)
    return v

class FunctionGeneratorController:
    """函數產生器控制器 - 高速電壓切換版本"""
    
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
    
    def setup_continuous_output(self):
        """設定連續輸出模式 - 所有模式預載入，透過電壓切換"""
        if not self.connected:
            return False
            
        try:
            print("設定連續輸出模式...")
            
            # 預設使用 Mode 1 作為基準輸出
            base_mode = self.sampling_rates[1]
            
            # 設定基本輸出參數
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
            
            # 設定同步
            self.inst.write('SOUR1:TRACK OFF')
            self.inst.write('SOUR2:TRACK OFF')
            self.inst.write('SOUR2:TRACK ON')
            self.inst.write('SOUR2:PHAS:SYNC')
            self.inst.write('*WAI')
            
            # 初始設定：電壓為0 (待機狀態)
            self.inst.write('SOUR1:VOLT 0')
            self.inst.write('SOUR2:VOLT 0')
            self.inst.write('OUTP1:POL NORM')
            self.inst.write('OUTP2:POL INV')
            self.inst.write('*WAI')
            
            # 開啟輸出並保持常駐
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
        """快速模式切換 - 使用電壓調整方式"""
        if not self.connected or mode_num not in self.sampling_rates:
            return False
        
        # 如果尚未設定連續輸出，先設定
        if not self.continuous_output_setup:
            if not self.setup_continuous_output():
                return False
        
        start_time = time.time()
        
        # 模式配置：電壓和極性
        mode_configs = {
            1: {
                'ch1_volt': 1.2, 'ch2_volt': 1.2,
                'ch1_pol': 'NORM', 'ch2_pol': 'INV', 
                'wave_type': '25k',
                'desc': '25k Hz, CH1=NORM, CH2=INV'
            },
            2: {
                'ch1_volt': 1.2, 'ch2_volt': 1.2,
                'ch1_pol': 'NORM', 'ch2_pol': 'INV',
                'wave_type': '47k', 
                'desc': '47k Hz, CH1=NORM, CH2=INV'
            },
            3: {
                'ch1_volt': 1.2, 'ch2_volt': 1.2,
                'ch1_pol': 'INV', 'ch2_pol': 'NORM',
                'wave_type': '25k',
                'desc': '25k Hz, CH1=INV, CH2=NORM'
            },
            4: {
                'ch1_volt': 1.2, 'ch2_volt': 1.2,
                'ch1_pol': 'INV', 'ch2_pol': 'NORM',
                'wave_type': '47k',
                'desc': '47k Hz, CH1=INV, CH2=NORM'
            }
        }
        
        config = mode_configs[mode_num]
        mode_data = self.sampling_rates[mode_num]
        
        try:
            # 檢查是否需要切換波形
            need_waveform_switch = False
            if self.current_mode is None:
                need_waveform_switch = config['wave_type'] != '25k'  # 預設是25k
            else:
                current_config = mode_configs[self.current_mode]
                need_waveform_switch = current_config['wave_type'] != config['wave_type']
            
            # 檢查是否需要極性變化
            need_polarity_change = False
            if self.current_mode is None:
                need_polarity_change = True
            else:
                current_config = mode_configs[self.current_mode]
                need_polarity_change = (current_config['ch1_pol'] != config['ch1_pol'] or 
                                      current_config['ch2_pol'] != config['ch2_pol'])
            
            # 如果需要切換波形或極性，暫時關閉輸出
            if need_waveform_switch or need_polarity_change:
                self.inst.write('OUTP1 OFF; OUTP2 OFF')
                
                if need_waveform_switch:
                    # 切換波形
                    self.inst.write(f'SOUR1:FUNC:ARB {mode_data["name1"]}')
                    self.inst.write(f'SOUR2:FUNC:ARB {mode_data["name2"]}')
                    self.inst.write(f'SOUR1:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
                    self.inst.write(f'SOUR2:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
                    self.inst.write(f'SOUR1:FREQ {mode_data["freq"]}')
                    self.inst.write(f'SOUR2:FREQ {mode_data["freq"]}')
                    self.inst.write('SOUR2:PHAS:SYNC')
                    self.inst.write('*WAI')
                
                if need_polarity_change:
                    # 設定極性
                    self.inst.write(f'OUTP1:POL {config["ch1_pol"]}')
                    self.inst.write(f'OUTP2:POL {config["ch2_pol"]}')
                    self.inst.write('*WAI')
                
                # 重新開啟輸出
                self.inst.write('OUTP1 ON; OUTP2 ON')
                self.inst.write('*WAI')
            
            # 最快的操作：調整電壓
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
        """關閉函數產生器輸出 - 使用電壓調整方式"""
        if not self.connected:
            return
        try:
            # 使用電壓調整而非關閉輸出，保持連續性
            self.inst.write('SOUR1:VOLT 0; SOUR2:VOLT 0')
            self.inst.write('*WAI')
            self.current_mode = None
            print("函數產生器輸出已關閉 (電壓=0V)")
        except Exception as e:
            print(f"關閉函數產生器失敗: {e}")

class CameraTracker:
    """攝影機追蹤器 - 修正FPS和位置計算"""
    
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
        self.gif_frames = []
        self.gif_path = None
        
        # 修正的時間戳記計算 - 使用實際測量FPS而非假設值
        self.start_time = None
        self.current_timestamp = None
        self.fps_measurement_started = False
        self.measured_fps = None
        self.fps_samples = deque(maxlen=200)  # 增加樣本數以獲得更穩定的FPS測量
        self.fps_measurement_window = 3.0     # 3秒測量窗口
        self.last_frame_time = None
        self.frame_times = deque(maxlen=1000) # 儲存實際幀時間用於分析
        
        # FPS統計
        self.fps_stats = {
            'min': np.inf,
            'max': 0,
            'mean': 0,
            'std': 0,
            'count': 0
        }
        
    def initialize_camera(self):
        """初始化攝影機"""
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS_REQ)
        
        # 暖機
        print("攝影機暖機中...")
        for _ in range(INITIAL_DETECTION_WARMUP):
            self.cap.read()
        
        ok, first = self.cap.read()
        if not ok:
            raise RuntimeError("攝影機開啟失敗")
        
        # 估算 mm/px - 改進網格檢測準確性，考慮設備尺寸
        if AUTO_GRID_MM_PER_PX:
            self.mm_per_px = self.estimate_mm_per_px_single_frame(first)
            if self.mm_per_px is not None:
                # 驗證網格檢測結果是否合理
                device_width_px = DEVICE_WIDTH_MM / self.mm_per_px
                device_height_px = DEVICE_HEIGHT_MM / self.mm_per_px
                print(f"網格比例：{self.mm_per_px:.4f} mm/px")
                print(f"設備尺寸估算：{device_width_px:.1f} × {device_height_px:.1f} pixels ({DEVICE_WIDTH_MM}mm × {DEVICE_HEIGHT_MM}mm)")
                print(f"網格間距：{GRID_SPACING_MM / self.mm_per_px:.1f} pixels ({GRID_SPACING_MM}mm)")
        else:
            self.mm_per_px = MANUAL_MM_PER_PX
            
        if self.mm_per_px is None:
            print("無法自動檢測網格，使用預設比例")
            self.mm_per_px = 0.1
        
        # 增強初始檢測 - 多次嘗試尋找目標
        print("正在尋找追蹤目標...")
        initial_detection = None
        for attempt in range(INITIAL_DETECTION_RETRIES):
            ok, frame = self.cap.read()
            if not ok:
                continue
                
            # 先旋轉畫面再檢測
            frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            detection = self.find_target_and_angle(frame_rotated)
            
            if detection is not None:
                initial_detection = detection
                print(f"✓ 目標檢測成功 (嘗試 {attempt + 1}/{INITIAL_DETECTION_RETRIES})")
                break
            
            if (attempt + 1) % 10 == 0:
                print(f"持續尋找目標... ({attempt + 1}/{INITIAL_DETECTION_RETRIES})")
        
        # 初始化 Kalman 濾波器
        self.kf = self.make_kalman()
        if initial_detection is not None:
            cx0, cy0, ang0, box0 = initial_detection
            print(f"✓ 初始位置: ({cx0:.1f}, {cy0:.1f}), 角度: {ang0:.1f}°")
        else:
            cx0, cy0, ang0 = CAM_WIDTH/2.0, CAM_HEIGHT/2.0, 0.0
            print("未檢測到目標，使用預設位置。請確保目標在視野中且光線充足。")
        
        self.kf.x[:2] = np.array([[cx0], [cy0]], dtype=np.float32)
        self.ema_x = cx0
        self.ema_y = cy0
        self.ema_ang = ang0
        
        print(f"✓ 攝影機初始化完成 (解析度: {CAM_WIDTH}×{CAM_HEIGHT}, 比例: {self.mm_per_px:.4f} mm/px)")
        return True
    
    def measure_actual_fps(self):
        """測量實際FPS - 修正版本"""
        current_time = time.time()
        
        if not self.fps_measurement_started:
            self.fps_measurement_started = True
            self.last_frame_time = current_time
            return None
            
        if self.last_frame_time is not None:
            dt = current_time - self.last_frame_time
            if dt > 0:
                instantaneous_fps = 1.0 / dt
                self.fps_samples.append(instantaneous_fps)
                self.frame_times.append(current_time)
                
                # 更新統計
                self.fps_stats['count'] += 1
                self.fps_stats['min'] = min(self.fps_stats['min'], instantaneous_fps)
                self.fps_stats['max'] = max(self.fps_stats['max'], instantaneous_fps)
                
                # 計算移動平均FPS
                if len(self.fps_samples) >= 10:
                    recent_fps = list(self.fps_samples)[-50:]  # 最近50幀
                    self.measured_fps = np.mean(recent_fps)
                    self.fps_stats['mean'] = np.mean(list(self.fps_samples))
                    self.fps_stats['std'] = np.std(list(self.fps_samples))
                
        self.last_frame_time = current_time
        return self.measured_fps
    
    def get_accurate_timestamp(self):
        """獲取準確的時間戳記"""
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
            self.current_timestamp = 0.0
        else:
            self.current_timestamp = current_time - self.start_time
            
        return self.current_timestamp
    
    def estimate_mm_per_px_single_frame(self, frame):
        """估算單幀的mm/px比例 - 基於5mm×5mm網格"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 檢測水平線
            horizontal_lines = cv2.HoughLinesP(
                cv2.Canny(gray, 50, 150), 
                1, np.pi/180, threshold=100, 
                minLineLength=50, maxLineGap=10
            )
            
            # 檢測垂直線
            vertical_lines = cv2.HoughLinesP(
                cv2.Canny(gray, 50, 150), 
                1, np.pi/2, threshold=100, 
                minLineLength=50, maxLineGap=10
            )
            
            if horizontal_lines is not None and vertical_lines is not None:
                # 計算水平間距
                h_spacings = []
                for i in range(len(horizontal_lines) - 1):
                    y1 = (horizontal_lines[i][0][1] + horizontal_lines[i][0][3]) / 2
                    y2 = (horizontal_lines[i+1][0][1] + horizontal_lines[i+1][0][3]) / 2
                    spacing = abs(y2 - y1)
                    if 20 < spacing < 200:  # 合理範圍
                        h_spacings.append(spacing)
                
                # 計算垂直間距
                v_spacings = []
                for i in range(len(vertical_lines) - 1):
                    x1 = (vertical_lines[i][0][0] + vertical_lines[i][0][2]) / 2
                    x2 = (vertical_lines[i+1][0][0] + vertical_lines[i+1][0][2]) / 2
                    spacing = abs(x2 - x1)
                    if 20 < spacing < 200:  # 合理範圍
                        v_spacings.append(spacing)
                
                if h_spacings and v_spacings:
                    avg_h_spacing = np.median(h_spacings)
                    avg_v_spacing = np.median(v_spacings)
                    avg_spacing_px = (avg_h_spacing + avg_v_spacing) / 2
                    
                    mm_per_px = GRID_SPACING_MM / avg_spacing_px
                    print(f"檢測到網格間距：{avg_spacing_px:.1f} pixels → {mm_per_px:.4f} mm/px")
                    return mm_per_px
            
        except Exception as e:
            print(f"網格檢測失敗：{e}")
        
        return None
    
    def find_target_and_angle(self, frame):
        """尋找目標和角度 - 考慮9mm×6mm設備尺寸"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 黃色遮罩
        mask_yellow = cv2.inRange(hsv, HSV_YELLOW_LO, HSV_YELLOW_HI)
        
        # 形態學操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        
        # 尋找輪廓
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 根據設備尺寸篩選輪廓
        expected_area_px = (DEVICE_WIDTH_MM * DEVICE_HEIGHT_MM) / (self.mm_per_px ** 2)
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_CONTOUR_AREA:
                # 檢查面積是否符合設備尺寸
                area_ratio = area / expected_area_px
                if 0.3 <= area_ratio <= 3.0:  # 合理範圍
                    valid_contours.append(contour)
        
        if not valid_contours:
            # 如果沒有符合尺寸的，使用最大的
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > MIN_CONTOUR_AREA:
                valid_contours = [largest_contour]
        
        if not valid_contours:
            return None
        
        # 選擇最大的有效輪廓
        contour = max(valid_contours, key=cv2.contourArea)
        
        # 計算中心和角度
        try:
            # 使用最小外接矩形
            rect = cv2.minAreaRect(contour)
            cx, cy = rect[0]
            angle = rect[2]
            
            # 角度標準化
            if rect[1][0] < rect[1][1]:  # width < height
                angle += 90
            
            angle = angle % 180
            if angle > 90:
                angle -= 180
            
            # 獲取矩形頂點
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            return cx, cy, angle, box
            
        except Exception as e:
            print(f"角度計算失敗：{e}")
            return None
    
    def make_kalman(self):
        """建立Kalman濾波器"""
        kf = cv2.KalmanFilter(4, 2)  # 4 states, 2 measurements
        
        # 狀態轉移矩陣 (x, y, vx, vy)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 測量矩陣
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # 過程噪聲協方差
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * KF_PROCESS_NOISE
        
        # 測量噪聲協方差
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KF_MEASURE_NOISE
        
        # 錯誤協方差
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        return kf
    
    def start_recording(self):
        """開始錄製"""
        if self.state != 0:
            return
            
        # 建立輸出目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{timestamp}_{MODE}_integrated"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 設定影片輸出
        if RECORD_OUTPUT:
            self.out_path = os.path.join(self.output_dir, f"camera_{MODE}_tracked.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 使用測量的FPS或預設值
            fps_for_video = self.measured_fps if self.measured_fps else CAM_FPS_REQ
            self.writer = cv2.VideoWriter(self.out_path, fourcc, fps_for_video, (CAM_HEIGHT, CAM_WIDTH))
            
            # GIF路徑
            if HAVE_PIL:
                self.gif_path = os.path.join(self.output_dir, f"camera_{MODE}_tracked.gif")
        
        # 重置記錄
        self.rec = []
        self.frame_buf = []
        self.ts_list = []
        self.frame_idx = 0
        self.gif_frames = []
        
        # 重置時間戳記
        self.start_time = None
        self.fps_measurement_started = False
        self.fps_samples.clear()
        self.frame_times.clear()
        
        self.state = 1
        print(f"✓ 開始錄製到：{self.output_dir}")
    
    def stop_recording(self):
        """停止錄製並生成分析圖表"""
        if self.state != 1:
            return
            
        self.state = 0
        print("停止錄製，正在生成分析...")
        
        # 關閉影片寫入器
        if self.writer:
            self.writer.release()
            self.writer = None
        
        # 生成分析
        if len(self.rec) > 10:
            try:
                self.generate_analysis()
                print(f"✓ 分析完成，檔案儲存在：{self.output_dir}")
            except Exception as e:
                print(f"分析生成失敗：{e}")
        else:
            print("記錄數據不足，跳過分析生成")
    
    def process_frame(self, frame):
        """處理單幀 - 修正時間戳記計算"""
        # 測量實際FPS
        measured_fps = self.measure_actual_fps()
        
        # 獲取準確時間戳記
        timestamp = self.get_accurate_timestamp()
        
        # 旋轉畫面
        frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h, w = frame_rotated.shape[:2]
        
        # 尋找目標
        detection = self.find_target_and_angle(frame_rotated)
        
        # 預測和更新
        self.kf.predict()
        
        if detection is not None:
            cx_meas, cy_meas, ang_meas, box = detection
            
            # 檢查測量跳躍
            if (self.ema_x is not None and self.ema_y is not None and
                np.sqrt((cx_meas - self.ema_x)**2 + (cy_meas - self.ema_y)**2) > MAX_MEAS_JUMP_PX):
                # 跳躍太大，使用預測值
                cx_filt = self.kf.statePre[0, 0]
                cy_filt = self.kf.statePre[1, 0]
                ang_filt = self.ema_ang
            else:
                # 正常更新
                measurement = np.array([[cx_meas], [cy_meas]], dtype=np.float32)
                self.kf.correct(measurement)
                
                cx_filt = self.kf.statePost[0, 0]
                cy_filt = self.kf.statePost[1, 0]
                
                # EMA平滑角度
                if self.ema_ang is None:
                    ang_filt = ang_meas
                else:
                    ang_diff = ang_meas - self.ema_ang
                    if abs(ang_diff) > 90:
                        ang_diff = ang_diff - 180 * np.sign(ang_diff)
                    ang_filt = self.ema_ang + EMA_ALPHA_ANGLE * ang_diff
                
                # 更新EMA
                if self.ema_x is None:
                    self.ema_x, self.ema_y = cx_filt, cy_filt
                else:
                    self.ema_x += EMA_ALPHA_POS * (cx_filt - self.ema_x)
                    self.ema_y += EMA_ALPHA_POS * (cy_filt - self.ema_y)
                self.ema_ang = ang_filt
        else:
            # 沒有檢測到，使用預測值
            cx_filt = self.kf.statePre[0, 0]
            cy_filt = self.kf.statePre[1, 0]
            ang_filt = self.ema_ang
            box = None
        
        # 計算實際位置 (mm)
        if self.origin_x is None or self.origin_y is None:
            self.origin_x, self.origin_y = cx_filt, cy_filt
        
        x_mm_rel = (cx_filt - self.origin_x) * self.mm_per_px
        y_mm_rel = (cy_filt - self.origin_y) * self.mm_per_px
        
        if INVERT_Y_AXIS:
            y_mm_rel = -y_mm_rel
        
        x_mm_abs = cx_filt * self.mm_per_px
        y_mm_abs = cy_filt * self.mm_per_px
        
        # 計算瞬時速度 (使用實際時間戳記)
        if (self.last_x_mm_abs is not None and self.last_y_mm_abs is not None and 
            self.last_t is not None and timestamp > self.last_t):
            dt = timestamp - self.last_t
            if dt > 0:
                dx = x_mm_abs - self.last_x_mm_abs
                dy = y_mm_abs - self.last_y_mm_abs
                self.inst_speed = np.sqrt(dx*dx + dy*dy) / dt  # mm/s
        
        self.last_x_mm_abs = x_mm_abs
        self.last_y_mm_abs = y_mm_abs
        self.last_t = timestamp
        
        # 記錄數據
        if self.state == 1:  # 錄製中
            self.rec.append({
                'frame': self.frame_idx,
                'timestamp': timestamp,
                'x_px': cx_filt,
                'y_px': cy_filt,
                'angle_deg': ang_filt,
                'x_mm_rel': x_mm_rel,
                'y_mm_rel': y_mm_rel,
                'x_mm_abs': x_mm_abs,
                'y_mm_abs': y_mm_abs,
                'speed_mm_s': self.inst_speed,
                'fps': measured_fps if measured_fps else np.nan
            })
            
            # 儲存幀到緩衝區
            self.frame_buf.append(frame_rotated.copy())
            self.ts_list.append(timestamp)
            
            # 寫入影片
            if self.writer:
                self.writer.write(frame_rotated)
            
            # 儲存GIF幀
            if HAVE_PIL and len(self.gif_frames) < 300:  # 限制GIF大小
                if self.frame_idx % 3 == 0:  # 每3幀取1幀
                    gif_frame = cv2.cvtColor(frame_rotated, cv2.COLOR_BGR2RGB)
                    gif_frame = cv2.resize(gif_frame, (frame_rotated.shape[1]//2, frame_rotated.shape[0]//2))
                    self.gif_frames.append(Image.fromarray(gif_frame))
        
        # 繪製結果
        display_frame = frame_rotated.copy()
        
        # 繪製檢測框 - 使用紅色框表示設備尺寸
        if box is not None:
            cv2.drawContours(display_frame, [box], -1, (0, 0, 255), 2)  # 紅色框
        
        # 繪製中心點
        center = (int(cx_filt), int(cy_filt))
        cv2.circle(display_frame, center, 5, (0, 255, 0), -1)
        
        # 繪製方向線
        if ang_filt is not None:
            length = 30
            end_x = int(cx_filt + length * np.cos(np.radians(ang_filt)))
            end_y = int(cy_filt + length * np.sin(np.radians(ang_filt)))
            cv2.arrowedLine(display_frame, center, (end_x, end_y), (255, 0, 0), 2)
        
        # 顯示資訊
        info_lines = [
            f"FPS: {measured_fps:.1f}" if measured_fps else f"FPS: {CAM_FPS_REQ}(設定)",
            f"位置: ({x_mm_rel:.1f}, {y_mm_rel:.1f}) mm",
            f"角度: {ang_filt:.1f}°" if ang_filt is not None else "角度: N/A",
            f"速度: {self.inst_speed:.1f} mm/s" if np.isfinite(self.inst_speed) else "速度: N/A",
            f"狀態: {'錄製中' if self.state == 1 else '預覽'}",
        ]
        
        if self.fg_controller.current_mode:
            info_lines.append(f"FG: Mode {self.fg_controller.current_mode}")
        else:
            info_lines.append("FG: 關閉")
        
        for i, line in enumerate(info_lines):
            cv2.putText(display_frame, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        self.frame_idx += 1
        
        # 調整顯示大小
        if DISPLAY_SCALE != 1.0:
            new_w = int(display_frame.shape[1] * DISPLAY_SCALE)
            new_h = int(display_frame.shape[0] * DISPLAY_SCALE)
            display_frame = cv2.resize(display_frame, (new_w, new_h))
        
        return display_frame
    
    def generate_analysis(self):
        """生成分析圖表 - 使用實際時間戳記"""
        if len(self.rec) < 2:
            return
        
        df = pd.DataFrame(self.rec)
        
        # 保存CSV (包含FPS數據)
        csv_path = os.path.join(self.output_dir, f"camera_{MODE}_pos_angle_speed.csv")
        df.to_csv(csv_path, index=False)
        
        # 準備數據
        t = df['timestamp'].values  # 使用實際時間戳記
        x_mm = df['x_mm_rel'].values
        y_mm = df['y_mm_rel'].values
        angles = df['angle_deg'].values
        
        # 計算導數使用實際時間
        vx = finite_diff(x_mm, t, smooth_win=5)
        vy = finite_diff(y_mm, t, smooth_win=5)
        speed = np.sqrt(vx**2 + vy**2)
        
        # 角速度計算
        angles_unwrapped = unwrap_angles_deg(angles)
        angular_speed = finite_diff(angles_unwrapped, t, smooth_win=5)
        
        # 生成圖表
        plot_paths = self.create_plots(t, x_mm, y_mm, angles, speed, angular_speed, df)
        
        # 生成GIF
        if HAVE_PIL and self.gif_frames and self.gif_path:
            try:
                self.gif_frames[0].save(
                    self.gif_path,
                    save_all=True,
                    append_images=self.gif_frames[1:],
                    duration=100,  # 100ms per frame
                    loop=0
                )
                print(f"✓ GIF 已儲存：{self.gif_path}")
            except Exception as e:
                print(f"GIF 生成失敗：{e}")
        
        return plot_paths
    
    def create_plots(self, t, x_mm, y_mm, angles, speed, angular_speed, df):
        """建立分析圖表"""
        plot_paths = {}
        
        # 圖1：位置軌跡
        plt.figure(figsize=(10, 8))
        plt.plot(x_mm, y_mm, 'b-', linewidth=2, alpha=0.7, label='軌跡')
        plt.scatter(x_mm[0], y_mm[0], color='green', s=100, marker='o', label='起點', zorder=5)
        plt.scatter(x_mm[-1], y_mm[-1], color='red', s=100, marker='s', label='終點', zorder=5)
        
        # 添加網格參考線
        x_range = max(x_mm) - min(x_mm)
        y_range = max(y_mm) - min(y_mm)
        plot_range = max(x_range, y_range) * PLOT_RANGE_SCALE
        
        x_center = (max(x_mm) + min(x_mm)) / 2
        y_center = (max(y_mm) + min(y_mm)) / 2
        
        # 5mm網格線
        grid_spacing = GRID_SPACING_MM
        x_grid = np.arange(x_center - plot_range/2, x_center + plot_range/2 + grid_spacing, grid_spacing)
        y_grid = np.arange(y_center - plot_range/2, y_center + plot_range/2 + grid_spacing, grid_spacing)
        
        for x_line in x_grid:
            plt.axvline(x=x_line, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        for y_line in y_grid:
            plt.axhline(y=y_line, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        
        plt.xlabel('X 位置 (mm)')
        plt.ylabel('Y 位置 (mm)')
        plt.title(f'設備位置軌跡 - {DEVICE_WIDTH_MM}mm×{DEVICE_HEIGHT_MM}mm 設備於 {GRID_SPACING_MM}mm×{GRID_SPACING_MM}mm 網格')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        pos_path = os.path.join(self.output_dir, f"camera_{MODE}_position.png")
        plt.tight_layout()
        plt.savefig(pos_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['position'] = pos_path
        
        # 圖2：速度和角度隨時間變化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 速度
        ax1.plot(t, speed, 'r-', linewidth=1.5, label='線性速度')
        ax1.set_xlabel('時間 (s)')
        ax1.set_ylabel('速度 (mm/s)')
        ax1.set_title('線性速度隨時間變化')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 角度
        if ORIENT_PLOT_WRAPPED:
            angles_plot = wrap_angles_deg(angles)
            ax2.set_ylim([-ORIENT_YLIM_DEG, ORIENT_YLIM_DEG])
        else:
            angles_plot = unwrap_angles_deg(angles)
        
        ax2.plot(t, angles_plot, 'g-', linewidth=1.5, label='角度')
        ax2.set_xlabel('時間 (s)')
        ax2.set_ylabel('角度 (度)')
        ax2.set_title('角度隨時間變化')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 角速度
        ax3.plot(t, angular_speed, 'm-', linewidth=1.5, label='角速度')
        ax3.set_xlabel('時間 (s)')
        ax3.set_ylabel('角速度 (度/s)')
        ax3.set_title('角速度隨時間變化')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # FPS統計
        if 'fps' in df.columns:
            fps_data = df['fps'].dropna()
            if len(fps_data) > 0:
                ax4.plot(t[:len(fps_data)], fps_data, 'c-', linewidth=1.5, label='實際FPS')
                ax4.axhline(y=fps_data.mean(), color='orange', linestyle='--', 
                           label=f'平均FPS: {fps_data.mean():.1f}')
                ax4.set_xlabel('時間 (s)')
                ax4.set_ylabel('FPS')
                ax4.set_title('實際幀率隨時間變化')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
        
        plt.tight_layout()
        speed_path = os.path.join(self.output_dir, f"camera_{MODE}_speed_orientation.png")
        plt.savefig(speed_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['speed_orientation'] = speed_path
        
        return plot_paths
    
    def cleanup(self):
        """清理資源"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

def show_help():
    """顯示幫助資訊"""
    help_text = """
=== 整合式攝影機追蹤與函數產生器控制 ===

攝影機控制：
  [Space]  開始/停止錄製
  [ESC/Q]  退出程式

函數產生器控制：
  [1]      切換到模式 1 (25kHz, CH1=NORM, CH2=INV)
  [2]      切換到模式 2 (47kHz, CH1=NORM, CH2=INV)  
  [3]      切換到模式 3 (25kHz, CH1=INV, CH2=NORM)
  [4]      切換到模式 4 (47kHz, CH1=INV, CH2=NORM)
  [0]      關閉輸出

其他：
  [H]      顯示此幫助
  [M]      切換操作模式

設備規格：
  - 設備尺寸：9mm × 6mm (紅色框線)
  - 網格間距：5mm × 5mm
  - FPS：實際測量（修正假設120FPS問題）

修正重點：
  ✓ 使用實際時間戳記而非假設FPS
  ✓ 基於5mm網格的準確比例計算
  ✓ 9mm×6mm設備尺寸驗證
  ✓ 改進的速度計算（position/實際時間）
"""
    print(help_text)

def main():
    """主函數"""
    print("=== 整合式攝影機追蹤與函數產生器控制系統 ===")
    print("初始化中...")
    
    # 初始化函數產生器控制器
    fg_controller = FunctionGeneratorController()
    fg_connected = fg_controller.connect()
    
    # 初始化攝影機追蹤器
    tracker = CameraTracker(fg_controller)
    
    try:
        if not tracker.initialize_camera():
            print("攝影機初始化失敗")
            return
        
        print("✓ 系統初始化完成")
        show_help()
        
        print(f"\n攝影機規格:")
        print(f"- 解析度: {CAM_WIDTH}×{CAM_HEIGHT}")
        print(f"- 設定FPS: {CAM_FPS_REQ}")
        print(f"- 實際測量FPS: 即時計算")
        print(f"- 設備尺寸: {DEVICE_WIDTH_MM}mm × {DEVICE_HEIGHT_MM}mm")
        print(f"- 網格間距: {GRID_SPACING_MM}mm × {GRID_SPACING_MM}mm")
        print(f"- 比例: {tracker.mm_per_px:.4f} mm/px")
        
        if fg_connected:
            print(f"\n函數產生器: 已連接並準備就緒")
        else:
            print(f"\n函數產生器: 未連接 (僅攝影機模式)")
        
        print("\n按 [H] 查看操作指南，[Space] 開始錄製...")
        
        # 主迴圈
        while True:
            ret, frame = tracker.cap.read()
            if not ret:
                print("攝影機讀取失敗")
                break
            
            # 處理幀
            display_frame = tracker.process_frame(frame)
            
            # 顯示
            cv2.imshow(WINDOW_TITLE, display_frame)
            
            # 鍵盤輸入處理
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - 開始/停止錄製
                if tracker.state == 0:
                    tracker.start_recording()
                else:
                    tracker.stop_recording()
            
            elif key == ord('h') or key == ord('H'):  # H - 幫助
                show_help()
            
            elif key == ord('q') or key == 27:  # Q or ESC - 退出
                break
            
            elif key == ord('m') or key == ord('M'):  # M - 切換操作模式
                print("模式切換功能預留")
            
            # 函數產生器控制
            elif fg_connected:
                if key == ord('1'):
                    fg_controller.switch_mode(1)
                elif key == ord('2'):
                    fg_controller.switch_mode(2)
                elif key == ord('3'):
                    fg_controller.switch_mode(3)
                elif key == ord('4'):
                    fg_controller.switch_mode(4)
                elif key == ord('0'):
                    fg_controller.turn_off()
    
    except KeyboardInterrupt:
        print("\n程式被使用者中斷")
    except Exception as e:
        print(f"程式發生錯誤：{e}")
    finally:
        # 清理資源
        print("正在清理資源...")
        tracker.cleanup()
        if fg_connected:
            fg_controller.disconnect()
        print("✓ 程式已退出")

if __name__ == "__main__":
    main()
