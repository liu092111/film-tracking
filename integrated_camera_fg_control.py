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
DISPLAY_SCALE     = 0.5  # 顯示縮放比例 - 縮小顯示畫面

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
    """有限差分計算導數"""
    v = np.full_like(values, np.nan, dtype=float)
    for i in range(1, len(values)):
        if np.isfinite(values[i]) and np.isfinite(values[i-1]) and (t[i] > t[i-1]):
            v[i] = (values[i] - values[i-1]) / (t[i] - t[i-1])
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
        self.output_dir = None  # 儲存輸出目錄路徑
        self.out_path = None    # 儲存影片檔案路徑
        
        # 改善 FPS 計算 - 增加緩衝區大小以收集更多數據
        self.fps_estimation_buffer = deque(maxlen=300)  # FPS 估算緩衝區
        self.last_frame_time = None
        self.actual_fps_buffer = deque(maxlen=600)  # 實際FPS計算緩衝區（增加容量）
        self.frame_timestamps = deque(maxlen=600)   # 幀時間戳緩衝區（增加容量）
        self.all_fps_data = []  # 儲存所有FPS數據用於CSV輸出
        
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
        
        # 估算 mm/px
        if AUTO_GRID_MM_PER_PX:
            self.mm_per_px = self.estimate_mm_per_px_single_frame(first)
        else:
            self.mm_per_px = MANUAL_MM_PER_PX
        if self.mm_per_px is None:
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
        
        self.kf.statePost = np.array([[cx0],[cy0],[0.0],[0.0]], dtype=np.float32)
        
        # 初始化位置歷史（用於靜止檢測）
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
        """估算 mm/px 比例 - 改善網格檢測"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)  # 減少模糊以保留更多細節
        edges = cv2.Canny(gray, 50, 150)  # 調整Canny參數
        # 增加線檢測的敏感度
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
            if abs(ang) < 8:  # 更嚴格的水平線判定
                horiz.extend([y1, y2])
            elif abs(abs(ang) - 90) < 8:  # 更嚴格的垂直線判定
                vert.extend([x1, x2])

        def spacing(pos):
            if len(pos) < 4: return None
            pos = np.array(sorted(pos))
            # 更精細的去重處理
            uniq = [pos[0]]
            for v in pos[1:]:
                if abs(v - uniq[-1]) > 1.5:  # 降低去重閾值
                    uniq.append(v)
            uniq = np.array(uniq)
            if len(uniq) < 3: return None
            
            diffs = np.diff(uniq)
            # 過濾掉異常值
            diffs = diffs[(diffs > 2) & (diffs < 300)]
            if len(diffs) == 0: return None
            
            # 尋找最小的合理間距（應該對應5mm網格）
            min_spacing = float(np.min(diffs))
            median_spacing = float(np.median(diffs))
            
            # 如果最小間距和中位數間距相差很大，可能檢測到了10mm間距
            if median_spacing > min_spacing * 1.8:
                return min_spacing
            else:
                return median_spacing

        sp_h = spacing(horiz); sp_v = spacing(vert)
        
        if sp_h and sp_v: 
            px_per_cell = (sp_h + sp_v)/2.0
        elif sp_h:        
            px_per_cell = sp_h
        elif sp_v:        
            px_per_cell = sp_v
        else:             
            return None
            
        # 修正10倍放大問題：確保使用正確的網格間距計算
        mm_per_px = GRID_SPACING_MM / px_per_cell if (px_per_cell and px_per_cell > 0) else None
        
        # 診斷輸出
        if mm_per_px:
            print(f"網格檢測結果：")
            if sp_h: print(f"  水平間距: {sp_h:.1f} pixels")
            if sp_v: print(f"  垂直間距: {sp_v:.1f} pixels")
            print(f"  平均間距: {px_per_cell:.1f} pixels")
            print(f"  計算比例: {mm_per_px:.4f} mm/pixel")
            # 驗證：5mm網格應該對應多少像素
            expected_pixels = GRID_SPACING_MM / mm_per_px
            print(f"  驗證：5mm網格 = {expected_pixels:.1f} pixels")
        
        return mm_per_px

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
        """尋找目標並計算角度 - 修正 90 度偏移問題"""
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
        
        # 修正 90 度偏移問題：不需要額外加 90 度
        if rw >= rh: 
            angle_deg = rect_angle  # 移除 + 90.0
        else:        
            angle_deg = rect_angle - 90.0  # 調整為減 90 度
        
        # 正規化角度到 [-90, 90) 範圍
        while angle_deg >= 90.0: angle_deg -= 180.0
        while angle_deg < -90.0: angle_deg += 180.0

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

        # 順時針旋轉90度以修正鏡像問題（先旋轉畫面內容）
        frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Kalman 預測
        pred = self.kf.predict()
        px_pred, py_pred = float(pred[0,0]), float(pred[1,0])

        # 目標檢測（在旋轉後的畫面上進行）
        m = self.find_target_and_angle(frame_rotated)
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

        # 繪製結果（在旋轉後的畫面上）
        if box is None:
            sz = 20
            bx = np.array([[fx-sz, fy-sz],[fx+sz, fy-sz],[fx+sz, fy+sz],[fx-sz, fy+sz]], dtype=int)
            box_draw = bx
            green_center_x, green_center_y = fx, fy
        else:
            box_draw = box
            green_center_x, green_center_y = cx, cy

        # 繪製追蹤結果
        cv2.polylines(frame_rotated, [box_draw], isClosed=True, color=(0,0,255), thickness=3)
        if np.isfinite(green_center_x) and np.isfinite(green_center_y):
            cv2.circle(frame_rotated, (int(round(green_center_x)), int(round(green_center_y))), 6, (0,255,0), -1)

        # 計算座標和速度
        fx_mm_abs = fx * self.mm_per_px if np.isfinite(fx) else np.nan
        fy_mm_abs = fy * self.mm_per_px if np.isfinite(fy) else np.nan

        fps_device = self.cap.get(cv2.CAP_PROP_FPS) or float(CAM_FPS_REQ)
        t_s = self.frame_idx / (fps_device if fps_device > 0 else 30.0)

        # 靜止檢測和額外平滑
        if np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs):
            current_pos = (fx_mm_abs, fy_mm_abs)
            self.position_history.append(current_pos)
            
            # 檢測是否靜止
            if len(self.position_history) >= STATIC_SMOOTHING_FRAMES:
                positions = np.array(self.position_history)
                max_movement = np.max(np.sqrt(np.sum((positions - positions[0])**2, axis=1)))
                self.is_static = max_movement < STATIC_DETECTION_THRESHOLD
                
                # 如果靜止，使用位置歷史的平均值進行額外平滑
                if self.is_static:
                    avg_pos = np.mean(positions, axis=0)
                    fx_mm_abs, fy_mm_abs = avg_pos[0], avg_pos[1]
                    # 同時平滑像素座標
                    fx = fx_mm_abs / self.mm_per_px
                    fy = fy_mm_abs / self.mm_per_px

        if (self.last_x_mm_abs is not None and self.last_y_mm_abs is not None and self.last_t is not None and
            np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs) and (t_s > self.last_t)):
            raw_speed = np.hypot(fx_mm_abs-self.last_x_mm_abs, fy_mm_abs-self.last_y_mm_abs) / (t_s - self.last_t)
            # 靜止時將速度設為接近零
            if hasattr(self, 'is_static') and self.is_static:
                self.inst_speed = 0.0
            else:
                self.inst_speed = raw_speed
        self.last_x_mm_abs, self.last_y_mm_abs, self.last_t = fx_mm_abs, fy_mm_abs, t_s

        if self.origin_x is None and np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs):
            self.origin_x, self.origin_y = fx_mm_abs, fy_mm_abs

        # 準備顯示用的畫面（複製旋轉後的畫面來添加文字）
        display_frame = frame_rotated.copy()
        
        # 疊字顯示（文字保持正常方向）
        if self.origin_x is not None and np.isfinite(fx_mm_abs) and np.isfinite(fy_mm_abs):
            rx = fx_mm_abs - self.origin_x
            ry = fy_mm_abs - self.origin_y
            pos_text = f"Pos(mm): ({rx:.2f}, {ry:.2f})"
        else:
            pos_text = f"Pos(mm): (NaN, NaN)"
        cv2.putText(display_frame, pos_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        if np.isfinite(ang_for_draw):
            cv2.putText(display_frame, f"Angle: {ang_for_draw:+.2f} deg", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(display_frame, "Angle: NaN", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
        if np.isfinite(self.inst_speed):
            cv2.putText(display_frame, f"Speed: {self.inst_speed:.2f} mm/s", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(display_frame, "Speed: NaN mm/s", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # 函數產生器狀態顯示
        fg_status = f"FG: Mode {self.fg_controller.current_mode}" if self.fg_controller.current_mode else "FG: OFF"
        cv2.putText(display_frame, fg_status, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # 錄影狀態顯示 - 持續顯示REC指示器
        if self.state == 1:  # RECORDING
            # REC文字 - 放在右上角更明顯的位置，持續顯示
            cv2.putText(display_frame, "REC", (CAM_HEIGHT-100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
            # 紅色圓點指示器
            cv2.circle(display_frame, (CAM_HEIGHT-130, 25), 8, (0,0,255), -1)
        
        # 狀態提示（由於畫面旋轉了，調整文字位置）
        if self.state == 0:  # PREVIEW
            # 旋轉後畫面尺寸變化，調整文字位置
            cv2.putText(display_frame, "INTEGRATED PREVIEW - [SPACE] Start Recording, [1-4] FG Mode, [0] FG Off",
                        (20, CAM_WIDTH-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # 改善的FPS計算與錄影處理
        if self.state == 1 and RECORD_OUTPUT:  # RECORD
            ts_now = time.perf_counter()
            self.ts_list.append(ts_now)
            self.frame_timestamps.append(ts_now)
            
            # 計算實際FPS
            if len(self.frame_timestamps) >= 2:
                recent_dt = self.frame_timestamps[-1] - self.frame_timestamps[-2]
                if recent_dt > 0:
                    current_fps = 1.0 / recent_dt
                    self.actual_fps_buffer.append(current_fps)
            
            if self.writer is None:
                self.frame_buf.append(display_frame.copy())  # 使用包含文字的畫面
                
                if len(self.frame_buf) >= WARMUP_FRAMES_FOR_FPS and len(self.ts_list) >= WARMUP_FRAMES_FOR_FPS:
                    # 使用實際測量的FPS而不是設備FPS
                    if len(self.actual_fps_buffer) > 0:
                        fps_out = float(np.median(list(self.actual_fps_buffer)))
                    else:
                        dt = np.diff(np.array(self.ts_list[-WARMUP_FRAMES_FOR_FPS:], dtype=float))
                        dt = dt[dt > 0]
                        fps_out = float(1.0 / np.median(dt)) if dt.size > 0 else 30.0
                    
                    # 限制FPS範圍，避免異常值
                    fps_out = max(10.0, min(fps_out, 200.0))
                    
                    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.output_dir = f"{run_tag}_{MODE}_integrated"
                    os.makedirs(self.output_dir, exist_ok=True)
                    self.out_path = os.path.join(self.output_dir, f"camera_{MODE}_tracked.mp4")
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # 注意：由於畫面旋轉了90度，寬高要對調
                    W = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 旋轉後原高度變寬度
                    H = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 旋轉後原寬度變高度
                    self.writer = cv2.VideoWriter(self.out_path, fourcc, fps_out, (W, H))
                    
                    print(f"錄影參數：FPS={fps_out:.1f}, 解析度={W}x{H}")
                    
                    for f in self.frame_buf:
                        self.writer.write(f)
                    self.frame_buf.clear()
            else:
                self.writer.write(display_frame)  # 寫入包含文字的畫面

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
            
            # 縮放顯示畫面
            if DISPLAY_SCALE != 1.0:
                display_height, display_width = int(frame.shape[0] * DISPLAY_SCALE), int(frame.shape[1] * DISPLAY_SCALE)
                frame_display = cv2.resize(frame, (display_width, display_height))
            else:
                frame_display = frame
            
            # 顯示
            cv2.imshow(WINDOW_TITLE, frame_display)
            
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
                    # 立即關閉函數產生器
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
        # 收尾：確保 writer 被正確關閉
        camera_tracker.cap.release()
        # 若在 RECORDING 但 writer 尚未建立，補建、估 FPS 後寫出緩衝
        if camera_tracker.state == 1 and RECORD_OUTPUT and camera_tracker.writer is None and len(camera_tracker.frame_buf) > 0:
            dt_all = np.diff(np.array(camera_tracker.ts_list, dtype=float))
            dt_all = dt_all[dt_all > 0]
            fps_device = camera_tracker.cap.get(cv2.CAP_PROP_FPS) or float(CAM_FPS_REQ)
            fps_out = float(1.0 / np.median(dt_all)) if dt_all.size > 0 else (fps_device or 30.0)
            
            run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            camera_tracker.output_dir = f"{run_tag}_{MODE}_integrated"
            os.makedirs(camera_tracker.output_dir, exist_ok=True)
            camera_tracker.out_path = os.path.join(camera_tracker.output_dir, f"camera_{MODE}_tracked.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            W = int(CAM_WIDTH)
            H = int(CAM_HEIGHT)
            camera_tracker.writer = cv2.VideoWriter(camera_tracker.out_path, fourcc, fps_out, (W, H))
            for f in camera_tracker.frame_buf:
                camera_tracker.writer.write(f)
            camera_tracker.frame_buf.clear()
        
        if camera_tracker.writer is not None:
            camera_tracker.writer.release()
        cv2.destroyAllWindows()

        # 資料處理和輸出（如果有錄製資料）
        if len(camera_tracker.rec) > 0:
            print("\n正在處理和輸出資料...")
            process_and_export_data(camera_tracker)
        else:
            print("未開始錄影，沒有輸出。")
        
        # 清理資源
        print("\n清理資源...")
        camera_tracker.cleanup_camera()
        fg_controller.disconnect()
        print("程式結束")

def process_and_export_data(camera_tracker):
    """處理並輸出追蹤資料 - 生成 CSV 和圖表"""
    if len(camera_tracker.rec) == 0:
        print("沒有追蹤資料可輸出")
        return
    
    # 使用已建立的輸出目錄，如果沒有則建立新的
    if camera_tracker.output_dir and os.path.exists(camera_tracker.output_dir):
        output_dir = camera_tracker.output_dir
    else:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{run_tag}_{MODE}_integrated"
        os.makedirs(output_dir, exist_ok=True)
    OUT_PREFIX = f"camera_{MODE}"
    
    # 整理資料為 DataFrame
    df = pd.DataFrame(camera_tracker.rec, columns=[
        "frame","t_s","x_px_filt","y_px_filt","x_mm_abs","y_mm_abs","angle_deg_raw","mm_per_px"
    ])
    
    # 計算相對起點座標 - 修正Y座標正負號
    x_abs = df["x_mm_abs"].to_numpy()
    y_abs = df["y_mm_abs"].to_numpy()
    valid = np.isfinite(x_abs) & np.isfinite(y_abs)
    if valid.any():
        x0, y0 = x_abs[valid][0], y_abs[valid][0]
        df["x_mm"] = x_abs - x0
        # 修正Y座標正負號 - 加負號
        df["y_mm"] = -(y_abs - y0)
    else:
        df["x_mm"] = x_abs
        df["y_mm"] = -y_abs  # 也對絕對座標加負號
    
    # 可選：Savitzky–Golay 平滑座標
    USE_SG_POS = False  # 可以根據需要調整
    SG_WIN = 7
    SG_POLY = 2
    if USE_SG_POS and HAVE_SG and len(df) >= SG_WIN:
        df["x_mm"] = savgol_filter(df["x_mm"].to_numpy(), SG_WIN, SG_POLY, mode="interp")
        df["y_mm"] = savgol_filter(df["y_mm"].to_numpy(), SG_WIN, SG_POLY, mode="interp")
    
    # 角度處理
    ang_raw = df["angle_deg_raw"].to_numpy()
    mask_ang = np.isfinite(ang_raw)
    ang_unwrap = np.full_like(ang_raw, np.nan, dtype=float)
    if mask_ang.any():
        ang_unwrap[mask_ang] = unwrap_angles_deg(ang_raw[mask_ang])
        ang_unwrap = moving_average(ang_unwrap, 5)
    df["angle_deg_unwrapped"] = ang_unwrap
    
    # 角速度計算
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
    
    # 線速度計算
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
    plot_paths = generate_plots(df, output_dir, OUT_PREFIX)
    
    # 輸出訊息
    print(f"[輸出] 目錄：{output_dir}")
    print(f"[輸出] CSV：{csv_path}")
    print(f"[輸出] Position 圖：{plot_paths['position']}")
    if MODE.lower() == "straight":
        print(f"[輸出] Speed/Orientation 圖：{plot_paths['speed_orientation']}")
    else:
        print(f"[輸出] Angular Speed 圖：{plot_paths['angular_speed']}")
    if RECORD_OUTPUT and camera_tracker.out_path and os.path.exists(camera_tracker.out_path):
        print(f"[輸出] 追蹤影片：{camera_tracker.out_path}")
    
    print(f"\n✓ 已成功輸出檔案：")
    print(f"  1. 追蹤影片：{os.path.basename(camera_tracker.out_path) if camera_tracker.out_path else 'N/A'}")
    print(f"  2. 位置圖表：{os.path.basename(plot_paths['position'])}")
    if MODE.lower() == "straight":
        print(f"  3. 速度/角度圖表：{os.path.basename(plot_paths['speed_orientation'])}")
    else:
        print(f"  3. 角速度圖表：{os.path.basename(plot_paths['angular_speed'])}")
    print(f"  4. CSV 資料檔：{os.path.basename(csv_path)}")
    
def generate_plots(df, output_dir, OUT_PREFIX):
    """生成追蹤結果圖表"""
    
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
    ax_pos.set_xlabel("x (mm)", fontsize=24)
    ax_pos.set_ylabel("y (mm)", fontsize=24)
    ax_pos.set_title("Position (Trajectory)", fontsize=22)
    #ax_pos.grid(True, linestyle="--", alpha=0.4)
    ax_pos.legend(loc="best", fontsize=14)
    
    plot_pos_path = os.path.join(output_dir, f"{OUT_PREFIX}_position.png")
    figA.savefig(plot_pos_path, dpi=1200, bbox_inches='tight', pad_inches=0.3)
    plt.close(figA)
    plot_paths['position'] = plot_pos_path
    
    # B) 速度和角度圖
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
        ax_s.set_xlabel("Time (s)", fontsize=24)
        ax_s.set_ylabel("Speed (mm/s)", fontsize=24)
        ax_s.set_title("Speed vs Time", fontsize=22)
        #ax_s.grid(True, linestyle="--", alpha=0.4)
        ax_s.legend(loc="best", fontsize=14)
        
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
        ax_a.set_xlabel("Time (s)", fontsize=24)
        ax_a.set_ylabel("Angle (deg)", fontsize=24)
        ax_a.set_title("Orientation vs Time", fontsize=22)
        if ORIENT_YLIM_DEG is not None and np.isfinite(ORIENT_YLIM_DEG):
            ylim = float(ORIENT_YLIM_DEG)
            ax_a.set_ylim(-ylim, +ylim)
        #ax_a.grid(True, linestyle="--", alpha=0.4)
        ax_a.legend(loc="best", fontsize=14)
        
        plot_so_path = os.path.join(output_dir, f"{OUT_PREFIX}_speed_orientation.png")
        figB.savefig(plot_so_path, dpi=1200, bbox_inches='tight', pad_inches=0.3)
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
        
        ax_w.set_xlabel("Time (s)", fontsize=24)
        ax_w.set_ylabel("Angular speed (deg/s)", fontsize=24)
        ax_w.set_title("Angular Speed vs Time", fontsize=22)
        #ax_w.grid(True, linestyle="--", alpha=0.4)
        ax_w.legend(loc="best", fontsize=14)
        
        plot_w_path = os.path.join(output_dir, f"{OUT_PREFIX}_angular_speed.png")
        figW.savefig(plot_w_path, dpi=1200, bbox_inches='tight', pad_inches=0.3)
        plt.close(figW)
        plot_paths['angular_speed'] = plot_w_path
    
    return plot_paths

if __name__ == "__main__":
    main()
