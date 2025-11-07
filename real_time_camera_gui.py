# -*- coding: utf-8 -*-
"""
即時攝影機GUI版：加入圖形介面與ROI選取功能
------------------------------------------------
新增功能：
1) Tkinter GUI介面，包含開始/暫停/停止按鈕
2) ROI (Region of Interest) 選取功能，可以選擇偵測範圍
3) 保留原有的物體追蹤和角度分析功能
4) 鍵盤快捷鍵：空白鍵暫停/繼續，ESC鍵停止
5) 只顯示選取的ROI區域

使用方法：
- 執行程式後會開啟GUI介面
- 點擊「選擇偵測區域」來設定ROI
- 點擊「開始偵測」開始即時偵測
- 可隨時暫停/繼續或停止偵測
- 鍵盤快捷鍵：空白鍵=暫停/繼續，ESC鍵=停止
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time

# ========= 使用者設定 =========
CAMERA_INDEX    = 1            # 攝影機索引
CAM_WIDTH       = 1280         # 取樣解析度
CAM_HEIGHT      = 720
CAM_FPS_REQ     = 12           # 目標幀率
RECORD_OUTPUT   = True         # 是否錄製輸出
GRID_SPACING_MM = 5.0          # 網格間距（mm）

# 色彩遮罩設定
HSV_YELLOW_LO = np.array([15, 60, 120], dtype=np.uint8)
HSV_YELLOW_HI = np.array([35, 255, 255], dtype=np.uint8)
HSV_WHITE_LO  = np.array([0,  0, 200], dtype=np.uint8)
HSV_WHITE_HI  = np.array([180, 60, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 50
PROCESS_EVERY_N  = 1
SMOOTH_WIN_ANGLE = 5

# 比例設定
AUTO_GRID_MM_PER_PX = True
MANUAL_MM_PER_PX    = None

# 固定紅框尺寸（mm）
BOX_W_MM = 9.0
BOX_H_MM = 6.0

# 視覺化參數
PLOT_RANGE_SCALE    = 1.35
INVERT_Y_AXIS       = True

class ROISelector:
    """ROI選取器類別"""
    def __init__(self):
        self.roi = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.temp_roi = None

    def mouse_callback(self, event, x, y, flags, param):
        frame = param['frame']
        display_frame = param['display_frame']
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_frame = frame.copy()
                cv2.rectangle(temp_frame, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("Select Detection Region", temp_frame)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            # 確保ROI座標正確
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            self.roi = (x1, y1, x2-x1, y2-y1)  # (x, y, width, height)
            
            # 顯示最終選擇
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(temp_frame, "Press ENTER to confirm, ESC to cancel", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("選擇偵測區域", temp_frame)

    def select_roi_interactive(self, frame):
        """Interactive ROI selection"""
        self.roi = None
        clone = frame.copy()
        
        cv2.namedWindow("Select Detection Region", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Detection Region", 800, 600)
        cv2.imshow("Select Detection Region", clone)
        
        # Set mouse callback
        param = {'frame': clone, 'display_frame': None}
        cv2.setMouseCallback("Select Detection Region", self.mouse_callback, param)
        
        cv2.putText(clone, "Drag to select ROI, ENTER to confirm, ESC to cancel", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Select Detection Region", clone)
        
        while True:
            key = cv2.waitKey(30) & 0xFF  # Increased wait time for better key detection
            if key == 13:  # Enter key
                break  # Accept ROI selection (even if None)
            elif key == 27:  # ESC key
                self.roi = None
                break
            elif key == ord('q'):  # Also allow 'q' to quit
                self.roi = None
                break
                
        cv2.destroyWindow("Select Detection Region")
        cv2.waitKey(1)  # Ensure window is properly closed
        return self.roi

class CameraDetectionGUI:
    """攝影機偵測GUI主類別"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Camera Object Detection System")
        self.root.geometry("400x520")
        
        # 狀態變數
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.roi = None
        self.detection_thread = None
        self.mm_per_px = 0.1
        
        # 資料記錄
        self.rec_data = []
        self.frame_idx = 0
        self.writer = None
        self.kf = None
        
        # 輸出目錄
        self.output_dir = "camera_live_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.setup_gui()
        
    def setup_gui(self):
        """設置GUI介面"""
        # 標題
        title_frame = ttk.Frame(self.root, padding="10")
        title_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(title_frame, text="Real-time Camera Object Detection System", 
                               font=("Arial", 16, "bold"))
        title_label.pack()
        
        # 狀態顯示
        status_frame = ttk.LabelFrame(self.root, text="System Status", padding="10")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="System Ready", 
                                     font=("Arial", 10))
        self.status_label.pack()
        
        self.roi_status_label = ttk.Label(status_frame, text="No Detection Region Selected", 
                                         font=("Arial", 9), foreground="red")
        self.roi_status_label.pack()
        
        # 控制按鈕
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding="10")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # ROI選擇按鈕
        self.roi_button = ttk.Button(control_frame, text="Select Detection Region", 
                                    command=self.select_roi, width=20)
        self.roi_button.pack(pady=5)
        
        # 開始按鈕
        self.start_button = ttk.Button(control_frame, text="Start Detection", 
                                      command=self.start_detection, width=20)
        self.start_button.pack(pady=5)
        self.start_button.config(state="disabled")
        
        # 暫停按鈕
        self.pause_button = ttk.Button(control_frame, text="Pause", 
                                      command=self.pause_detection, width=20)
        self.pause_button.pack(pady=5)
        self.pause_button.config(state="disabled")
        
        # 停止按鈕
        self.stop_button = ttk.Button(control_frame, text="Stop Detection", 
                                     command=self.stop_detection, width=20)
        self.stop_button.pack(pady=5)
        self.stop_button.config(state="disabled")
        
        # 快捷鍵說明
        shortcut_frame = ttk.LabelFrame(self.root, text="Keyboard Shortcuts", padding="10")
        shortcut_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(shortcut_frame, text="SPACE: Pause/Resume Detection", 
                 font=("Arial", 9)).pack(anchor=tk.W)
        ttk.Label(shortcut_frame, text="ESC: Stop Detection", 
                 font=("Arial", 9)).pack(anchor=tk.W)
        
        # 設定框架
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 攝影機索引設定
        cam_frame = ttk.Frame(settings_frame)
        cam_frame.pack(fill=tk.X, pady=2)
        ttk.Label(cam_frame, text="Camera Index:").pack(side=tk.LEFT)
        self.cam_index_var = tk.StringVar(value=str(CAMERA_INDEX))
        ttk.Entry(cam_frame, textvariable=self.cam_index_var, width=5).pack(side=tk.RIGHT)
        
        # 錄製設定
        self.record_var = tk.BooleanVar(value=RECORD_OUTPUT)
        ttk.Checkbutton(settings_frame, text="Record Detection Video", 
                       variable=self.record_var).pack(anchor=tk.W, pady=2)
        
        # 資訊顯示
        info_frame = ttk.LabelFrame(self.root, text="Detection Information", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=6, width=50)
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 退出按鈕
        exit_frame = ttk.Frame(self.root, padding="10")
        exit_frame.pack(fill=tk.X)
        
        ttk.Button(exit_frame, text="Exit Program", command=self.on_closing).pack()
        
        # 綁定關閉事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def log_info(self, message):
        """記錄資訊到GUI"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.info_text.insert(tk.END, log_message)
        self.info_text.see(tk.END)
        self.root.update_idletasks()
        
    def select_roi(self):
        """選擇ROI區域"""
        try:
            # 暫時開啟攝影機來選取ROI
            cam_index = int(self.cam_index_var.get())
            temp_cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            
            # Quick warmup - reduced from 5 to 2 frames for faster startup
            for _ in range(2):
                temp_cap.read()
                
            ret, frame = temp_cap.read()
            if not ret:
                messagebox.showerror("Error", "Cannot open camera, please check camera index settings")
                temp_cap.release()
                return
                
            self.log_info("Starting ROI selection...")
            
            # 使用ROI選取器
            roi_selector = ROISelector()
            self.roi = roi_selector.select_roi_interactive(frame)
            
            temp_cap.release()
            
            if self.roi is not None:
                x, y, w, h = self.roi
                self.roi_status_label.config(text=f"已選擇區域: ({x},{y}) 尺寸:{w}x{h}", 
                                           foreground="green")
                self.start_button.config(state="normal")
                self.log_info(f"ROI設置完成: x={x}, y={y}, width={w}, height={h}")
            else:
                self.roi_status_label.config(text="未選擇偵測區域", foreground="red")
                self.start_button.config(state="disabled")
                self.log_info("ROI選擇已取消")
                
        except Exception as e:
            messagebox.showerror("錯誤", f"ROI選擇失敗: {str(e)}")
            self.log_info(f"ROI選擇錯誤: {str(e)}")
            
    def start_detection(self):
        """開始偵測"""
        if self.roi is None:
            messagebox.showwarning("警告", "請先選擇偵測區域")
            return
            
        try:
            self.log_info("正在初始化攝影機...")
            
            # 開啟攝影機
            cam_index = int(self.cam_index_var.get())
            self.log_info(f"嘗試開啟攝影機索引: {cam_index}")
            
            # 嘗試不同的後端
            backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            self.cap = None
            
            for backend in backends_to_try:
                self.log_info(f"嘗試後端: {backend}")
                temp_cap = cv2.VideoCapture(cam_index, backend)
                if temp_cap.isOpened():
                    self.cap = temp_cap
                    self.log_info(f"成功使用後端: {backend}")
                    break
                else:
                    temp_cap.release()
                    self.log_info(f"後端 {backend} 失敗")
            
            if self.cap is None or not self.cap.isOpened():
                self.log_info("所有後端都失敗，嘗試不指定後端")
                self.cap = cv2.VideoCapture(cam_index)
                
            if self.cap is None or not self.cap.isOpened():
                error_msg = f"無法開啟攝影機索引 {cam_index}。請檢查：\n1. 攝影機是否已連接\n2. 索引是否正確\n3. 攝影機是否被其他程式占用"
                messagebox.showerror("攝影機錯誤", error_msg)
                self.log_info("攝影機開啟失敗")
                return
                
            self.log_info("攝影機已開啟，設定參數...")
            
            # 設定攝影機參數
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS_REQ)
            
            # 檢查實際參數
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.log_info(f"攝影機參數: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Quick warmup - reduced for faster startup
            self.log_info("Camera warming up...")
            for i in range(2):
                ret, frame = self.cap.read()
                if ret and i == 1:  # Only log the last successful frame
                    self.log_info(f"Frame size: {frame.shape}")
                    
            ret, first_frame = self.cap.read()
            if not ret or first_frame is None:
                error_msg = "攝影機已開啟但無法讀取影像。可能原因：\n1. 攝影機被其他程式占用\n2. 攝影機驅動問題\n3. USB連接不穩定"
                messagebox.showerror("影像讀取錯誤", error_msg)
                self.log_info("無法讀取攝影機影像")
                if self.cap:
                    self.cap.release()
                return
                
            self.log_info(f"成功讀取第一幀，尺寸: {first_frame.shape}")
            
            # 檢查ROI是否在影像範圍內
            x, y, w, h = self.roi
            if y + h > first_frame.shape[0] or x + w > first_frame.shape[1]:
                error_msg = f"ROI區域 ({x},{y},{w},{h}) 超出影像範圍 {first_frame.shape[:2]}"
                messagebox.showerror("ROI錯誤", error_msg)
                self.log_info(error_msg)
                if self.cap:
                    self.cap.release()
                return
            
            self.log_info("ROI區域有效")
                
            # 估算mm/px比例
            self.log_info("估算 mm/px 比例...")
            self.mm_per_px = self.estimate_mm_per_px_single_frame(first_frame)
            if self.mm_per_px is None:
                self.mm_per_px = 0.1
                self.log_info("使用預設 mm/px 比例")
            else:
                self.log_info(f"自動估算 mm/px 比例: {self.mm_per_px:.6f}")
            
            # 初始化Kalman濾波器
            self.log_info("初始化Kalman濾波器...")
            self.kf = self.make_kalman()
            
            # ROI中心作為初始位置
            init_x, init_y = x + w/2, y + h/2
            self.kf.statePost = np.array([[init_x], [init_y], [0.0], [0.0]], dtype=np.float32)
            self.log_info(f"Kalman初始位置: ({init_x:.1f}, {init_y:.1f})")
            
            # 初始化錄製（錄製ROI區域）
            self.writer = None
            if self.record_var.get():
                self.log_info("初始化影片錄製...")
                fps = actual_fps if actual_fps > 0 else CAM_FPS_REQ
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_path = os.path.join(self.output_dir, "camera_live_roi_tracked.mp4")
                # 錄製ROI區域的大小
                roi_w, roi_h = w, h
                self.writer = cv2.VideoWriter(out_path, fourcc, max(5.0, fps/PROCESS_EVERY_N), (roi_w, roi_h))
                if self.writer.isOpened():
                    self.log_info(f"影片錄製已啟用: {out_path} (ROI尺寸: {roi_w}x{roi_h})")
                else:
                    self.log_info("影片錄製初始化失敗，將跳過錄製")
                    self.writer = None
                
            # 重置資料
            self.rec_data = []
            self.frame_idx = 0
            self.is_running = True
            self.is_paused = False
            
            # 更新GUI狀態
            self.status_label.config(text="偵測進行中...")
            self.start_button.config(state="disabled")
            self.pause_button.config(state="normal")
            self.stop_button.config(state="normal")
            self.roi_button.config(state="disabled")
            
            # 啟動偵測執行緒
            self.log_info("啟動偵測執行緒...")
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            self.log_info("偵測已開始！使用空白鍵暫停/繼續，ESC鍵停止")
            
        except Exception as e:
            error_msg = f"啟動偵測失敗: {str(e)}"
            messagebox.showerror("錯誤", error_msg)
            self.log_info(error_msg)
            # 確保清理資源
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None
            
    def pause_detection(self):
        """Pause/Resume Detection"""
        if self.is_paused:
            self.is_paused = False
            self.pause_button.config(text="Pause")
            self.status_label.config(text="Detection Running...")
            self.log_info("Detection resumed")
        else:
            self.is_paused = True
            self.pause_button.config(text="Resume")
            self.status_label.config(text="Detection Paused")
            self.log_info("Detection paused")
            
    def stop_detection(self):
        """Stop detection - improved to prevent freezing"""
        if not self.is_running:
            return
            
        self.log_info("Stopping detection...")
        self.is_running = False
        self.is_paused = False
        
        # Update GUI status immediately
        self.status_label.config(text="Stopping detection...")
        
        # Disable all control buttons immediately
        self.start_button.config(state="disabled")
        self.pause_button.config(state="disabled") 
        self.stop_button.config(state="disabled")
        self.roi_button.config(state="disabled")
        
        # Force close OpenCV windows immediately to prevent hanging
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Allow OpenCV to process the destroy command
        except:
            pass
            
        # Use a shorter delay for faster response
        self.root.after(50, self._finish_stop_detection)
        
    def _finish_stop_detection(self):
        """Complete the stop detection process - improved to prevent hanging"""
        try:
            # Ensure OpenCV windows are closed
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
            # Clean up resources without waiting for thread
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
                
            if self.writer:
                try:
                    self.writer.release()
                except:
                    pass
                self.writer = None
            
            # Save data if available
            if len(self.rec_data) > 0:
                self.log_info("Saving results...")
                self.root.update_idletasks()
                try:
                    self.save_results()
                    self.log_info("Results saved successfully")
                except Exception as e:
                    self.log_info(f"Error saving results: {str(e)}")
            
            # Reset GUI state
            self.status_label.config(text="System Ready")
            self.start_button.config(state="normal" if self.roi is not None else "disabled")
            self.roi_button.config(state="normal")
            self.pause_button.config(text="Pause")
            
            self.log_info("Detection stopped successfully")
            
        except Exception as e:
            self.log_info(f"Error during stop: {str(e)}")
            # Force reset GUI state even if cleanup fails
            self.status_label.config(text="System Ready")
            self.start_button.config(state="normal" if self.roi is not None else "disabled")
            self.roi_button.config(state="normal") 
            self.pause_button.config(text="Pause")
        
    def detection_loop(self):
        """偵測主迴圈"""
        try:
            self.log_info("偵測迴圈已啟動")
            fps = self.cap.get(cv2.CAP_PROP_FPS) or CAM_FPS_REQ
            self.log_info(f"使用FPS: {fps}")
            
            frame_count = 0
            
            while self.is_running:
                try:
                    # 檢查暫停狀態
                    if self.is_paused:
                        time.sleep(0.1)
                        continue
                    
                    # 檢查攝影機是否還有效
                    if not self.cap or not self.cap.isOpened():
                        self.log_info("攝影機已斷線或關閉")
                        self.is_running = False
                        break
                        
                    ret, frame = self.cap.read()
                    if not ret:
                        self.log_info("無法讀取攝影機幀")
                        self.is_running = False
                        break
                    
                    frame_count += 1
                    if frame_count <= 3:  # 只記錄前幾幀
                        self.log_info(f"成功讀取第 {frame_count} 幀，尺寸: {frame.shape}")
                        
                    if self.frame_idx % PROCESS_EVERY_N != 0:
                        self.frame_idx += 1
                        continue
                    
                    # 檢查ROI是否還有效
                    if not self.roi:
                        self.is_running = False
                        break
                        
                    # 在ROI區域內進行偵測
                    x, y, w, h = self.roi
                    if y+h > frame.shape[0] or x+w > frame.shape[1]:
                        self.is_running = False
                        break
                        
                    roi_frame = frame[y:y+h, x:x+w]
                    
                    # Kalman預測
                    if self.kf:
                        pred = self.kf.predict()
                    
                    # 在ROI內尋找目標
                    meas = self.find_target_and_angle(roi_frame)
                    
                    angle_deg = np.nan
                    if meas is not None:
                        cx, cy, bx, by, bw, bh, angle_deg, cnt = meas
                        # 將ROI座標轉換回全圖座標
                        cx_global = cx + x
                        cy_global = cy + y
                        
                        # Kalman更新
                        if self.kf:
                            est = self.kf.correct(np.array([[cx_global],[cy_global]], dtype=np.float32))
                            fx, fy = float(est[0,0]), float(est[1,0])
                        else:
                            fx, fy = cx_global, cy_global
                    else:
                        fx = fy = np.nan
                    
                    # 準備顯示ROI區域（只顯示選取的範圍）
                    display_frame = roi_frame.copy()
                    
                    # 在ROI框內顯示偵測結果
                    if meas is not None:
                        cx, cy, bx, by, bw, bh, angle_deg, cnt = meas
                        # 繪製輪廓（座標已經是ROI內的）
                        cv2.drawContours(display_frame, [cnt], -1, (0, 0, 255), 3)
                        # 繪製中心點（座標已經是ROI內的）
                        cv2.circle(display_frame, (int(round(cx)), int(round(cy))), 6, (0,255,0), -1)
                    
                    # 顯示資訊（調整位置適合ROI大小）
                    fx_mm = fx * self.mm_per_px if not np.isnan(fx) else np.nan
                    fy_mm = fy * self.mm_per_px if not np.isnan(fy) else np.nan
                    cv2.putText(display_frame, f"Pos(mm): ({fx_mm:.2f}, {fy_mm:.2f})", 
                               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    if not np.isnan(angle_deg):
                        cv2.putText(display_frame, f"Angle: {angle_deg:+.2f} deg", 
                                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    else:
                        cv2.putText(display_frame, f"Angle: NaN", 
                                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    
                    # 顯示控制說明
                    cv2.putText(display_frame, "SPACE: Start/Pause, ESC: Stop", 
                               (10, display_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    
                    # 記錄資料
                    t_s = self.frame_idx / fps
                    self.rec_data.append((self.frame_idx, t_s, fx, fy, fx_mm, fy_mm, angle_deg, self.mm_per_px))
                    
                    # 錄製影片（錄製ROI區域）
                    if self.writer is not None and self.writer.isOpened():
                        self.writer.write(display_frame)
                        
                    # 顯示影像（只有在運行時才顯示，且只顯示ROI區域）
                    if self.is_running:
                        cv2.imshow("Real-time Detection - ROI Region", display_frame)
                        key = cv2.waitKey(10) & 0xFF  # Increased wait time for better key detection
                        
                        # 鍵盤控制
                        if key == 32:  # 空白鍵 - 暫停/繼續
                            self.root.after(0, self.pause_detection)
                        elif key == 27:  # ESC鍵 - 停止
                            self.log_info("ESC key pressed, stopping detection...")
                            self.is_running = False
                            # Immediately trigger stop detection in GUI thread
                            self.root.after(0, self.stop_detection)
                            break
                        elif key == ord('q'):  # 保留q鍵停止功能
                            self.log_info("Q key pressed, stopping detection...")
                            self.is_running = False
                            self.root.after(0, self.stop_detection)
                            break
                            
                    self.frame_idx += 1
                    
                except Exception as e:
                    self.log_info(f"偵測迴圈錯誤: {str(e)}")
                    self.is_running = False
                    break
                    
        except Exception as e:
            self.log_info(f"偵測迴圈嚴重錯誤: {str(e)}")
            self.is_running = False
            
        finally:
            # 確保清理資源
            try:
                cv2.destroyAllWindows()
                if self.cap:
                    self.cap.release()
                if self.writer and self.writer.isOpened():
                    self.writer.release()
            except:
                pass
                
        # 通知GUI停止偵測完成
        if self.root:
            self.root.after(0, self._finish_stop_detection)
        
    def save_results(self):
        """儲存偵測結果"""
        try:
            # 建立DataFrame
            df = pd.DataFrame(self.rec_data, columns=[
                "frame","t_s","x_px_filt","y_px_filt",
                "x_mm","y_mm","angle_deg_raw","mm_per_px"
            ])
            
            # 角度展開與平滑
            angle_series = df["angle_deg_raw"].to_numpy()
            mask_valid = ~np.isnan(angle_series)
            angle_unwrapped = np.full_like(angle_series, np.nan, dtype=float)
            if mask_valid.any():
                angle_unwrapped[mask_valid] = self.unwrap_angles_deg(angle_series[mask_valid])
                angle_unwrapped = self.moving_average(angle_unwrapped, SMOOTH_WIN_ANGLE)
            df["angle_deg_unwrapped"] = angle_unwrapped
            
            # 角速度計算
            dt = np.gradient(df["t_s"].to_numpy())
            ang_vel = np.full_like(angle_unwrapped, np.nan, dtype=float)
            valid_idx = np.where(mask_valid)[0]
            if len(valid_idx) >= 2:
                for i0, i1 in zip(valid_idx[:-1], valid_idx[1:]):
                    dtheta = angle_unwrapped[i1] - angle_unwrapped[i0]
                    dt_seg = df.loc[i1, "t_s"] - df.loc[i0, "t_s"]
                    if dt_seg > 0:
                        ang_vel[i1] = dtheta / dt_seg
                ang_vel = self.moving_average(ang_vel, SMOOTH_WIN_ANGLE)
            df["angular_vel_dps"] = ang_vel
            
            # 儲存CSV
            csv_path = os.path.join(self.output_dir, "camera_live_pos_angle.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            
            # 產生圖表
            self.create_plots(df)
            
            self.log_info(f"結果已儲存至: {self.output_dir}")
            self.log_info(f"共處理 {len(df)} 幀資料")
            
        except Exception as e:
            self.log_info(f"儲存結果時發生錯誤: {str(e)}")
            
    def create_plots(self, df):
        """產生分析圖表"""
        try:
            fig, (ax_pos, ax_w) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
            
            # 位置軌跡
            ax_pos.plot(df["x_mm"], df["y_mm"], lw=2, label="Trajectory", color='blue')
            valid_pos = df[["x_mm", "y_mm"]].dropna()
            if len(valid_pos) > 0:
                x_start, y_start = valid_pos.iloc[0]["x_mm"], valid_pos.iloc[0]["y_mm"]
                x_end, y_end = valid_pos.iloc[-1]["x_mm"], valid_pos.iloc[-1]["y_mm"]
                ax_pos.scatter([x_start], [y_start], s=100, c="green", marker="o", label="Start", zorder=5)
                ax_pos.scatter([x_end], [y_end], s=100, c="red", marker="o", label="End", zorder=5)
                
            ax_pos.set_aspect("equal", adjustable="box")
            if INVERT_Y_AXIS:
                ax_pos.invert_yaxis()
            ax_pos.set_xlabel("x (mm)")
            ax_pos.set_ylabel("y (mm)")
            ax_pos.set_title("Position (Trajectory)")
            ax_pos.grid(True, linestyle="--", alpha=0.4)
            ax_pos.legend(loc="best")
            
            # 角速度
            ax_w.plot(df["t_s"], df["angular_vel_dps"], lw=2, color='red')
            ax_w.set_xlabel("Time (s)")
            ax_w.set_ylabel("Angular Velocity (deg/s)")
            ax_w.set_title("Angular Velocity vs Time")
            ax_w.grid(True, linestyle="--", alpha=0.4)
            
            # 儲存圖片
            plot_path = os.path.join(self.output_dir, "camera_live_pos_angvel_subplot.png")
            fig.savefig(plot_path, dpi=220)
            plt.close(fig)
            
        except Exception as e:
            self.log_info(f"產生圖表時發生錯誤: {str(e)}")
            
    def on_closing(self):
        """關閉視窗時的處理"""
        if self.is_running:
            self.stop_detection()
        self.root.destroy()
        
    # ========= 輔助函數 =========
    def estimate_mm_per_px_single_frame(self, frame):
        """估算mm/px比例"""
        if not AUTO_GRID_MM_PER_PX:
            return MANUAL_MM_PER_PX
            
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

        return (GRID_SPACING_MM / px_per_cell) if (px_per_cell and px_per_cell > 0) else None

    def unwrap_angles_deg(self, angles_deg):
        """將角度序列（度）做展開，避免跨 ±180 度造成跳變"""
        if len(angles_deg) == 0:
            return angles_deg
        rad = np.deg2rad(angles_deg)
        rad_unwrap = np.unwrap(rad)
        return np.rad2deg(rad_unwrap)

    def moving_average(self, a, w):
        """移動平均"""
        if w is None or w <= 1:
            return a
        a = np.asarray(a, dtype=float)
        if len(a) < w:
            return a
        kernel = np.ones(w) / float(w)
        return np.convolve(a, kernel, mode='same')

    def make_kalman(self):
        """建立Kalman濾波器"""
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

    def find_target_and_angle(self, frame_bgr):
        """尋找目標並計算角度"""
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

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), rect_angle = rect

        if rw >= rh:
            angle_deg = rect_angle + 90.0
        else:
            angle_deg = rect_angle

        while angle_deg >= 90.0:
            angle_deg -= 180.0
        while angle_deg < -90.0:
            angle_deg += 180.0

        x, y, w, h = cv2.boundingRect(cnt)
        return (cx, cy, x, y, w, h, angle_deg, cnt)


def main():
    """主函數"""
    root = tk.Tk()
    app = CameraDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
