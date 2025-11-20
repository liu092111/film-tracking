# -*- coding: utf-8 -*-
"""
多執行緒即時相機追蹤
解決單執行緒丟幀問題
----------------------------------------------------------------------
架構：
- Thread 1: 專門快速讀取相機幀（不做處理）
- Thread 2: 從佇列取出幀並處理
- 這樣可以盡可能不丟失相機的幀
"""

import os
import cv2
import time
import numpy as np
import threading
import queue
from datetime import datetime
from collections import deque

# ========== 設定 ==========
CAMERA_INDEX      = 1
CAM_WIDTH         = 1280
CAM_HEIGHT        = 720
CAM_FPS_REQ       = 120

# 顏色遮罩（黃 + 白）
HSV_YELLOW_LO = np.array([15, 60, 120], dtype=np.uint8)
HSV_YELLOW_HI = np.array([35, 255, 255], dtype=np.uint8)
HSV_WHITE_LO  = np.array([0,  0, 200], dtype=np.uint8)
HSV_WHITE_HI  = np.array([180, 60, 255], dtype=np.uint8)
MIN_CONTOUR_AREA = 50

# 佇列設定
FRAME_QUEUE_SIZE = 120  # 可以緩存 1 秒的幀（120 FPS）
PROCESS_QUEUE_SIZE = 10  # 處理結果佇列

# 統計資訊
class Stats:
    def __init__(self):
        self.frames_captured = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.capture_fps = 0.0
        self.process_fps = 0.0
        self.lock = threading.Lock()
    
    def update_capture(self, fps):
        with self.lock:
            self.frames_captured += 1
            self.capture_fps = fps
    
    def update_process(self, fps):
        with self.lock:
            self.frames_processed += 1
            self.process_fps = fps
    
    def dropped_frame(self):
        with self.lock:
            self.frames_dropped += 1
    
    def get_info(self):
        with self.lock:
            return {
                'captured': self.frames_captured,
                'processed': self.frames_processed,
                'dropped': self.frames_dropped,
                'capture_fps': self.capture_fps,
                'process_fps': self.process_fps,
                'queue_utilization': 0.0  # 會從外部更新
            }

stats = Stats()

# ========== 影像處理函數 ==========
def process_frame(frame):
    """完整的影像處理流程"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_y = cv2.inRange(hsv, HSV_YELLOW_LO, HSV_YELLOW_HI)
    mask_w = cv2.inRange(hsv, HSV_WHITE_LO, HSV_WHITE_HI)
    mask = cv2.bitwise_or(mask_y, mask_w)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = {
        'found': False,
        'center': None,
        'angle': None,
        'box': None
    }
    
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), rect_angle = rect
            box = cv2.boxPoints(rect).astype(int)
            
            result['found'] = True
            result['center'] = (cx, cy)
            result['angle'] = rect_angle
            result['box'] = box
    
    return result

# ========== Thread 1: 相機讀取 ==========
def capture_thread(cap, frame_queue, running):
    """
    專門負責快速讀取相機幀
    目標：盡可能快速讀取，不丟幀
    """
    print("[Capture Thread] 啟動")
    
    frame_count = 0
    start_time = time.perf_counter()
    fps_update_interval = 30  # 每 30 幀更新一次 FPS
    
    while running[0]:
        ret, frame = cap.read()
        if not ret:
            print("[Capture Thread] 讀取失敗")
            break
        
        frame_count += 1
        timestamp = time.perf_counter()
        
        # 嘗試放入佇列
        try:
            frame_queue.put_nowait((frame.copy(), timestamp, frame_count))
        except queue.Full:
            # 佇列滿了，丟棄此幀
            stats.dropped_frame()
        
        # 更新 FPS
        if frame_count % fps_update_interval == 0:
            elapsed = timestamp - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                stats.update_capture(fps)
    
    print(f"[Capture Thread] 結束 - 總共讀取 {frame_count} 幀")

# ========== Thread 2: 影像處理 ==========
def process_thread(frame_queue, result_queue, running):
    """
    從佇列取出幀並處理
    """
    print("[Process Thread] 啟動")
    
    frame_count = 0
    start_time = time.perf_counter()
    process_times = deque(maxlen=100)  # 保留最近 100 幀的處理時間
    
    while running[0] or not frame_queue.empty():
        try:
            # 從佇列取出幀（timeout 避免卡住）
            frame, timestamp, frame_id = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        frame_count += 1
        
        # 處理幀
        process_start = time.perf_counter()
        result = process_frame(frame)
        process_end = time.perf_counter()
        
        process_time = (process_end - process_start) * 1000  # ms
        process_times.append(process_time)
        
        # 嘗試放入結果佇列
        try:
            result_queue.put_nowait({
                'frame': frame,
                'result': result,
                'timestamp': timestamp,
                'frame_id': frame_id,
                'process_time': process_time
            })
        except queue.Full:
            # 結果佇列滿了，丟棄舊的結果
            try:
                result_queue.get_nowait()
                result_queue.put_nowait({
                    'frame': frame,
                    'result': result,
                    'timestamp': timestamp,
                    'frame_id': frame_id,
                    'process_time': process_time
                })
            except:
                pass
        
        # 更新統計
        elapsed = process_end - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            stats.update_process(fps)
    
    avg_process_time = np.mean(list(process_times)) if process_times else 0
    print(f"[Process Thread] 結束 - 總共處理 {frame_count} 幀")
    print(f"[Process Thread] 平均處理時間: {avg_process_time:.2f} ms")

# ========== 主程式 ==========
def main():
    print("="*70)
    print("多執行緒即時相機追蹤")
    print("="*70)
    
    # 初始化相機
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
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
    
    # 建立佇列
    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    result_queue = queue.Queue(maxsize=PROCESS_QUEUE_SIZE)
    
    # 執行控制
    running = [True]  # 使用 list 讓 threads 可以共享
    
    # 啟動執行緒
    capture_t = threading.Thread(target=capture_thread, args=(cap, frame_queue, running), daemon=True)
    process_t = threading.Thread(target=process_thread, args=(frame_queue, result_queue, running), daemon=True)
    
    capture_t.start()
    process_t.start()
    
    print("\n執行緒已啟動")
    print("按 [q] 或 [ESC] 結束\n")
    
    # 顯示視窗
    cv2.namedWindow("Multi-threaded Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-threaded Tracking", CAM_WIDTH//2, CAM_HEIGHT//2)
    
    # 主迴圈：負責顯示
    last_display_time = time.perf_counter()
    display_interval = 1.0 / 30.0  # 30 FPS 顯示（降低顯示負擔）
    
    try:
        while running[0]:
            current_time = time.perf_counter()
            
            # 控制顯示幀率
            if current_time - last_display_time < display_interval:
                time.sleep(0.001)
                continue
            
            try:
                # 取得最新的處理結果
                data = result_queue.get_nowait()
                frame = data['frame']
                result = data['result']
                process_time = data['process_time']
                frame_id = data['frame_id']
                
                # 繪製結果
                if result['found']:
                    box = result['box']
                    center = result['center']
                    cv2.polylines(frame, [box], isClosed=True, color=(0,0,255), thickness=3)
                    cv2.circle(frame, (int(center[0]), int(center[1])), 6, (0,255,0), -1)
                
                # 顯示統計資訊
                info = stats.get_info()
                queue_util = (frame_queue.qsize() / FRAME_QUEUE_SIZE) * 100
                
                y_pos = 30
                line_height = 30
                cv2.putText(frame, f"Capture FPS: {info['capture_fps']:.1f}", 
                           (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                y_pos += line_height
                
                cv2.putText(frame, f"Process FPS: {info['process_fps']:.1f}", 
                           (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                y_pos += line_height
                
                cv2.putText(frame, f"Process Time: {process_time:.1f} ms", 
                           (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                y_pos += line_height
                
                cv2.putText(frame, f"Frames: {info['captured']} cap / {info['processed']} proc", 
                           (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                y_pos += line_height
                
                cv2.putText(frame, f"Dropped: {info['dropped']}", 
                           (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                y_pos += line_height
                
                cv2.putText(frame, f"Queue: {frame_queue.qsize()}/{FRAME_QUEUE_SIZE} ({queue_util:.0f}%)", 
                           (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                cv2.imshow("Multi-threaded Tracking", frame)
                last_display_time = current_time
                
            except queue.Empty:
                pass
            
            # 按鍵檢查
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                print("\n停止中...")
                running[0] = False
                break
    
    except KeyboardInterrupt:
        print("\n接收到中斷信號...")
        running[0] = False
    
    # 等待執行緒結束
    print("等待執行緒結束...")
    capture_t.join(timeout=2.0)
    process_t.join(timeout=2.0)
    
    # 清理
    cap.release()
    cv2.destroyAllWindows()
    
    # 最終統計
    info = stats.get_info()
    print("\n"+"="*70)
    print("最終統計")
    print("="*70)
    print(f"讀取幀數: {info['captured']}")
    print(f"處理幀數: {info['processed']}")
    print(f"丟棄幀數: {info['dropped']}")
    print(f"讀取 FPS: {info['capture_fps']:.2f}")
    print(f"處理 FPS: {info['process_fps']:.2f}")
    print(f"處理效率: {(info['processed']/info['captured']*100):.1f}%" if info['captured'] > 0 else "N/A")
    print("="*70)

if __name__ == "__main__":
    main()
