# -*- coding: utf-8 -*-
"""
測試不同解析度的實際 FPS
用於診斷 USB 傳輸瓶頸
"""

import cv2
import time

CAMERA_INDEX = 1
TEST_DURATION = 5.0

# 測試不同的解析度和 FPS 組合
test_configs = [
    # (width, height, fps_target, name)
    (640, 480, 120, "VGA @ 120"),
    (640, 480, 60, "VGA @ 60"),
    (800, 600, 120, "SVGA @ 120"),
    (800, 600, 60, "SVGA @ 60"),
    (1280, 720, 120, "HD @ 120"),
    (1280, 720, 60, "HD @ 60"),
    (1280, 720, 30, "HD @ 30"),
]

print("="*70)
print("解析度 FPS 測試")
print("目的：找出最佳的解析度和 FPS 組合")
print("="*70)

results = []

for width, height, fps_target, name in test_configs:
    print(f"\n測試: {name} ({width}x{height} @ {fps_target} FPS)")
    print("-" * 70)
    
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps_target)
        
        # 確認設定
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"相機回報: {actual_width}x{actual_height} @ {reported_fps:.1f} FPS")
        
        # 暖機
        for _ in range(10):
            cap.read()
        
        # 測試實際 FPS
        frame_count = 0
        start = time.perf_counter()
        
        while time.perf_counter() - start < TEST_DURATION:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
            else:
                print("  警告：讀取失敗")
                break
        
        elapsed = time.perf_counter() - start
        actual_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"實際測試: {actual_fps:.2f} FPS")
        print(f"效率: {(actual_fps/fps_target*100):.1f}% (實際/目標)")
        
        results.append({
            'name': name,
            'width': width,
            'height': height,
            'target_fps': fps_target,
            'actual_fps': actual_fps,
            'efficiency': actual_fps/fps_target*100 if fps_target > 0 else 0
        })
        
        cap.release()
        
    except Exception as e:
        print(f"錯誤: {e}")
        results.append({
            'name': name,
            'width': width,
            'height': height,
            'target_fps': fps_target,
            'actual_fps': 0,
            'efficiency': 0
        })

# 總結
print("\n" + "="*70)
print("測試總結")
print("="*70)

# 按實際 FPS 排序
results_sorted = sorted(results, key=lambda x: x['actual_fps'], reverse=True)

print(f"\n{'配置':<20} {'目標 FPS':>10} {'實際 FPS':>10} {'效率':>8}")
print("-" * 70)
for r in results_sorted:
    print(f"{r['name']:<20} {r['target_fps']:>10} {r['actual_fps']:>10.2f} {r['efficiency']:>7.1f}%")

# 找出最佳配置
best = max(results, key=lambda x: x['actual_fps'])
print(f"\n最高 FPS 配置: {best['name']} - {best['actual_fps']:.2f} FPS")

# 建議
print("\n" + "="*70)
print("建議")
print("="*70)

if best['actual_fps'] > 100:
    print("✓ 恭喜！您的相機可以達到高 FPS")
    print(f"  建議使用: {best['width']}x{best['height']} @ {best['target_fps']} FPS")
elif best['actual_fps'] > 50:
    print("✓ 可以達到中等 FPS")
    print(f"  建議使用: {best['width']}x{best['height']} @ {best['target_fps']} FPS")
    print("  提示：降低解析度可能獲得更高 FPS")
elif best['actual_fps'] > 20:
    print("⚠ FPS 較低，可能的原因：")
    print("  1. USB 2.0 頻寬限制 → 升級到 USB 3.0")
    print("  2. 相機設定問題 → 檢查相機設定")
    print("  3. 驅動問題 → 使用原廠 SDK")
else:
    print("✗ FPS 非常低，建議：")
    print("  1. 檢查 USB 連接")
    print("  2. 確認相機型號和規格")
    print("  3. 使用 Arducam 原廠軟體測試")
    print("  4. 考慮更換相機或電腦")

# USB 頻寬分析
print("\n" + "="*70)
print("USB 頻寬分析")
print("="*70)

for r in results_sorted[:3]:  # 顯示前三個配置
    # 計算所需頻寬 (假設 MJPEG 壓縮比 1:10)
    pixels = r['width'] * r['height']
    uncompressed_bw = pixels * 3 * r['target_fps'] / (1024 * 1024)  # MB/s
    mjpeg_bw = uncompressed_bw / 10  # 假設壓縮比 1:10
    
    print(f"\n{r['name']}:")
    print(f"  未壓縮需求: {uncompressed_bw:.1f} MB/s")
    print(f"  MJPEG 需求: {mjpeg_bw:.1f} MB/s")
    print(f"  實際達到: {r['actual_fps']:.1f} FPS")
    
    if mjpeg_bw > 40:
        print(f"  → 需要 USB 3.0 (可達 625 MB/s)")
    elif mjpeg_bw > 30:
        print(f"  → USB 2.0 可能勉強可以 (理論 60 MB/s)")
    else:
        print(f"  → USB 2.0 應該足夠 (理論 60 MB/s)")

print("\n" + "="*70)
print("USB 規格參考:")
print("  USB 2.0: 理論 60 MB/s, 實際約 30-40 MB/s")
print("  USB 3.0: 理論 625 MB/s, 實際約 300-400 MB/s")
print("="*70)
