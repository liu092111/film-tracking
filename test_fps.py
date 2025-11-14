import cv2
import time

CAM_ID = 0  # 如果不是 0 再改

resolutions = [
    (640, 480),
    (800, 600),
    (960, 540),
    (1280, 720),
]

def test_mode(w, h, fps_req=120, seconds=3.0):
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)

    # 設 MJPEG，很多高速相機必須這樣才吃得到高 FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps_req)

    reported_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    reported_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    reported_fps = cap.get(cv2.CAP_PROP_FPS)

    t0 = time.perf_counter()
    count = 0
    while time.perf_counter() - t0 < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

    t1 = time.perf_counter()
    actual_fps = count / (t1 - t0) if count > 0 else 0.0

    print(f"Request {w}x{h} @ {fps_req}fps, "
          f"device reports {int(reported_w)}x{int(reported_h)} @ {reported_fps:.1f}fps, "
          f"actual capture ≈ {actual_fps:.2f}fps")

    cap.release()


if __name__ == "__main__":
    for (w, h) in resolutions:
        test_mode(w, h)
