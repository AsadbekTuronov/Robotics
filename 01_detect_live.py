"""01 - Live kamerada YOLO bilan object detection (eng oddiy variant).

Ko'rsatiladigan g'oyalar:
  - model(frame) -> boxes, confidence, class
  - conf slider: pastroq = ko'proq (xato ham ko'p) obyekt
  - iou slider: NMS chegarasi. Past = ustma-ust boxlar ko'proq o'chadi
  - 'a' tugmasi: agnostic NMS (sinfi har xil ustma-ust boxlarni ham tozalash)
Chiqish: q yoki ESC
"""
import time
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # n = nano: CPU da ham yetarli tez. Katta: s, m, l, x
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Kamera ochilmadi (VideoCapture(1) ni sinab ko'ring).")

WIN = "01 - detection"
cv2.namedWindow(WIN)
cv2.createTrackbar("conf %", WIN, 40, 95, lambda v: None)
cv2.createTrackbar("iou % (NMS)", WIN, 50, 95, lambda v: None)
agnostic = False

while True:
    ok, frame = cap.read()
    if not ok:
        break
    conf = max(cv2.getTrackbarPos("conf %", WIN), 5) / 100
    iou = max(cv2.getTrackbarPos("iou % (NMS)", WIN), 5) / 100

    t0 = time.time()
    result = model(frame, conf=conf, iou=iou, agnostic_nms=agnostic, verbose=False)[0]
    dt = time.time() - t0

    # result.plot() boxlarni o'zi chizadi. Xom ma'lumot kerak bo'lsa:
    #   for b in result.boxes:
    #       x1, y1, x2, y2 = b.xyxy[0];  conf = b.conf[0];  name = model.names[int(b.cls[0])]
    out = result.plot()
    info = f"FPS {1/dt:4.1f} | obyekt: {len(result.boxes)} | conf {conf:.2f} iou {iou:.2f} | agnostic(a): {agnostic}"
    cv2.putText(out, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(WIN, out)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord("a"):
        agnostic = not agnostic

cap.release()
cv2.destroyAllWindows()
