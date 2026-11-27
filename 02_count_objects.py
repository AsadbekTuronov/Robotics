"""02 - Raqamlangan ro'yxat va sanash.

Har bir obyekt #raqam bilan chiziladi va o'ng panelda ro'yxat chiqadi (ustma-ust boxlarni ajratish oson).
Sichqoncha bilan obyektga bosing -> shu sinf SANALADIGAN (target) sinf bo'ladi.

Ishga tushirish:
  python 02_count_objects.py                      # COCO sinflari (bottle, cup, book, cell phone ...)
  python 02_count_objects.py --target bottle
  python 02_count_objects.py --world "pen,pencil"  # COCO da 'pen' YO'Q -> YOLO-World bilan (train shart emas)
Tugmalar: c = tanlovni tozalash, q/ESC = chiqish
"""
import argparse
import cv2
from common import open_camera, load_model, detect, draw_det, side_panel, pick_at, FONT

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--target", default=None, help="sanaladigan sinf nomi (sichqoncha bilan ham tanlanadi)")
ap.add_argument("--world", default=None, help="YOLO-World sinflari, vergul bilan: 'pen,pencil'")
ap.add_argument("--conf", type=float, default=0.35)
args = ap.parse_args()

model = load_model(args.world)
cap = open_camera(args.cam)
target = args.target
click = {"pt": None}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click["pt"] = (x, y)


WIN = "02 - ro'yxat va sanash"
cv2.namedWindow(WIN)
cv2.setMouseCallback(WIN, on_mouse)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    dets = detect(model, frame, conf=args.conf)

    if click["pt"] is not None:
        d = pick_at(dets, click["pt"])
        if d:
            target = d["name"]
        click["pt"] = None

    for d in dets:
        is_t = d["name"] == target
        draw_det(frame, d, color=(0, 255, 0) if is_t else None, thickness=3 if is_t else 2)

    count = sum(1 for d in dets if d["name"] == target) if target else 0
    header = [f"Jami obyekt: {len(dets)}",
              f"SANALADI: {target or '(bosib tanlang)'}",
              f"SON: {count}"]
    panel = side_panel(h, dets, header)
    cv2.putText(frame, f"{target or '-'}: {count}", (10, 40), FONT, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow(WIN, cv2.hconcat([frame, panel]))

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord("c"):
        target = None

cap.release()
cv2.destroyAllWindows()
