"""Umumiy yordamchi funksiyalar (02 va 03 skriptlar ishlatadi)."""
import os
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def open_camera(index=0, width=640, height=480):
    # Windows da CAP_DSHOW kamerani tezroq ochadi
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit("Kamera ochilmadi. --cam 1 ni sinab ko'ring yoki kamerani boshqa dastur band qilmaganini tekshiring.")
    return cap


def load_model(world=None, weights="yolo11n.pt"):
    """world='pen,notebook' berilsa YOLO-World (open-vocabulary) ishlatiladi:
    COCO ro'yxatida yo'q obyektlarni ham matn orqali topadi, train shart emas."""
    if world:
        from ultralytics import YOLOWorld
        model = YOLOWorld("yolov8s-world.pt")
        model.set_classes([c.strip() for c in world.split(",")])
        return model
    from ultralytics import YOLO
    return YOLO(weights)


def detect(model, frame, conf=0.4, iou=0.5, agnostic=True):
    """Bitta kadrda detection. Natija: chapdan o'ngga tartiblangan, #id berilgan ro'yxat.
    agnostic=True: bir-biriga ustma-ust tushgan, lekin sinfi har xil boxlarni ham NMS qiladi
    (masalan bitta obyekt uchun ham 'cup', ham 'bottle' chiqib qolishining oldini oladi)."""
    r = model(frame, conf=conf, iou=iou, agnostic_nms=agnostic, verbose=False)[0]
    dets = []
    for b in r.boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        c = int(b.cls[0])
        dets.append({"box": (x1, y1, x2, y2), "conf": float(b.conf[0]), "cls": c, "name": r.names[c]})
    dets.sort(key=lambda d: d["box"][0])
    for i, d in enumerate(dets, 1):
        d["id"] = i
    return dets


def color_of(cls_id):
    return ((cls_id * 67 + 80) % 200 + 55, (cls_id * 131 + 40) % 200 + 55, (cls_id * 197 + 100) % 200 + 55)


def draw_det(frame, d, color=None, thickness=2, extra=""):
    x1, y1, x2, y2 = (int(v) for v in d["box"])
    color = color or color_of(d["cls"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"#{d['id']} {d['name']} {d['conf']:.0%}{extra}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, max(th + 8, y1)), color, -1)
    cv2.putText(frame, label, (x1 + 3, max(th + 2, y1 - 4)), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def side_panel(h, dets, header_lines, selected_id=None, width=270):
    """O'ng tomondagi ro'yxat paneli: har bir obyekt #raqam bilan."""
    p = np.full((h, width, 3), 30, np.uint8)
    y = 28
    for t in header_lines:
        cv2.putText(p, t, (10, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    cv2.line(p, (8, y - 8), (width - 8, y - 8), (90, 90, 90), 1)
    y += 14
    for d in dets:
        sel = d["id"] == selected_id
        text = f"#{d['id']} {d['name']} {d['conf']:.0%}"
        cv2.putText(p, text, (10, y), FONT, 0.55, (0, 255, 0) if sel else color_of(d["cls"]), 2 if sel else 1, cv2.LINE_AA)
        y += 24
        if y > h - 10:
            break
    return p


def point_in_box(pt, box):
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def pick_at(dets, pt):
    """Sichqoncha bosilgan nuqtadagi obyektni tanlaydi. Bir nechta box ustma-ust bo'lsa,
    eng KICHIK box tanlanadi (odatda u ustdagi/ichkaridagi obyekt)."""
    hit = [d for d in dets if point_in_box(pt, d["box"])]
    if not hit:
        return None
    return min(hit, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
