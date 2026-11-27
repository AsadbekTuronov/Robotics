# Object Detection demo (YOLO + laptop kamerasi)

Faqat jonli kamera, o'qitish (train) yo'q, GPU shart emas.

## 1. O'rnatish (darsdan OLDIN, internet bilan)

```
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Modellarni oldindan yuklab oling (darsda internet bo'lmasa ham ishlashi uchun):

```
python 01_detect_live.py                        # yolo11n.pt yuklanadi, q bilan chiqing
python 02_count_objects.py --world "pen,pencil" # YOLO-World va CLIP yuklanadi, q bilan chiqing
```

Modellar shu papkada saqlanadi; keyingi ishga tushirishlarda internet kerak emas.

## 2. Skriptlar

| Skript | Nima ko'rsatadi | Tugmalar |
|---|---|---|
| `01_detect_live.py` | Jonli detection. `conf` va `iou` (NMS) slayderlari, FPS | `a` agnostic NMS, `q` chiqish |
| `02_count_objects.py` | Har box `#raqam` bilan, o'ngda ro'yxat, tanlangan sinfni sanash | sichqoncha = tanlash, `c` tozalash |
| `02_count_objects.py --world "pen"` | COCO da yo'q obyekt (ruchka) ni matn bilan topish va sanash | |
| `03_follow_object.py` | Obyektni sichqoncha bilan tanlash, CHAPGA / ONGGA / MARKAZDA, YAQINLASH / UZOQLASH | sichqoncha yoki `1-9`, `c`, `q` |

Foydali bayroqlar: `--cam 1` (boshqa kamera), `--conf 0.5`, `--mirror` (03, ko'zgu rejimi).

## 3. Ustma-ust boxlar muammosi

- Har bir box `#raqam` oladi va o'ngda ro'yxat chiqadi, shuning uchun ustma-ust boxlar chalkashmaydi.
- `agnostic_nms=True` (common.py): bitta obyektga ham `cup`, ham `bottle` chiqishining oldini oladi.
- Sichqoncha bosilganda bir nechta box mos kelsa, eng kichigi tanlanadi.

## 4. Dars tartibi (taxminiy)

| Vaqt | Nima |
|---|---|
| 0:00-0:40 | Slaydlar 1-8: CV, detection, metrikalar, robotika, YOLO |
| 0:40-1:00 | Slaydlar 9-13: NMS, model o'lchamlari, alternativlar, haqiqiy loyiha, detection -> harakat |
| 1:00-1:20 | Demo 01 (conf, iou, `a`), 02 (ro'yxat, sanash, `--world "pen"`) |
| 1:20-1:40 | Demo 03 (ergashish), kodni ko'rsatish: `decide()` va `send_to_robot()` |
| 1:40-1:50 | Xulosa, mini topshiriq, savollar |

Slaydlarning har birida speaker notes bor (PowerPoint da "Presenter view" yoki Notes paneli).

## 5. Demo oldidan tekshiruv

- [ ] Kamera ochiladi (`python 01_detect_live.py`). Ochilmasa `--cam 1`, boshqa dasturlar (Zoom, Teams) kamerani band qilmagan bo'lsin.
- [ ] `yolo11n.pt` va `yolov8s-world.pt` papkada bor.
- [ ] Stolda ruchkalar, butilka/krujka/telefon tayyor (COCO da bular bor: bottle, cup, cell phone, book).
- [ ] Yorug'lik yetarli, fon sodda.
- [ ] Terminal shrifti katta, kod pastki-chap emas, hammaga ko'rinadigan.

## 6. Haqiqiy robotga ulash

`03_follow_object.py` dagi `send_to_robot(turn, move, err)` funksiyasi shu uchun. Ichida Arduino (serial) va ROS 2 (Twist) uchun namuna izoh bor. Hozir faqat konsolga chiqaradi.

## 7. Sozlamalar (03 boshida)

`DEAD_ZONE` (o'lik zona), `TOO_SMALL` / `TOO_BIG` (masofa chegaralari), `KP`, `SMOOTH`. Darsda ularni o'zgartirib ta'sirini ko'rsatish mumkin.

## 8. Ma'lum cheklovlar

- Bitta kamera bilan masofa faqat box o'lchamidan taxmin qilinadi (aniq emas).
- Bir xil sinfdan ko'p obyekt bo'lsa, maqsad "oldingi joyga eng yaqin box" orqali ushlab turiladi; obyektlar kesishsa chalkashishi mumkin (bu tracking emas, oddiy variant).
- YOLO-World (`--world`) CPU da oddiy YOLO dan sekinroq; sinflar sonini kam (1-3 ta) tuting.
# Robotics
# Robotics
# Robotics
