"""03 - Obyektni tanlash va unga qarab yo'nalish berish (robotning "ko'zi").

Qanday ishlaydi:
  1. YOLO kadrdagi obyektlarni topadi (har biri #raqam bilan).
  2. Siz sichqoncha bilan obyektga bosasiz (yoki 1-9 tugmasi) -> shu obyekt "maqsad".
  3. Har kadrda shu sinfdan, oldingi joyga eng yaqin boxni maqsad deb olamiz (oddiy association).
  4. Box markazining kadr markaziga nisbatan siljishi (err) -> CHAPGA / ONGGA / MARKAZDA.
  5. Box yuzasi kadrga nisbatan -> YAQINLASH / UZOQLASH / MASOFA YAXSHI (kamera 1 ta bo'lsa, o'lcham ~ masofa).

Ishga tushirish:
  python 03_follow_object.py
  python 03_follow_object.py --world "pen"     # YOLO-World: COCO da yo'q obyektlar
  python 03_follow_object.py --mirror          # kamerani "ko'zgu" qilib ko'rsatish (faqat demo qulayligi uchun)
Tugmalar: sichqoncha chap = tanlash, 1-9 = raqam bo'yicha tanlash, c = tozalash, q/ESC = chiqish

Eslatma: tasvirdagi CHAP = robotning chapi (kamera robotga qarab turgan holat), shuning uchun
default holatda kamera tasviri aylantirilmaydi.
"""
import argparse
import math
import cv2
from common import open_camera, load_model, detect, draw_det, side_panel, pick_at, FONT

# ---- sozlamalar (darsda qiymatlarni o'zgartirib ko'rsatish mumkin) ----
DEAD_ZONE = 0.12   # markaz atrofida +-12%: "to'g'ri" deb hisoblanadi (titrashni oldini oladi)
TOO_SMALL = 0.06   # box yuzasi kadrning 6% idan kichik -> uzoq -> YAQINLASH
TOO_BIG = 0.35     # 35% idan katta -> juda yaqin -> UZOQLASH
KP = 1.0           # proporsional koeffitsient: burilish tezligi = KP * err
SMOOTH = 0.6       # err ni silliqlash (0 = silliqlashsiz, 1 = qotib qoladi)

TEXT = {"LEFT": "CHAPGA BURIL", "RIGHT": "ONGGA BURIL", "CENTER": "MARKAZDA",
        "FORWARD": "YAQINLASH", "BACKWARD": "UZOQLASH", "HOLD": "MASOFA YAXSHI"}


def decide(box, w, h, dead_zone=DEAD_ZONE, too_small=TOO_SMALL, too_big=TOO_BIG):
    """Box -> (err, area, turn, move). err: -1 (chap chekka) ... 0 (markaz) ... +1 (o'ng chekka)."""
    x1, y1, x2, y2 = box
    err = ((x1 + x2) / 2 - w / 2) / (w / 2)
    area = (x2 - x1) * (y2 - y1) / (w * h)
    turn = "LEFT" if err < -dead_zone else "RIGHT" if err > dead_zone else "CENTER"
    move = "FORWARD" if area < too_small else "BACKWARD" if area > too_big else "HOLD"
    return err, area, turn, move


def send_to_robot(turn, move, err):
    """Haqiqiy robotda aynan shu yerga yoziladi. Masalan:
      Arduino (serial):  ser.write(f"{turn},{move},{err:.2f}\\n".encode())
      ROS 2 (Twist):     msg.angular.z = -KP * err      # obyekt o'ngda (err>0) -> soat strelkasi bo'ylab buril
                         msg.linear.x  = 0.2 if move == "FORWARD" else -0.1 if move == "BACKWARD" else 0.0
    Hozircha faqat konsolga chiqaramiz."""
    print(f"CMD  turn={turn:<6} move={move:<8} err={err:+.2f}")


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--world", default=None)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--mirror", action="store_true")
    args = ap.parse_args()

    model = load_model(args.world)
    cap = open_camera(args.cam)

    sel = {"name": None, "center": None, "lost": 0}
    click = {"pt": None}
    smooth_err = 0.0
    last_cmd = None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = (x, y)

    WIN = "03 - obyektga ergashish"
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        ok, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        dets = detect(model, frame, conf=args.conf)

        # --- tanlash (sichqoncha) ---
        if click["pt"] is not None:
            d = pick_at(dets, click["pt"])
            if d:
                sel.update(name=d["name"], center=center(d["box"]), lost=0)
            click["pt"] = None

        # --- maqsadni topish: shu sinf + oldingi joyga eng yaqin box ---
        target = None
        if sel["name"]:
            cand = [d for d in dets if d["name"] == sel["name"]]
            if cand:
                best = min(cand, key=lambda d: math.dist(center(d["box"]), sel["center"]))
                # sakrab ketmaslik uchun: juda uzoq boxni olmaymiz (yo'qolganiga uzoq vaqt bo'lmasa)
                if sel["lost"] > 15 or math.dist(center(best["box"]), sel["center"]) < 0.35 * max(w, h):
                    target = best
            if target:
                sel["center"], sel["lost"] = center(target["box"]), 0
            else:
                sel["lost"] += 1

        # --- chizish ---
        for d in dets:
            if target is not None and d is target:
                continue
            draw_det(frame, d, color=(150, 150, 150), thickness=1)

        # markaz va o'lik zona
        cx = w // 2
        dz = int(DEAD_ZONE * w / 2)
        cv2.line(frame, (cx, 0), (cx, h), (255, 255, 0), 1)
        cv2.line(frame, (cx - dz, 0), (cx - dz, h), (255, 200, 0), 1)
        cv2.line(frame, (cx + dz, 0), (cx + dz, h), (255, 200, 0), 1)

        status = ["Tanlang: sichqoncha yoki 1-9"]
        if target is not None:
            err, area, turn, move = decide(target["box"], w, h)
            smooth_err = SMOOTH * smooth_err + (1 - SMOOTH) * err
            # burilish qarorini silliqlangan err bo'yicha olamiz (titrash kamayadi)
            turn = "LEFT" if smooth_err < -DEAD_ZONE else "RIGHT" if smooth_err > DEAD_ZONE else "CENTER"
            draw_det(frame, target, color=(0, 255, 0), thickness=4)
            tx, ty = (int(v) for v in center(target["box"]))
            cv2.line(frame, (cx, h // 2), (tx, ty), (0, 255, 255), 2)
            cv2.circle(frame, (tx, ty), 6, (0, 0, 255), -1)

            # katta ko'rsatma (yuqorida)
            col = (0, 255, 0) if (turn == "CENTER" and move == "HOLD") else (0, 200, 255)
            cv2.putText(frame, TEXT[turn], (10, 40), FONT, 1.2, col, 3, cv2.LINE_AA)
            cv2.putText(frame, TEXT[move], (10, 80), FONT, 0.9, col, 2, cv2.LINE_AA)
            if turn == "LEFT":
                cv2.arrowedLine(frame, (cx + 60, 40), (cx - 60, 40), col, 6, tipLength=0.4)
            elif turn == "RIGHT":
                cv2.arrowedLine(frame, (cx - 60, 40), (cx + 60, 40), col, 6, tipLength=0.4)
            cv2.putText(frame, f"burilish tezligi = KP*err = {KP * smooth_err:+.2f}", (10, h - 12), FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

            cmd = (turn, move)
            if cmd != last_cmd:
                send_to_robot(turn, move, smooth_err)
                last_cmd = cmd
            status = [f"Maqsad: #{target['id']} {target['name']}", f"err: {smooth_err:+.2f}", f"box yuzasi: {area:.0%}"]
        elif sel["name"]:
            cv2.putText(frame, "OBYEKT YOQOLDI - qidirilmoqda...", (10, 40), FONT, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            if last_cmd != "STOP":
                print("CMD  STOP (maqsad yo'q)")
                last_cmd = "STOP"
            status = [f"Qidirilmoqda: {sel['name']}", f"yo'qolgan kadrlar: {sel['lost']}"]

        panel = side_panel(h, dets, status, selected_id=target["id"] if target else None)
        cv2.imshow(WIN, cv2.hconcat([frame, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            sel.update(name=None, center=None, lost=0)
            last_cmd = None
        if ord("1") <= key <= ord("9"):
            d = next((d for d in dets if d["id"] == key - ord("0")), None)
            if d:
                sel.update(name=d["name"], center=center(d["box"]), lost=0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
