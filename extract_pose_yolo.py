"""
extract_pose_yolo.py
====================
จับ keypoints จากไฟล์รูปภาพหรือวิดีโอด้วย YOLOv8-Pose
แล้วบันทึกลง CSV เพื่อนำไป train AI model

เปลี่ยนจาก MediaPipe (33 landmarks) → YOLOv8-Pose (17 keypoints COCO)
Feature columns: 51 (17 keypoints × x, y, confidence)

วิธีใช้:
    # รูปภาพ 1 ไฟล์
    python extract_pose_yolo.py --input photo.jpg --label squat_good

    # รูปภาพทั้งโฟลเดอร์
    python extract_pose_yolo.py --input ./images/squat_good/ --label squat_good

    # วิดีโอ (บันทึกทุก N เฟรม)
    python extract_pose_yolo.py --input video.mp4 --label squat_bad_heel --every 5

    # ดู skeleton preview
    python extract_pose_yolo.py --input video.mp4 --label plank_good --preview

ติดตั้ง:
    pip install ultralytics opencv-python pandas numpy

โมเดล yolov8n-pose.pt จะดาวน์โหลดอัตโนมัติครั้งแรก
หรือระบุ path เองด้วย --model yolov8n-pose.pt
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import os
import sys

# ── YOLO Keypoint ทั้ง 17 จุด (COCO format) ──────────────────────────────────
YOLO_KEYPOINTS = [
    "nose",
    "left_eye",    "right_eye",
    "left_ear",    "right_ear",
    "left_shoulder","right_shoulder",
    "left_elbow",  "right_elbow",
    "left_wrist",  "right_wrist",
    "left_hip",    "right_hip",
    "left_knee",   "right_knee",
    "left_ankle",  "right_ankle",
]

OUTPUT_CSV   = "squat_good.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ── Skeleton connections สำหรับวาดเส้น ───────────────────────────────────────
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),          # หัว
    (5,6),                             # ไหล่
    (5,7),(7,9),(6,8),(8,10),          # แขน
    (5,11),(6,12),(11,12),             # ลำตัว
    (11,13),(13,15),(12,14),(14,16),   # ขา
]
SKELETON_COLOR = (0, 220, 100)
KP_COLOR       = (0, 180, 255)


# ── Column headers ────────────────────────────────────────────────────────────

def get_columns() -> list:
    cols = []
    for kp in YOLO_KEYPOINTS:
        cols += [f"{kp}_x", f"{kp}_y", f"{kp}_conf"]
    cols.append("label")
    return cols


# ── แปลง keypoints → row (normalize เป็น 0-1) ────────────────────────────────

def keypoints_to_row(kps: np.ndarray, img_w: int, img_h: int, label: str) -> list:
    """
    kps: array shape (17, 3) — [x_pixel, y_pixel, confidence]
    normalize x ด้วย img_w, y ด้วย img_h → 0-1
    """
    row = []
    for kp in kps:
        x    = float(np.clip(kp[0] / img_w, 0, 1))
        y    = float(np.clip(kp[1] / img_h, 0, 1))
        conf = float(np.clip(kp[2], 0, 1))
        row += [x, y, conf]
    row.append(label)
    return row


# ── บันทึก CSV ────────────────────────────────────────────────────────────────

def save_to_csv(rows: list, filepath: str):
    if not rows:
        print("  [!] ไม่มีแถวที่จะบันทึก")
        return
    df = pd.DataFrame(rows, columns=get_columns())
    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, index=False)
    print(f"  [✓] บันทึก {len(rows)} แถว → {filepath}")


# ── วาด skeleton บน frame ────────────────────────────────────────────────────

def draw_skeleton(frame, kps: np.ndarray, conf_thresh: float = 0.3):
    h, w = frame.shape[:2]
    pts = []
    for kp in kps:
        px = int(kp[0])
        py = int(kp[1])
        cf = float(kp[2])
        pts.append((px, py, cf))

    # วาดเส้น skeleton
    for i, j in SKELETON:
        if pts[i][2] > conf_thresh and pts[j][2] > conf_thresh:
            cv2.line(frame, (pts[i][0], pts[i][1]),
                     (pts[j][0], pts[j][1]), SKELETON_COLOR, 2)

    # วาด keypoint
    for px, py, cf in pts:
        if cf > conf_thresh:
            cv2.circle(frame, (px, py), 5, KP_COLOR, -1)


# ── เลือก person ที่มี confidence สูงสุด ─────────────────────────────────────

def best_person(results) -> np.ndarray | None:
    """
    YOLO อาจเจอหลายคนในภาพ → เลือกคนที่ detection confidence สูงสุด
    คืนค่า keypoints shape (17, 3) หรือ None ถ้าไม่เจอ
    """
    if results is None or len(results) == 0:
        return None
    r = results[0]
    if r.keypoints is None or len(r.keypoints) == 0:
        return None

    boxes = r.boxes
    kps   = r.keypoints.data.cpu().numpy()  # (N, 17, 3)

    if len(kps) == 0:
        return None

    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy()
        best  = int(np.argmax(confs))
    else:
        best  = 0

    return kps[best]  # (17, 3)


# ── Process รูปภาพ 1 ไฟล์ ────────────────────────────────────────────────────

def process_image(path: str, label: str, model, preview: bool) -> list:
    img = cv2.imread(path)
    if img is None:
        print(f"  [!] อ่านไฟล์ไม่ได้: {path}")
        return []

    h, w = img.shape[:2]
    results = model(img, verbose=False)
    kps     = best_person(results)

    if kps is None:
        print(f"  [!] ไม่พบ pose ใน: {os.path.basename(path)}")
        return []

    row = keypoints_to_row(kps, w, h, label)

    if preview:
        display = img.copy()
        draw_skeleton(display, kps)
        cv2.putText(display, f"label: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 80), 2)
        cv2.imshow("Preview — SPACE: ถัดไป  Q: ออก", display)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord("q"):
            sys.exit(0)

    print(f"  [+] {os.path.basename(path)}")
    return [row]


# ── Process โฟลเดอร์รูปภาพ ───────────────────────────────────────────────────

def process_image_folder(folder: str, label: str, model, preview: bool) -> list:
    files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])
    if not files:
        print(f"  [!] ไม่พบรูปภาพในโฟลเดอร์: {folder}")
        return []

    print(f"  พบ {len(files)} ไฟล์")
    all_rows = []
    for path in files:
        all_rows.extend(process_image(path, label, model, preview))
    return all_rows


# ── Process วิดีโอ ───────────────────────────────────────────────────────────

def process_video(path: str, label: str, model,
                  every: int, preview: bool) -> list:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  [!] เปิดวิดีโอไม่ได้: {path}")
        return []

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_v  = cap.get(cv2.CAP_PROP_FPS)
    dur    = total / fps_v if fps_v > 0 else 0

    print(f"  วิดีโอ  : {os.path.basename(path)}")
    print(f"  ความยาว : {dur:.1f} วินาที | {total} เฟรม | {fps_v:.1f} fps")
    print(f"  บันทึกทุก {every} เฟรม → ประมาณ {total // every} แถว")

    all_rows    = []
    frame_idx   = 0
    saved_count = 0
    no_pose     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every == 0:
            h, w    = frame.shape[:2]
            results = model(frame, verbose=False)
            kps     = best_person(results)

            if kps is not None:
                row = keypoints_to_row(kps, w, h, label)
                all_rows.append(row)
                saved_count += 1

                if preview:
                    display = frame.copy()
                    draw_skeleton(display, kps)
                    cv2.rectangle(display, (0, 0), (w, 44), (20, 20, 20), -1)
                    cv2.putText(
                        display,
                        f"Frame {frame_idx}/{total}  |  saved: {saved_count}  |  label: {label}",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 100), 2,
                    )
                    cv2.rectangle(display, (0, h-28), (w, h), (20, 20, 20), -1)
                    cv2.putText(display, "Q: หยุดและบันทึก  |  SPACE: pause",
                                (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1)
                    cv2.imshow("Preview — YOLOv8 Pose", display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print(f"\n  [i] หยุดที่เฟรม {frame_idx}")
                        break
                    elif key == ord(" "):
                        print("  [pause] กด SPACE อีกครั้งเพื่อเล่นต่อ...")
                        while cv2.waitKey(100) & 0xFF != ord(" "):
                            pass
            else:
                no_pose += 1

        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"  {pct:5.1f}%  เฟรม {frame_idx}/{total}  บันทึก: {saved_count}", end="\r")

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    print(f"\n  [✓] เสร็จ — บันทึก {saved_count} แถว | ไม่พบ pose {no_pose} เฟรม")
    return all_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract YOLOv8-Pose keypoints → CSV",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input",      required=True,
                        help="ไฟล์รูปภาพ, โฟลเดอร์รูปภาพ หรือไฟล์วิดีโอ")
    parser.add_argument("--label",      required=True,
                        help="ชื่อ class เช่น squat_good, plank_bad_hip")
    parser.add_argument("--output",     default=OUTPUT_CSV,
                        help="ไฟล์ CSV ปลายทาง (default: pose_data.csv)")
    parser.add_argument("--model",      default="yolov8n-pose.pt",
                        help="path ไปยัง YOLO model (default: yolov8n-pose.pt)")
    parser.add_argument("--every",      type=int, default=3,
                        help="วิดีโอ: บันทึกทุกกี่เฟรม (default: 3)")
    parser.add_argument("--conf",       type=float, default=0.5,
                        help="YOLO detection confidence threshold (default: 0.5)")
    parser.add_argument("--preview",    action="store_true",
                        help="แสดง skeleton preview ระหว่างประมวลผล")
    args = parser.parse_args()

    # import YOLO ตอน runtime เพื่อให้ error message ชัดเจน
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ยังไม่ได้ติดตั้ง ultralytics")
        print("  รัน: pip install ultralytics")
        sys.exit(1)

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[ERROR] ไม่พบไฟล์หรือโฟลเดอร์: {input_path}")
        sys.exit(1)

    ext = os.path.splitext(input_path)[1].lower()

    print(f"\n{'─'*50}")
    print(f"  Model  : {args.model}")
    print(f"  Input  : {input_path}")
    print(f"  Label  : {args.label}")
    print(f"  Output : {args.output}")
    print(f"  Conf   : {args.conf}")
    print(f"{'─'*50}")

    # โหลด YOLO model
    print("  โหลด YOLO model...")
    yolo = YOLO(args.model)
    yolo.conf = args.conf
    print("  ✓ โหลด model สำเร็จ\n")

    # ── รูปภาพ 1 ไฟล์
    if os.path.isfile(input_path) and ext in IMAGE_EXTS:
        print("  โหมด: รูปภาพ 1 ไฟล์")
        rows = process_image(input_path, args.label, yolo, args.preview)

    # ── โฟลเดอร์รูปภาพ
    elif os.path.isdir(input_path):
        print("  โหมด: โฟลเดอร์รูปภาพ")
        rows = process_image_folder(input_path, args.label, yolo, args.preview)

    # ── วิดีโอ
    elif os.path.isfile(input_path) and ext in VIDEO_EXTS:
        print("  โหมด: วิดีโอ")
        rows = process_video(input_path, args.label, yolo, args.every, args.preview)

    else:
        print(f"[ERROR] ไม่รองรับไฟล์ประเภท: {ext}")
        sys.exit(1)

    save_to_csv(rows, args.output)
    print(f"\n  รวมทั้งหมด {len(rows)} แถว | features: 51 (17 kp × x,y,conf)")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()
