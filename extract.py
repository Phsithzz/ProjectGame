"""
extract_pose.py
===============
จับ skeleton จากไฟล์รูปภาพหรือวิดีโอ แล้วบันทึกลง CSV

วิธีใช้:
    # จากรูปภาพ (1 ไฟล์)
    python extract_pose.py --input photo.jpg --label squat_good

    # จากรูปภาพทั้งโฟลเดอร์
    python extract_pose.py --input ./images/squat_good/ --label squat_good

    # จากวิดีโอ (บันทึกทุก N เฟรม)
    python extract_pose.py --input video.mp4 --label squat_bad_heel --every 5

    # ดูภาพ skeleton ด้วยระหว่างประมวลผล
    python extract_pose.py --input video.mp4 --label plank_good --preview

ติดตั้ง:
    pip install mediapipe==0.10.14 opencv-python pandas numpy
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import argparse
import os
import sys

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_CSV   = "squat_data_bad_heel.csv"
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

MP_POSE        = mp.solutions.pose
MP_DRAWING     = mp.solutions.drawing_utils
MP_DRAW_STYLE  = mp.solutions.drawing_styles
LANDMARK_NAMES = [lm.name.lower() for lm in MP_POSE.PoseLandmark]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_columns() -> list[str]:
    cols = []
    for name in LANDMARK_NAMES:
        for axis in ("x", "y", "z", "visibility"):
            cols.append(f"{name}_{axis}")
    cols.append("label")
    return cols


def landmarks_to_row(landmarks, label: str) -> list:
    row = []
    for lm in landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    row.append(label)
    return row


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


def draw_skeleton(frame, landmarks):
    """วาด skeleton ทับบนเฟรม"""
    MP_DRAWING.draw_landmarks(
        frame,
        landmarks,
        MP_POSE.POSE_CONNECTIONS,
        landmark_drawing_spec=MP_DRAW_STYLE.get_default_pose_landmarks_style(),
    )


# ── Process รูปภาพ 1 ไฟล์ ─────────────────────────────────────────────────────

def process_image(path: str, label: str, pose, preview: bool) -> list:
    img = cv2.imread(path)
    if img is None:
        print(f"  [!] อ่านไฟล์ไม่ได้: {path}")
        return []

    rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        print(f"  [!] ไม่พบ pose ใน: {os.path.basename(path)}")
        return []

    row = landmarks_to_row(results.pose_landmarks, label)

    if preview:
        display = img.copy()
        draw_skeleton(display, results.pose_landmarks)
        h, w = display.shape[:2]
        cv2.putText(display, f"label: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 80), 2)
        cv2.imshow("Preview — กด SPACE ถัดไป, Q ออก", display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit(0)
        cv2.destroyAllWindows()

    print(f"  [+] {os.path.basename(path)} → บันทึกแล้ว")
    return [row]


# ── Process โฟลเดอร์รูปภาพ ────────────────────────────────────────────────────

def process_image_folder(folder: str, label: str, pose, preview: bool) -> list:
    files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])

    if not files:
        print(f"  [!] ไม่พบรูปภาพในโฟลเดอร์: {folder}")
        return []

    print(f"  พบ {len(files)} ไฟล์ในโฟลเดอร์")
    all_rows = []
    for path in files:
        rows = process_image(path, label, pose, preview)
        all_rows.extend(rows)
    return all_rows


# ── Process วิดีโอ ────────────────────────────────────────────────────────────

def process_video(path: str, label: str, pose, every: int, preview: bool) -> list:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  [!] เปิดวิดีโอไม่ได้: {path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS)
    duration     = total_frames / fps_video if fps_video > 0 else 0

    print(f"  วิดีโอ: {os.path.basename(path)}")
    print(f"  ความยาว: {duration:.1f} วินาที | {total_frames} เฟรม | {fps_video:.1f} fps")
    print(f"  บันทึกทุก {every} เฟรม → ประมาณ {total_frames // every} แถว")

    all_rows    = []
    frame_idx   = 0
    saved_count = 0
    no_pose     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # บันทึกทุก N เฟรม
        if frame_idx % every == 0:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                row = landmarks_to_row(results.pose_landmarks, label)
                all_rows.append(row)
                saved_count += 1

                if preview:
                    display = frame.copy()
                    draw_skeleton(display, results.pose_landmarks)

                    # แถบข้อมูล
                    h, w = display.shape[:2]
                    cv2.rectangle(display, (0, 0), (w, 44), (20, 20, 20), -1)
                    cv2.putText(
                        display,
                        f"Frame {frame_idx}/{total_frames}  |  saved: {saved_count}  |  label: {label}",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 100), 2,
                    )
                    cv2.rectangle(display, (0, h - 28), (w, h), (20, 20, 20), -1)
                    cv2.putText(display, "Q: หยุดและบันทึก  |  SPACE: หยุดชั่วคราว",
                                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

                    cv2.imshow("Preview Video", display)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print(f"\n  [i] หยุดที่เฟรม {frame_idx}")
                        break
                    elif key == ord(" "):
                        # pause — กด SPACE อีกครั้งเพื่อเล่นต่อ
                        print("  [pause] กด SPACE อีกครั้งเพื่อเล่นต่อ...")
                        while True:
                            if cv2.waitKey(100) & 0xFF == ord(" "):
                                break
            else:
                no_pose += 1

        frame_idx += 1

        # แสดง progress ทุก 100 เฟรม
        if frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  {pct:5.1f}%  เฟรม {frame_idx}/{total_frames}  บันทึก: {saved_count}", end="\r")

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    print(f"\n  [✓] เสร็จ — บันทึก {saved_count} แถว | ไม่พบ pose {no_pose} เฟรม")
    return all_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe pose landmarks → CSV",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input",   required=True,
                        help="ไฟล์รูปภาพ, โฟลเดอร์รูปภาพ หรือไฟล์วิดีโอ")
    parser.add_argument("--label",   required=True,
                        help="ชื่อ class เช่น squat_good, pushup_bad_hip")
    parser.add_argument("--output",  default=OUTPUT_CSV,
                        help=f"ไฟล์ CSV ปลายทาง (default: {OUTPUT_CSV})")
    parser.add_argument("--every",   type=int, default=3,
                        help="วิดีโอ: บันทึกทุกกี่เฟรม (default: 3)")
    parser.add_argument("--preview", action="store_true",
                        help="แสดงหน้าต่าง skeleton ระหว่างประมวลผล")
    parser.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                        help="MediaPipe model complexity 0=เร็ว 2=แม่น (default: 1)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[ERROR] ไม่พบไฟล์หรือโฟลเดอร์: {input_path}")
        sys.exit(1)

    ext = os.path.splitext(input_path)[1].lower()

    print(f"\n{'─'*50}")
    print(f"  Input  : {input_path}")
    print(f"  Label  : {args.label}")
    print(f"  Output : {args.output}")
    print(f"{'─'*50}")

    with MP_POSE.Pose(
        static_image_mode   = os.path.isfile(input_path) and ext in IMAGE_EXTS,
        model_complexity    = args.complexity,
        smooth_landmarks    = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    ) as pose:

        # ── รูปภาพ 1 ไฟล์
        if os.path.isfile(input_path) and ext in IMAGE_EXTS:
            print("  โหมด: รูปภาพ 1 ไฟล์")
            rows = process_image(input_path, args.label, pose, args.preview)

        # ── โฟลเดอร์รูปภาพ
        elif os.path.isdir(input_path):
            print("  โหมด: โฟลเดอร์รูปภาพ")
            rows = process_image_folder(input_path, args.label, pose, args.preview)

        # ── วิดีโอ
        elif os.path.isfile(input_path) and ext in VIDEO_EXTS:
            print("  โหมด: วิดีโอ")
            rows = process_video(input_path, args.label, pose, args.every, args.preview)

        else:
            print(f"[ERROR] ไม่รองรับไฟล์ประเภท: {ext}")
            sys.exit(1)

    save_to_csv(rows, args.output)
    print(f"\n  รวมทั้งหมด {len(rows)} แถวใน {args.output}")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()