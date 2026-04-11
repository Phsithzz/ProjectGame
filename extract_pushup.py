"""
extract_pushup.py
=================
จับ skeleton จากวิดีโอหรือรูปภาพท่า pushup ด้วย MediaPipe
แล้วบันทึกลง CSV สำหรับ train AI

วิธีใช้:
    python extract_pushup.py --input video.mp4 --label pushup_good
    python extract_pushup.py --input video.mp4 --label pushup_bad_neck --every 5
    python extract_pushup.py --input ./images/good/ --label pushup_good --preview
    python extract_pushup.py --input photo.jpg --label pushup_bad_back

Labels ที่รองรับ:
    pushup_good
    pushup_bad_neck
    pushup_bad_back

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
OUTPUT_CSV   = "pushup_data_bad_back.csv"
VALID_LABELS = {"pushup_good", "pushup_bad_neck", "pushup_bad_back"}
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

MP_POSE        = mp.solutions.pose
MP_DRAWING     = mp.solutions.drawing_utils
MP_DRAW_STYLE  = mp.solutions.drawing_styles
LANDMARK_NAMES = [lm.name.lower() for lm in MP_POSE.PoseLandmark]

# landmark index สำคัญสำหรับ pushup
IDX = {
    "NOSE": 0,
    "LEFT_EAR": 7,  "RIGHT_EAR": 8,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,      "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,     "RIGHT_KNEE": 26,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_columns() -> list:
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


def calc_angle(a, b, c) -> float:
    """คำนวณมุมที่จุด b จาก 3 จุด a-b-c (องศา)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def get_elbow_angle(lm) -> float:
    """มุมข้อศอกซ้าย (wrist-elbow-shoulder)"""
    wrist    = [lm[IDX["LEFT_WRIST"]].x,    lm[IDX["LEFT_WRIST"]].y]
    elbow    = [lm[IDX["LEFT_ELBOW"]].x,    lm[IDX["LEFT_ELBOW"]].y]
    shoulder = [lm[IDX["LEFT_SHOULDER"]].x, lm[IDX["LEFT_SHOULDER"]].y]
    return calc_angle(wrist, elbow, shoulder)


def draw_skeleton(frame, landmarks):
    MP_DRAWING.draw_landmarks(
        frame, landmarks, MP_POSE.POSE_CONNECTIONS,
        landmark_drawing_spec=MP_DRAW_STYLE.get_default_pose_landmarks_style(),
    )


def draw_hud(frame, label: str, saved: int, elbow_angle: float = None):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 44), (20, 20, 20), -1)
    cv2.putText(frame, f"label: {label}  |  saved: {saved}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 100), 2)
    if elbow_angle is not None:
        cv2.putText(frame, f"elbow: {elbow_angle:.0f}deg",
                    (w - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 60), 2)
    cv2.rectangle(frame, (0, h - 28), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "Q: หยุดและบันทึก  |  SPACE: pause",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)


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
        lm    = results.pose_landmarks.landmark
        angle = get_elbow_angle(lm)
        draw_hud(display, label, 1, angle)
        cv2.imshow("Preview — SPACE: ถัดไป  Q: ออก", display)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord("q"):
            sys.exit(0)

    print(f"  [+] {os.path.basename(path)}")
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

    print(f"  พบ {len(files)} ไฟล์")
    all_rows = []
    for path in files:
        all_rows.extend(process_image(path, label, pose, preview))
    return all_rows


# ── Process วิดีโอ ────────────────────────────────────────────────────────────

def process_video(path: str, label: str, pose,
                  every: int, preview: bool) -> list:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  [!] เปิดวิดีโอไม่ได้: {path}")
        return []

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_v   = cap.get(cv2.CAP_PROP_FPS)
    dur     = total / fps_v if fps_v > 0 else 0

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
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            if results.pose_landmarks:
                row = landmarks_to_row(results.pose_landmarks, label)
                all_rows.append(row)
                saved_count += 1

                if preview:
                    display = frame.copy()
                    draw_skeleton(display, results.pose_landmarks)
                    lm    = results.pose_landmarks.landmark
                    angle = get_elbow_angle(lm)
                    draw_hud(display, label, saved_count, angle)
                    cv2.imshow("Preview — pushup extractor", display)

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

        if frame_idx % 100 == 0 and total > 0:
            pct = frame_idx / total * 100
            print(f"  {pct:5.1f}%  เฟรม {frame_idx}/{total}  บันทึก: {saved_count}", end="\r")

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    print(f"\n  [✓] เสร็จ — บันทึก {saved_count} แถว | ไม่พบ pose {no_pose} เฟรม")
    return all_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks สำหรับท่า pushup → CSV",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input",      required=True,
                        help="ไฟล์รูปภาพ, โฟลเดอร์รูปภาพ หรือไฟล์วิดีโอ")
    parser.add_argument("--label",      required=True,
                        choices=list(VALID_LABELS),
                        help="class: pushup_good | pushup_bad_neck | pushup_bad_back")
    parser.add_argument("--output",     default=OUTPUT_CSV,
                        help="ไฟล์ CSV ปลายทาง (default: pushup_data.csv)")
    parser.add_argument("--every",      type=int, default=3,
                        help="วิดีโอ: บันทึกทุกกี่เฟรม (default: 3)")
    parser.add_argument("--preview",    action="store_true",
                        help="แสดง skeleton preview ระหว่างประมวลผล")
    parser.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                        help="MediaPipe complexity 0=เร็ว 2=แม่น (default: 1)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] ไม่พบไฟล์หรือโฟลเดอร์: {args.input}")
        sys.exit(1)

    ext = os.path.splitext(args.input)[1].lower()

    print(f"\n{'─'*52}")
    print(f"  Input      : {args.input}")
    print(f"  Label      : {args.label}")
    print(f"  Output     : {args.output}")
    print(f"  Complexity : {args.complexity}")
    print(f"{'─'*52}")

    with MP_POSE.Pose(
        static_image_mode        = os.path.isfile(args.input) and ext in IMAGE_EXTS,
        model_complexity         = args.complexity,
        smooth_landmarks         = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    ) as pose:

        if os.path.isfile(args.input) and ext in IMAGE_EXTS:
            print("  โหมด: รูปภาพ 1 ไฟล์")
            rows = process_image(args.input, args.label, pose, args.preview)

        elif os.path.isdir(args.input):
            print("  โหมด: โฟลเดอร์รูปภาพ")
            rows = process_image_folder(args.input, args.label, pose, args.preview)

        elif os.path.isfile(args.input) and ext in VIDEO_EXTS:
            print("  โหมด: วิดีโอ")
            rows = process_video(args.input, args.label, pose,
                                 args.every, args.preview)
        else:
            print(f"[ERROR] ไม่รองรับไฟล์ประเภท: {ext}")
            sys.exit(1)

    save_to_csv(rows, args.output)
    print(f"\n  รวม {len(rows)} แถว | features: 132 (33 landmarks × x,y,z,vis)")
    print(f"{'─'*52}\n")


if __name__ == "__main__":
    main()
