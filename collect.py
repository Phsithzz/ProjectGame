"""
collect_squat_data.py
=====================
บันทึกข้อมูลข้อต่อจาก MediaPipe Pose ลงไฟล์ CSV
สำหรับ train โมเดลตรวจจับท่า squat

Labels: good | bad_heel | bad_back

วิธีใช้:
    python collect_squat_data.py --label good
    python collect_squat_data.py --label bad_heel
    python collect_squat_data.py --label bad_back

ติดตั้ง dependencies:
    pip install mediapipe opencv-python pandas
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import argparse
import os
import time


# ─── Config ───────────────────────────────────────────────────────────────────

OUTPUT_CSV = "squat_data_good.csv"
VALID_LABELS = ["good", "bad_heel", "bad_back"]

# MediaPipe landmark ชื่อ (ใช้สร้าง header ของ CSV)
MP_POSE = mp.solutions.pose
LANDMARK_NAMES = [lm.name.lower() for lm in MP_POSE.PoseLandmark]


# ─── สร้าง column headers ──────────────────────────────────────────────────────

def get_column_names() -> list[str]:
    """สร้างชื่อ column: <landmark>_x, _y, _z, _visibility × 33 + label"""
    cols = []
    for name in LANDMARK_NAMES:
        for axis in ("x", "y", "z", "visibility"):
            cols.append(f"{name}_{axis}")
    cols.append("label")
    return cols


# ─── แปลง landmarks → row ─────────────────────────────────────────────────────

def landmarks_to_row(landmarks, label: str) -> list:
    """แปลง MediaPipe NormalizedLandmarkList เป็น list 1 แถว"""
    row = []
    for lm in landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    row.append(label)
    return row


# ─── เขียน CSV (append หรือสร้างใหม่) ─────────────────────────────────────────

def append_to_csv(rows: list[list], filepath: str):
    """บันทึก rows ลง CSV (สร้างใหม่ถ้าไม่มีไฟล์, ต่อท้ายถ้ามีอยู่แล้ว)"""
    cols = get_column_names()
    df_new = pd.DataFrame(rows, columns=cols)

    if os.path.exists(filepath):
        df_new.to_csv(filepath, mode="a", header=False, index=False)
        print(f"[✓] ต่อท้ายข้อมูล {len(rows)} แถว → {filepath}")
    else:
        df_new.to_csv(filepath, index=False)
        print(f"[✓] สร้างไฟล์ใหม่ {len(rows)} แถว → {filepath}")


# ─── Main loop ────────────────────────────────────────────────────────────────

def collect(label: str, camera_id: int = 0):
    print(f"\n{'─'*50}")
    print(f"  Mode  : {label.upper()}")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"{'─'*50}")
    print("  กด SPACE เพื่อบันทึกเฟรม")
    print("  กด S    เพื่อบันทึกทั้งหมดลง CSV และออก")
    print("  กด Q    เพื่อออกโดยไม่บันทึก")
    print(f"{'─'*50}\n")

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"ไม่สามารถเปิดกล้อง index {camera_id} ได้")

    collected_rows: list[list] = []
    last_saved_time = 0.0

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[!] ไม่สามารถอ่านภาพจากกล้องได้")
                break

            # แปลงสี BGR → RGB สำหรับ MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # วาด skeleton
            display = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            # แสดง HUD
            h, w = display.shape[:2]
            status_color = (0, 220, 100)  # เขียว = พร้อม
            status_text = "READY"

            if not results.pose_landmarks:
                status_color = (0, 80, 255)  # ส้ม = ไม่เจอคน
                status_text = "NO POSE"

            # กรอบ label
            cv2.rectangle(display, (0, 0), (w, 38), (30, 30, 30), -1)
            cv2.putText(
                display,
                f"Label: {label}   Frames: {len(collected_rows)}   [{status_text}]",
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                status_color,
                2,
            )

            # คำแนะนำด้านล่าง
            cv2.rectangle(display, (0, h - 32), (w, h), (30, 30, 30), -1)
            cv2.putText(
                display,
                "SPACE: save frame   S: export CSV   Q: quit",
                (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (180, 180, 180),
                1,
            )

            cv2.imshow("Squat Data Collector", display)
            key = cv2.waitKey(1) & 0xFF

            # กด SPACE → บันทึกเฟรมปัจจุบัน
            if key == ord(" "):
                now = time.time()
                if results.pose_landmarks and (now - last_saved_time) > 0.1:
                    row = landmarks_to_row(results.pose_landmarks, label)
                    collected_rows.append(row)
                    last_saved_time = now
                    print(f"[+] บันทึกเฟรม {len(collected_rows)}")
                elif not results.pose_landmarks:
                    print("[!] ตรวจไม่พบท่า ไม่สามารถบันทึกได้")

            # กด S → export CSV แล้วออก
            elif key == ord("s"):
                break

            # กด Q → ออกโดยไม่บันทึก
            elif key == ord("q"):
                print("[x] ออกโดยไม่บันทึก")
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

    if collected_rows:
        append_to_csv(collected_rows, OUTPUT_CSV)
    else:
        print("[!] ไม่มีข้อมูลที่บันทึก")


# ─── Auto-record mode (กดค้าง) ────────────────────────────────────────────────

def collect_auto(label: str, camera_id: int = 0, fps_save: int = 5):
    """
    บันทึกอัตโนมัติ N เฟรมต่อวินาที โดยไม่ต้องกด SPACE
    fps_save = จำนวนเฟรมที่บันทึกต่อวินาที
    กด S เพื่อหยุด + บันทึก  |  Q เพื่อออกโดยไม่บันทึก
    """
    print(f"\n[AUTO MODE] บันทึก {fps_save} fps — label: {label}")
    print("กด S เพื่อหยุด + บันทึก  |  Q เพื่อออกโดยไม่บันทึก\n")

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"ไม่สามารถเปิดกล้อง index {camera_id} ได้")

    collected_rows: list[list] = []
    interval = 1.0 / fps_save
    last_saved_time = 0.0

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            display = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # บันทึกตาม interval
                now = time.time()
                if (now - last_saved_time) >= interval:
                    row = landmarks_to_row(results.pose_landmarks, label)
                    collected_rows.append(row)
                    last_saved_time = now

            h, w = display.shape[:2]
            cv2.rectangle(display, (0, 0), (w, 38), (30, 30, 30), -1)
            cv2.putText(
                display,
                f"AUTO | Label: {label}   Frames: {len(collected_rows)}",
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (80, 220, 100),
                2,
            )

            cv2.imshow("Squat Data Collector [AUTO]", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                break
            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

    if collected_rows:
        append_to_csv(collected_rows, OUTPUT_CSV)
    else:
        print("[!] ไม่มีข้อมูลที่บันทึก")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Squat Pose Data Collector")
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        choices=VALID_LABELS,
        help="class ของท่า: good | bad_heel | bad_back",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="index ของกล้อง (default: 0)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="บันทึกอัตโนมัติโดยไม่ต้องกด SPACE",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="จำนวนเฟรมต่อวินาทีในโหมด auto (default: 5)",
    )
    args = parser.parse_args()

    if args.auto:
        collect_auto(label=args.label, camera_id=args.camera, fps_save=args.fps)
    else:
        collect(label=args.label, camera_id=args.camera)