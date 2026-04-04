"""
predict_realtime.py
===================
ใช้ model ที่ train แล้วตรวจจับท่า squat แบบ real-time

วางไฟล์เหล่านี้ในโฟลเดอร์เดียวกัน:
    predict_realtime.py
    squat_model.pkl
    label_encoder.pkl

รัน:
    python predict_realtime.py
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib

# ── โหลด model ──────────────────────────────────────────────

model = joblib.load("squat_model.pkl")
le    = joblib.load("label_encoder.pkl")

print(f"✓ โหลด model สำเร็จ")
print(f"  Classes: {le.classes_}")


# ── Config แสดงผล ───────────────────────────────────────────

# สีและข้อความแต่ละ class  (BGR)
CLASS_CONFIG = {
    "good":     {"color": (60, 200, 80),  "text": "GOOD ✓",      "emoji": "✓"},
    "bad_heel": {"color": (0,  120, 255), "text": "BAD HEEL ✗",  "emoji": "✗"},
    "bad_back": {"color": (0,  60,  220), "text": "BAD BACK ✗",  "emoji": "✗"},
}
DEFAULT_CONFIG = {"color": (180, 180, 180), "text": "DETECTING...", "emoji": ""}


# ── Helper: แปลง landmarks → feature vector ─────────────────

def landmarks_to_vector(landmarks) -> np.ndarray:
    row = []
    for lm in landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(row).reshape(1, -1)


# ── Helper: วาด overlay ─────────────────────────────────────

def draw_overlay(frame, label: str, confidence: float, proba: np.ndarray, classes):
    h, w = frame.shape[:2]
    cfg = CLASS_CONFIG.get(label, DEFAULT_CONFIG)
    color = cfg["color"]
    text  = cfg["text"]

    # กล่องหลัก (บนซ้าย)
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)

    # ชื่อ class + confidence
    cv2.putText(
        frame, f"{text}  {confidence*100:.1f}%",
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        color, 2, cv2.LINE_AA,
    )

    # แถบ confidence ด้านล่าง
    bar_w = int((w - 24) * confidence)
    cv2.rectangle(frame, (12, 44), (w - 12, 50), (60, 60, 60), -1)
    cv2.rectangle(frame, (12, 44), (12 + bar_w, 50), color, -1)

    # แสดง probability ทุก class (ด้านล่างซ้าย)
    y0 = h - (len(classes) * 26) - 10
    for i, cls in enumerate(classes):
        p = proba[0][i]
        bar_len = int(160 * p)
        y = y0 + i * 26
        c = CLASS_CONFIG.get(cls, DEFAULT_CONFIG)["color"]

        cv2.rectangle(frame, (12, y), (12 + bar_len, y + 18), c, -1)
        cv2.rectangle(frame, (12, y), (172, y + 18), (80, 80, 80), 1)
        cv2.putText(
            frame, f"{cls}: {p*100:.0f}%",
            (178, y + 13),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (210, 210, 210), 1, cv2.LINE_AA,
        )

    # คำแนะนำด้านล่าง
    cv2.rectangle(frame, (0, h - 28), (w, h), (20, 20, 20), -1)
    cv2.putText(
        frame, "Q: quit",
        (12, h - 9),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (150, 150, 150), 1,
    )


# ── Main loop ────────────────────────────────────────────────

def run(camera_id: int = 0):
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"ไม่สามารถเปิดกล้อง index {camera_id} ได้")

    # smooth prediction (เฉลี่ย N เฟรมหลัง เพื่อไม่ให้กระตุก)
    SMOOTH_N = 5
    pred_buffer = []

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            label      = None
            confidence = 0.0
            proba      = np.zeros((1, len(le.classes_)))

            if results.pose_landmarks:
                # วาด skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # predict
                vec   = landmarks_to_vector(results.pose_landmarks)
                proba = model.predict_proba(vec)
                idx   = int(np.argmax(proba))

                # smooth buffer
                pred_buffer.append(idx)
                if len(pred_buffer) > SMOOTH_N:
                    pred_buffer.pop(0)

                # majority vote
                smooth_idx = max(set(pred_buffer), key=pred_buffer.count)
                label      = le.inverse_transform([smooth_idx])[0]
                confidence = float(proba[0][smooth_idx])

                draw_overlay(frame, label, confidence, proba, le.classes_)

            else:
                # ไม่เจอคน
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
                cv2.putText(
                    frame, "NO POSE DETECTED",
                    (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (100, 100, 100), 2,
                )

            cv2.imshow("Squat Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("ปิดโปรแกรมแล้ว")


if __name__ == "__main__":
    run(camera_id=0)
