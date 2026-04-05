"""
predict_realtime.py
===================
ใช้ model ที่ train แล้วตรวจจับท่า squat แบบ real-time
- UI อ่านง่ายขึ้น ตัวอักษรใหญ่ขึ้น
- นับจำนวนครั้งอัตโนมัติ
- แสดง feedback เมื่อทำผิดท่า

วางไฟล์เหล่านี้ในโฟลเดอร์เดียวกัน:
    predict_realtime.py
    squat_model.pkl
    label_encoder.pkl

รัน:
    python predict_realtime.py

ปุ่ม:
    Q — ออกจากโปรแกรม
    R — reset นับใหม่
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# ── โหลด model ────────────────────────────────────────────────────────────────

model = joblib.load("squat_model.pkl")
le    = joblib.load("label_encoder.pkl")

print(f"✓ โหลด model สำเร็จ")
print(f"  Classes: {le.classes_}")

# ── Config ─────────────────────────────────────────────────────────────────────

# BGR colors
COLOR_GREEN  = (60,  210,  80)
COLOR_ORANGE = (0,   165, 255)
COLOR_RED    = (60,   60, 220)
COLOR_WHITE  = (240, 240, 240)
COLOR_GRAY   = (130, 130, 130)
COLOR_DARK   = (25,   25,  25)
COLOR_BLACK  = (0,     0,   0)

CLASS_CONFIG = {
    "squat_good":     {"color": COLOR_GREEN,  "text": "GOOD",     "feedback": ""},
    "squat_bad_heel": {"color": COLOR_ORANGE, "text": "BAD HEEL", "feedback": "ส้นเท้าลอย!"},
    "squat_bad_back": {"color": COLOR_RED,    "text": "BAD BACK", "feedback": "หลังงอ!"},
}
DEFAULT_CONFIG = {"color": COLOR_GRAY, "text": "DETECTING...", "feedback": ""}

SMOOTH_N       = 7     # เฉลี่ยกี่เฟรม
GOOD_THRESHOLD = 0.65  # confidence ขั้นต่ำถึงจะนับ


# ── Rep Counter ────────────────────────────────────────────────────────────────

class RepCounter:
    """
    นับครั้งแบบ state machine
    UP → DOWN (สะโพกลงต่ำกว่าเข่า) → UP = 1 rep
    """
    def __init__(self):
        self.count      = 0
        self.state      = "UP"
        self.good_count = 0
        self.bad_count  = 0
        self.last_label = None

    def update(self, landmarks, label: str, confidence: float) -> bool:
        """คืนค่า True เมื่อนับ rep ใหม่"""
        lm = landmarks.landmark

        left_hip   = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP].y
        right_hip  = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y
        left_knee  = lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y
        right_knee = lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].y

        hip_y  = (left_hip  + right_hip)  / 2
        knee_y = (left_knee + right_knee) / 2

        # DOWN = สะโพกลงต่ำกว่า 88% ของเข่า
        is_down = hip_y > knee_y * 0.88

        new_rep = False
        if self.state == "UP" and is_down:
            self.state      = "DOWN"
            self.last_label = label

        elif self.state == "DOWN" and not is_down:
            self.state = "UP"
            if confidence >= GOOD_THRESHOLD:
                self.count += 1
                new_rep = True
                if self.last_label == "squat_good":
                    self.good_count += 1
                else:
                    self.bad_count += 1

        return new_rep

    def reset(self):
        self.__init__()


# ── Draw Helpers ───────────────────────────────────────────────────────────────

def draw_rounded_rect(frame, x1, y1, x2, y2, color, radius=12, alpha=0.75):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def put_text_shadow(frame, text, pos, font, scale, color, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x+2, y+2), font, scale, COLOR_BLACK, thickness+2, cv2.LINE_AA)
    cv2.putText(frame, text, (x,   y),   font, scale, color,       thickness,   cv2.LINE_AA)


def draw_ui(frame, label, confidence, proba, classes, counter, flash):
    h, w = frame.shape[:2]
    cfg   = CLASS_CONFIG.get(label, DEFAULT_CONFIG)
    color = cfg["color"]

    # ── แถบบนสุด: สถานะ + confidence ──────────────────────────
    bar_h = 70
    draw_rounded_rect(frame, 0, 0, w, bar_h, COLOR_DARK, radius=0, alpha=0.82)

    put_text_shadow(frame, cfg["text"],
                    (16, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, color, thickness=2)

    conf_text = f"{confidence*100:.0f}%"
    (tw, _), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)
    put_text_shadow(frame, conf_text,
                    (w - tw - 16, 50), cv2.FONT_HERSHEY_DUPLEX, 1.3, color, thickness=2)

    # แถบ confidence
    cv2.rectangle(frame, (0, bar_h - 8), (w, bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, bar_h - 8), (int(w * confidence), bar_h), color, -1)

    # ── Feedback กลางจอ (เฉพาะท่าผิด + flash) ─────────────────
    if cfg["feedback"] and flash:
        fb = cfg["feedback"]
        scale = 2.0
        (fw, fh), _ = cv2.getTextSize(fb, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
        fx = (w - fw) // 2
        fy = h // 2 + 20
        draw_rounded_rect(frame, fx-20, fy-fh-16, fx+fw+20, fy+16, color, alpha=0.6)
        put_text_shadow(frame, fb, (fx, fy), cv2.FONT_HERSHEY_DUPLEX, scale, COLOR_WHITE, 3)

    # ── กล่องนับครั้ง (ขวากลาง) ───────────────────────────────
    box_w, box_h = 220, 130
    bx = w - box_w - 14
    by = bar_h + 14
    draw_rounded_rect(frame, bx, by, bx+box_w, by+box_h, COLOR_DARK, alpha=0.78)

    rep_text = str(counter.count)
    (rw, _), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_DUPLEX, 3.2, 4)
    put_text_shadow(frame, rep_text,
                    (bx + (box_w - rw)//2, by + 80),
                    cv2.FONT_HERSHEY_DUPLEX, 3.2, COLOR_WHITE, 4)
    cv2.putText(frame, "REPS", (bx + (box_w - 60)//2, by + 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_GRAY, 1, cv2.LINE_AA)

    # good / bad ใต้กล่อง reps
    gy = by + box_h + 10
    draw_rounded_rect(frame, bx, gy, bx+box_w//2-4, gy+44, COLOR_DARK, alpha=0.75)
    draw_rounded_rect(frame, bx+box_w//2+4, gy, bx+box_w, gy+44, COLOR_DARK, alpha=0.75)
    put_text_shadow(frame, f"GOOD {counter.good_count}",
                    (bx+10, gy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN)
    put_text_shadow(frame, f"BAD  {counter.bad_count}",
                    (bx+box_w//2+14, gy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ORANGE)

    # ── Probability bars (ซ้ายล่าง) ───────────────────────────
    panel_y = h - len(classes) * 38 - 50
    draw_rounded_rect(frame, 0, panel_y - 8, 320, h - 36, COLOR_DARK, alpha=0.72)

    for i, cls in enumerate(classes):
        p   = proba[0][i]
        c   = CLASS_CONFIG.get(cls, DEFAULT_CONFIG)["color"]
        lbl = CLASS_CONFIG.get(cls, DEFAULT_CONFIG)["text"]
        y   = panel_y + i * 38

        cv2.putText(frame, lbl, (12, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, c, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (12, y + 22), (300, y + 34), (60, 60, 60), -1)
        cv2.rectangle(frame, (12, y + 22), (12 + int(288 * p), y + 34), c, -1)
        cv2.putText(frame, f"{p*100:.0f}%", (305, y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    # ── แถบล่างสุด: ปุ่ม ──────────────────────────────────────
    draw_rounded_rect(frame, 0, h - 34, w, h, COLOR_DARK, radius=0, alpha=0.82)
    cv2.putText(frame, "Q: ออก    R: reset นับใหม่",
                (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_GRAY, 1, cv2.LINE_AA)


# ── Helper ─────────────────────────────────────────────────────────────────────

def landmarks_to_vector(landmarks) -> np.ndarray:
    row = []
    for lm in landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(row).reshape(1, -1)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(camera_id: int = 0):
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"ไม่สามารถเปิดกล้อง index {camera_id} ได้")

    pred_buffer = []
    counter     = RepCounter()
    flash_until = 0.0

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
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                vec   = landmarks_to_vector(results.pose_landmarks)
                proba = model.predict_proba(vec)
                idx   = int(np.argmax(proba))

                pred_buffer.append(idx)
                if len(pred_buffer) > SMOOTH_N:
                    pred_buffer.pop(0)
                smooth_idx = max(set(pred_buffer), key=pred_buffer.count)
                label      = le.inverse_transform([smooth_idx])[0]
                confidence = float(proba[0][smooth_idx])

                new_rep = counter.update(results.pose_landmarks, label, confidence)
                if new_rep and label != "squat_good":
                    flash_until = time.time() + 1.5

                draw_ui(frame, label, confidence, proba,
                        le.classes_, counter, time.time() < flash_until)

            else:
                h, w = frame.shape[:2]
                draw_rounded_rect(frame, 0, 0, w, 60, COLOR_DARK, radius=0, alpha=0.82)
                put_text_shadow(frame, "NO POSE DETECTED",
                                (16, 44), cv2.FONT_HERSHEY_DUPLEX, 1.3, COLOR_GRAY)

            cv2.imshow("Squat Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                counter.reset()
                pred_buffer.clear()
                print("[R] Reset — นับใหม่")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n=== สรุปผล ===")
    print(f"  รวม  : {counter.count} ครั้ง")
    print(f"  GOOD : {counter.good_count} ครั้ง")
    print(f"  BAD  : {counter.bad_count} ครั้ง")


if __name__ == "__main__":
    run(camera_id=0)