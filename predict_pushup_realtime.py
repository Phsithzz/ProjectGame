"""
predict_pushup_realtime.py
==========================
ตรวจจับท่า pushup แบบ real-time ด้วย MediaPipe + ML model
- นับ rep อัตโนมัติจากมุมข้อศอก
- แสดง feedback เมื่อทำผิดท่า
- pushup_bad_back / pushup_bad_neck → นับ rep แต่แจ้งเตือน

วางไฟล์ในโฟลเดอร์เดียวกัน:
    predict_pushup_realtime.py
    pushup_model.pkl
    pushup_label_encoder.pkl

รัน:
    python predict_pushup_realtime.py

ปุ่ม:
    Q — ออก
    R — reset นับใหม่
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# ── โหลด model ────────────────────────────────────────────────────────────────

model = joblib.load("pushup_model.pkl")
le    = joblib.load("pushup_label_encoder.pkl")

print(f"✓ โหลด model สำเร็จ")
print(f"  Classes: {le.classes_}")

# ── Config ─────────────────────────────────────────────────────────────────────

COLOR_GREEN  = (60,  210,  80)
COLOR_ORANGE = (0,   165, 255)
COLOR_RED    = (60,   60, 220)
COLOR_WHITE  = (240, 240, 240)
COLOR_GRAY   = (130, 130, 130)
COLOR_DARK   = (25,   25,  25)
COLOR_BLACK  = (0,     0,   0)
COLOR_YELLOW = (30,   220, 220)

CLASS_CONFIG = {
    "pushup_good":     {"color": COLOR_GREEN,  "text": "GOOD",     "feedback": ""},
    "pushup_bad_neck": {"color": COLOR_ORANGE, "text": "BAD NECK", "feedback": "คอก้มเกิน!"},
    "pushup_bad_back": {"color": COLOR_RED,    "text": "BAD BACK", "feedback": "หลังแอ่น!"},
}
DEFAULT_CONFIG = {"color": COLOR_GRAY, "text": "DETECTING...", "feedback": ""}

SMOOTH_N        = 7
GOOD_THRESHOLD  = 0.60
ELBOW_DOWN_DEG  = 100   # มุมข้อศอก <= นี้ = ลงแล้ว (DOWN)
ELBOW_UP_DEG    = 155   # มุมข้อศอก >= นี้ = ขึ้นแล้ว (UP)

# landmark index
IDX = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,      "RIGHT_HIP": 24,
}


# ── Rep Counter ────────────────────────────────────────────────────────────────

class RepCounter:
    """
    นับจาก elbow angle:
    UP (ข้อศอกเหยียด) → DOWN (ข้อศอกงอ) → UP = 1 rep
    จดไว้ว่า DOWN ด้วย label อะไร → บันทึก good/bad ตอน UP
    """
    def __init__(self):
        self.count      = 0
        self.state      = "UP"
        self.good_count = 0
        self.bad_count  = 0
        self.last_label = None

    def update(self, landmarks, label: str, confidence: float) -> bool:
        lm    = landmarks.landmark
        angle = self._elbow_angle(lm)
        if angle is None:
            return False

        new_rep = False

        if self.state == "UP" and angle <= ELBOW_DOWN_DEG:
            self.state      = "DOWN"
            self.last_label = label

        elif self.state == "DOWN" and angle >= ELBOW_UP_DEG:
            self.state = "UP"
            if confidence >= GOOD_THRESHOLD:
                self.count += 1
                new_rep = True
                if self.last_label == "pushup_good":
                    self.good_count += 1
                else:
                    self.bad_count += 1

        return new_rep

    def _elbow_angle(self, lm) -> float | None:
        """มุมข้อศอกซ้าย (wrist → elbow → shoulder)"""
        try:
            w = np.array([lm[IDX["LEFT_WRIST"]].x,    lm[IDX["LEFT_WRIST"]].y])
            e = np.array([lm[IDX["LEFT_ELBOW"]].x,    lm[IDX["LEFT_ELBOW"]].y])
            s = np.array([lm[IDX["LEFT_SHOULDER"]].x, lm[IDX["LEFT_SHOULDER"]].y])
            ew = w - e
            es = s - e
            cos_a = np.dot(ew, es) / (np.linalg.norm(ew) * np.linalg.norm(es) + 1e-8)
            return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
        except Exception:
            return None

    def reset(self):
        self.__init__()


# ── Draw Helpers ───────────────────────────────────────────────────────────────

def draw_rounded_rect(frame, x1, y1, x2, y2, color, radius=12, alpha=0.75):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def put_text_shadow(frame, text, pos, font, scale, color, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x+2, y+2), font, scale, COLOR_BLACK, thickness+2, cv2.LINE_AA)
    cv2.putText(frame, text, (x,   y),   font, scale, color,       thickness,   cv2.LINE_AA)


def draw_elbow_angle(frame, landmarks, angle: float):
    """วาดมุมข้อศอกบนภาพ"""
    lm = landmarks.landmark
    h, w = frame.shape[:2]

    ex = int(lm[IDX["LEFT_ELBOW"]].x * w)
    ey = int(lm[IDX["LEFT_ELBOW"]].y * h)

    color = COLOR_GREEN if angle >= ELBOW_UP_DEG else (
            COLOR_ORANGE if angle <= ELBOW_DOWN_DEG else COLOR_YELLOW)

    cv2.circle(frame, (ex, ey), 14, color, -1)
    put_text_shadow(frame, f"{angle:.0f}",
                    (ex - 16, ey + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)


def draw_ui(frame, label, confidence, proba, classes,
            counter: RepCounter, flash_text: str, elbow_angle: float):
    h, w = frame.shape[:2]
    cfg   = CLASS_CONFIG.get(label, DEFAULT_CONFIG)
    color = cfg["color"]

    # ── แถบบนสุด ──────────────────────────────────────────
    draw_rounded_rect(frame, 0, 0, w, 70, COLOR_DARK, radius=0, alpha=0.82)
    put_text_shadow(frame, cfg["text"],
                    (16, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 2)

    conf_text = f"{confidence*100:.0f}%"
    (tw, _), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)
    put_text_shadow(frame, conf_text,
                    (w-tw-16, 50), cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 2)

    cv2.rectangle(frame, (0, 62), (w, 70), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, 62), (int(w*confidence), 70), color, -1)

    # ── Elbow angle indicator ──────────────────────────────
    if elbow_angle is not None:
        a_color = COLOR_GREEN if elbow_angle >= ELBOW_UP_DEG else (
                  COLOR_RED if elbow_angle <= ELBOW_DOWN_DEG else COLOR_YELLOW)
        state_text = "UP" if elbow_angle >= ELBOW_UP_DEG else (
                     "DOWN" if elbow_angle <= ELBOW_DOWN_DEG else "MID")
        put_text_shadow(frame, f"ELBOW {elbow_angle:.0f} ({state_text})",
                        (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, a_color, 1)

    # ── Feedback กลางจอ ────────────────────────────────────
    if flash_text:
        scale = 2.0
        (fw, fh), _ = cv2.getTextSize(flash_text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
        fx = (w - fw) // 2
        fy = h // 2 + 20
        draw_rounded_rect(frame, fx-20, fy-fh-16, fx+fw+20, fy+16, color, alpha=0.6)
        put_text_shadow(frame, flash_text,
                        (fx, fy), cv2.FONT_HERSHEY_DUPLEX, scale, COLOR_WHITE, 3)

    # ── กล่องนับครั้ง (ขวากลาง) ───────────────────────────
    box_w, box_h = 220, 130
    bx = w - box_w - 14
    by = 84
    draw_rounded_rect(frame, bx, by, bx+box_w, by+box_h, COLOR_DARK, alpha=0.78)

    rep_text = str(counter.count)
    (rw, _), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_DUPLEX, 3.2, 4)
    put_text_shadow(frame, rep_text,
                    (bx+(box_w-rw)//2, by+80), cv2.FONT_HERSHEY_DUPLEX, 3.2, COLOR_WHITE, 4)
    cv2.putText(frame, "REPS", (bx+(box_w-60)//2, by+108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_GRAY, 1, cv2.LINE_AA)

    # good / bad
    gy = by + box_h + 10
    draw_rounded_rect(frame, bx, gy, bx+box_w//2-4, gy+44, COLOR_DARK, alpha=0.75)
    draw_rounded_rect(frame, bx+box_w//2+4, gy, bx+box_w, gy+44, COLOR_DARK, alpha=0.75)
    put_text_shadow(frame, f"GOOD {counter.good_count}",
                    (bx+10, gy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN)
    put_text_shadow(frame, f"BAD  {counter.bad_count}",
                    (bx+box_w//2+14, gy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ORANGE)

    # ── Probability bars ───────────────────────────────────
    panel_y = h - len(classes) * 38 - 50
    draw_rounded_rect(frame, 0, panel_y-8, 320, h-36, COLOR_DARK, alpha=0.72)

    for i, cls in enumerate(classes):
        p   = proba[0][i]
        c   = CLASS_CONFIG.get(cls, DEFAULT_CONFIG)["color"]
        lbl = CLASS_CONFIG.get(cls, DEFAULT_CONFIG)["text"]
        y   = panel_y + i * 38

        cv2.putText(frame, lbl, (12, y+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, c, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (12, y+22), (300, y+34), (60, 60, 60), -1)
        cv2.rectangle(frame, (12, y+22), (12+int(288*p), y+34), c, -1)
        cv2.putText(frame, f"{p*100:.0f}%", (305, y+34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    # ── แถบล่างสุด ─────────────────────────────────────────
    draw_rounded_rect(frame, 0, h-34, w, h, COLOR_DARK, radius=0, alpha=0.82)
    cv2.putText(frame, "Q: ออก    R: reset นับใหม่",
                (12, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_GRAY, 1, cv2.LINE_AA)


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
    flash_text  = ""

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

            label       = None
            confidence  = 0.0
            proba       = np.zeros((1, len(le.classes_)))
            elbow_angle = None

            if results.pose_landmarks:
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

                pred_buffer.append(idx)
                if len(pred_buffer) > SMOOTH_N:
                    pred_buffer.pop(0)
                smooth_idx  = max(set(pred_buffer), key=pred_buffer.count)
                label       = le.inverse_transform([smooth_idx])[0]
                confidence  = float(proba[0][smooth_idx])

                # คำนวณมุมข้อศอก
                elbow_angle = counter._elbow_angle(results.pose_landmarks.landmark)

                # วาดมุมบนภาพ
                if elbow_angle is not None:
                    draw_elbow_angle(frame, results.pose_landmarks, elbow_angle)

                # นับ rep
                new_rep = counter.update(results.pose_landmarks, label, confidence)

                # flash feedback เมื่อท่าผิดและนับ rep
                if new_rep and label != "pushup_good":
                    fb = CLASS_CONFIG.get(label, DEFAULT_CONFIG)["feedback"]
                    if fb:
                        flash_until = time.time() + 1.5
                        flash_text  = fb

                current_flash = flash_text if time.time() < flash_until else ""
                draw_ui(frame, label, confidence, proba,
                        le.classes_, counter, current_flash, elbow_angle)

            else:
                h, w = frame.shape[:2]
                draw_rounded_rect(frame, 0, 0, w, 60, COLOR_DARK, radius=0, alpha=0.82)
                put_text_shadow(frame, "NO POSE DETECTED",
                                (16, 44), cv2.FONT_HERSHEY_DUPLEX, 1.3, COLOR_GRAY)

            cv2.imshow("Pushup Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                counter.reset()
                pred_buffer.clear()
                flash_until = 0
                flash_text  = ""
                print("[R] Reset — นับใหม่")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n=== สรุปผล ===")
    print(f"  รวม  : {counter.count} ครั้ง")
    print(f"  GOOD : {counter.good_count} ครั้ง")
    print(f"  BAD  : {counter.bad_count} ครั้ง")


if __name__ == "__main__":
    run(camera_id=0)
