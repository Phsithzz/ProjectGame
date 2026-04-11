"""
predict_YOLO.py (ใช้งาน squat_model_yolo.pkl)
==============================================
classes: squat_good, squat_bad_heel, squat_bad_back, squat_bad_foot

- squat_bad_foot → ไม่นับ rep + แสดง feedback "เท้าไม่ติดพื้น!"
- กด Q: ออก  |  R: reset นับใหม่
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# ── โหลด model ────────────────────────────────────────────────────────────────

model = joblib.load("squat_model_yolo.pkl")
le    = joblib.load("label_encoder_yolo.pkl")

print(f"✓ โหลด model สำเร็จ")
print(f"  Classes: {le.classes_}")

# ── Config ─────────────────────────────────────────────────────────────────────

COLOR_GREEN  = (60,  210,  80)
COLOR_ORANGE = (0,   165, 255)
COLOR_RED    = (60,   60, 220)
COLOR_PURPLE = (200,  80, 200)
COLOR_WHITE  = (240, 240, 240)
COLOR_GRAY   = (130, 130, 130)
COLOR_DARK   = (25,   25,  25)
COLOR_BLACK  = (0,     0,   0)

CLASS_CONFIG = {
    "squat_good":     {"color": COLOR_GREEN,  "text": "GOOD",           "feedback": "",                  "count_rep": True},
    "squat_bad_kneel": {"color": COLOR_ORANGE, "text": "BAD KNEEL",      "feedback": "หัวเข่าเข้า!",      "count_rep": True},
    "squat_bad_back": {"color": COLOR_RED,    "text": "BAD BACK",       "feedback": "หลังงอ!",           "count_rep": True},
    "squat_bad_foot": {"color": COLOR_PURPLE, "text": "BAD FOOT",       "feedback": "เท้าไม่ติดพื้น!",   "count_rep": False},
}
DEFAULT_CONFIG = {"color": COLOR_GRAY, "text": "DETECTING...", "feedback": "", "count_rep": False}

SMOOTH_N       = 7
GOOD_THRESHOLD = 0.65

# ── Rep Counter ────────────────────────────────────────────────────────────────

class RepCounter:
    """
    UP → DOWN → UP = 1 rep
    ถ้า label เป็น bad_foot จะไม่นับ rep เลย
    """
    def __init__(self):
        self.count      = 0
        self.state      = "UP"
        self.good_count = 0
        self.bad_count  = 0
        self.last_label = None

    def update(self, landmarks, label: str, confidence: float) -> bool:
        lm = landmarks.landmark

        left_hip   = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP].y
        right_hip  = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y
        left_knee  = lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y
        right_knee = lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].y

        hip_y  = (left_hip  + right_hip)  / 2
        knee_y = (left_knee + right_knee) / 2
        is_down = hip_y > knee_y * 0.88

        new_rep = False

        if self.state == "UP" and is_down:
            self.state      = "DOWN"
            self.last_label = label

        elif self.state == "DOWN" and not is_down:
            self.state = "UP"

            cfg = CLASS_CONFIG.get(self.last_label, DEFAULT_CONFIG)

            # ── จุดสำคัญ: bad_foot ไม่นับ rep ──────────────
            if not cfg["count_rep"]:
                pass  # ข้ามการนับ

            elif confidence >= GOOD_THRESHOLD:
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


def draw_ui(frame, label, confidence, proba, classes, counter, flash_text):
    h, w = frame.shape[:2]
    cfg   = CLASS_CONFIG.get(label, DEFAULT_CONFIG)
    color = cfg["color"]

    # ── แถบบนสุด ──────────────────────────────────────────
    draw_rounded_rect(frame, 0, 0, w, 70, COLOR_DARK, radius=0, alpha=0.82)
    put_text_shadow(frame, cfg["text"],
                    (16, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 2)

    # ถ้า bad_foot แสดง "ไม่นับ" ต่อท้าย
    if not cfg["count_rep"] and label != None:
        put_text_shadow(frame, "(ไม่นับ)",
                        (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_PURPLE, 1)

    conf_text = f"{confidence*100:.0f}%"
    (tw, _), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)
    put_text_shadow(frame, conf_text,
                    (w-tw-16, 50), cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 2)

    cv2.rectangle(frame, (0, 62), (w, 70), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, 62), (int(w*confidence), 70), color, -1)

    # ── Feedback กลางจอ ────────────────────────────────────
    if flash_text:
        scale = 2.0
        (fw, fh), _ = cv2.getTextSize(flash_text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
        fx = (w - fw) // 2
        fy = h // 2 + 20
        draw_rounded_rect(frame, fx-20, fy-fh-16, fx+fw+20, fy+16, color, alpha=0.6)
        put_text_shadow(frame, flash_text, (fx, fy), cv2.FONT_HERSHEY_DUPLEX, scale, COLOR_WHITE, 3)

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

    # good / bad ใต้กล่อง
    gy = by + box_h + 10
    draw_rounded_rect(frame, bx, gy, bx+box_w//2-4, gy+44, COLOR_DARK, alpha=0.75)
    draw_rounded_rect(frame, bx+box_w//2+4, gy, bx+box_w, gy+44, COLOR_DARK, alpha=0.75)
    put_text_shadow(frame, f"GOOD {counter.good_count}",
                    (bx+10, gy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN)
    put_text_shadow(frame, f"BAD  {counter.bad_count}",
                    (bx+box_w//2+14, gy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ORANGE)

    # ── Probability bars (ซ้ายล่าง) ───────────────────────
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
    """
    Extract first 17 landmarks with x, y, z (without visibility)
    17 landmarks * 3 coordinates = 51 features (as expected by squat_model_yolo.pkl)
    """
    row = []
    for i, lm in enumerate(landmarks.landmark):
        if i >= 17:  # Only use first 17 landmarks
            break
        row.extend([lm.x, lm.y, lm.z])
    return np.array(row).reshape(1, -1)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(camera_id: int = 0):
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"ไม่สามารถเปิดกล้อง index {camera_id} ได้")

    pred_buffer  = []
    counter      = RepCounter()
    flash_until  = 0.0
    flash_text   = ""

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

                # flash feedback
                if new_rep or label == "squat_bad_foot":
                    fb = CLASS_CONFIG.get(label, DEFAULT_CONFIG)["feedback"]
                    if fb:
                        flash_until = time.time() + 1.5
                        flash_text  = fb

                current_flash = flash_text if time.time() < flash_until else ""
                draw_ui(frame, label, confidence, proba,
                        le.classes_, counter, current_flash)

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
