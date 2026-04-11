"""
predict_realtime_plank.py 
=============================================
- เปลี่ยนจากระบบ "นับครั้ง" เป็น "จับเวลา (วินาที)" สำหรับท่า Plank
- จับเวลาสะสมเฉพาะตอนที่ทำท่าถูกต้อง (plank_good)
- กด Q: ออก  |  R: reset เวลาใหม่
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# ── โหลด model ────────────────────────────────────────────────────────────────

# เปลี่ยนมาโหลดไฟล์ของท่า Plank
model = joblib.load("plank_model.pkl")
le    = joblib.load("plank_label_encoder.pkl")

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

# *** สำคัญ: แก้ไขชื่อคีย์เวิร์ดด้านซ้ายมือ ให้ตรงกับ class ที่คุณ train มา ***
CLASS_CONFIG = {
    "plank_good":     {"color": COLOR_GREEN,  "text": "GOOD",     "feedback": ""},
    "plank_bad_back": {"color": COLOR_RED,    "text": "BAD BACK", "feedback": "หลังแอ่น / ก้นตก!"},
    "plank_bad_hip":  {"color": COLOR_ORANGE, "text": "HIGH HIP", "feedback": "ก้นโด่งเกินไป!"},
}
DEFAULT_CONFIG = {"color": COLOR_GRAY, "text": "DETECTING...", "feedback": ""}

SMOOTH_N       = 7
GOOD_THRESHOLD = 0.65

# ── Hold Timer (เปลี่ยนจาก Rep Counter) ────────────────────────────────────────

class HoldTimer:
    """
    จับเวลาสะสมเฉพาะตอนที่ AI ตรวจเจอท่า 'plank_good'
    """
    def __init__(self):
        self.total_time = 0.0
        self.start_time = None
        self.is_holding = False

    def update(self, label: str, confidence: float):
        current_time = time.time()
        
        # ถ้าระบุว่าเป็นท่าที่ถูกต้อง และมีความมั่นใจสูงพอ
        if label == "plank_good" and confidence >= GOOD_THRESHOLD:
            if not self.is_holding:
                # เริ่มจับเวลา
                self.is_holding = True
                self.start_time = current_time
        else:
            if self.is_holding:
                # หยุดทำท่า good, เก็บเวลาสะสมเข้า total_time
                self.total_time += (current_time - self.start_time)
                self.is_holding = False
                self.start_time = None

    def get_current_time(self):
        # คำนวณเวลาที่จะแสดงผลบนจอแบบ Real-time
        if self.is_holding and self.start_time:
            return self.total_time + (time.time() - self.start_time)
        return self.total_time

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


def draw_ui(frame, label, confidence, proba, classes, timer, flash_text):
    h, w = frame.shape[:2]
    cfg  = CLASS_CONFIG.get(label, DEFAULT_CONFIG)
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

    # ── Feedback กลางจอ ────────────────────────────────────
    if flash_text:
        scale = 2.0
        (fw, fh), _ = cv2.getTextSize(flash_text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
        fx = (w - fw) // 2
        fy = h // 2 + 20
        draw_rounded_rect(frame, fx-20, fy-fh-16, fx+fw+20, fy+16, color, alpha=0.6)
        put_text_shadow(frame, flash_text, (fx, fy), cv2.FONT_HERSHEY_DUPLEX, scale, COLOR_WHITE, 3)

    # ── กล่องจับเวลา (ขวากลาง) ───────────────────────────
    box_w, box_h = 240, 130
    bx = w - box_w - 14
    by = 84
    draw_rounded_rect(frame, bx, by, bx+box_w, by+box_h, COLOR_DARK, alpha=0.78)

    # แสดงเวลาสะสมเป็นทศนิยม 1 ตำแหน่ง
    time_text = f"{timer.get_current_time():.1f}"
    (rw, _), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_DUPLEX, 3.2, 4)
    put_text_shadow(frame, time_text,
                    (bx+(box_w-rw)//2, by+80), cv2.FONT_HERSHEY_DUPLEX, 3.2, COLOR_WHITE, 4)
    cv2.putText(frame, "SECONDS (GOOD FORM)", (bx+(box_w-150)//2, by+108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1, cv2.LINE_AA)

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
    cv2.putText(frame, "Q: ออก    R: reset จับเวลาใหม่",
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

    pred_buffer  = []
    timer        = HoldTimer()
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

                # อัปเดตตัวจับเวลา
                timer.update(label, confidence)

                # Flash feedback เมื่อท่าผิด (เช็คจาก config ว่ามีข้อความ feedback หรือไม่)
                fb = CLASS_CONFIG.get(label, DEFAULT_CONFIG).get("feedback", "")
                if fb and confidence >= GOOD_THRESHOLD:
                    flash_until = time.time() + 1.0 # โชว์แจ้งเตือน 1 วินาที
                    flash_text  = fb

                current_flash = flash_text if time.time() < flash_until else ""
                draw_ui(frame, label, confidence, proba,
                        le.classes_, timer, current_flash)

            else:
                h, w = frame.shape[:2]
                draw_rounded_rect(frame, 0, 0, w, 60, COLOR_DARK, radius=0, alpha=0.82)
                put_text_shadow(frame, "NO POSE DETECTED",
                                (16, 44), cv2.FONT_HERSHEY_DUPLEX, 1.3, COLOR_GRAY)

            cv2.imshow("Plank Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                timer.reset()
                pred_buffer.clear()
                flash_until = 0
                flash_text  = ""
                print("[R] Reset — จับเวลาใหม่")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n=== สรุปผล ===")
    print(f"  เวลาทำท่าที่ถูกต้องรวม : {timer.get_current_time():.2f} วินาที")


if __name__ == "__main__":
    run(camera_id=0)