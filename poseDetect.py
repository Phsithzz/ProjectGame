import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a) # Hip
    b = np.array(b) # Knee
    c = np.array(c) # Ankle
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def get_real_coords(landmark, width, height):
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return (x, y)

cap = cv2.VideoCapture(0)

# ตัวแปรสำหรับการนับ
squat_count = 0
current_stage = None # เก็บสถานะปัจจุบันว่า 'up' หรือ 'down'

# ปรับ model_complexity เป็น 2 เพื่อความแม่นยำสูงสุด (ถ้าเครื่องช้าให้ลดเหลือ 1)
with mp_pose.Pose(min_detection_confidence=0.5, 
                  min_tracking_confidence=0.5, 
                  model_complexity=1) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip ภาพเพื่อให้เหมือนกระจก (optional)
        frame = cv2.flip(frame, 1)  
        height, width, _ = frame.shape
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            
            landmarks = results.pose_landmarks.landmark

            # --- แก้ไขจุดที่ 1: เลือกข้างอัตโนมัติ (Auto-Select Side) ---
            # เช็คค่า visibility (ความชัดเจน) ว่าข้างไหนเห็นชัดกว่ากัน
            left_hip_vis = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility
            right_hip_vis = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility

            if left_hip_vis > right_hip_vis:
                # ใช้ด้านซ้าย
                side = "LEFT"
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                display_coords = get_real_coords(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], width, height)
            else:
                # ใช้ด้านขวา
                side = "RIGHT"
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                display_coords = get_real_coords(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], width, height)

            # คำนวณองศา
            angle = calculate_angle(hip, knee, ankle)

            # --- แก้ไขจุดที่ 2: ปรับ Logic การนับ (Full Range Motion) ---
            # ต้องลงให้ลึกกว่า 90 องศา (Down) และต้องขึ้นให้สุดกว่า 160 องศา (Up) ถึงจะนับ 1
            if angle > 160:
                current_stage = "up"
            if angle < 90 and current_stage == 'up':
                current_stage = "down"
                squat_count += 1
                print(f"Squat Count: {squat_count}")

            # แสดงผลมุมที่หัวเข่า
            cv2.putText(frame, str(int(angle)), 
                        display_coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # แสดงสถานะว่าจับด้านไหนอยู่
            cv2.putText(frame, f"Side: {side}", (width - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # UI แสดงผล
        cv2.rectangle(frame, (0,0), (225,73), (245,117,16), -1)
        
        cv2.putText(frame, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(squat_count), (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, 'STAGE', (100,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(current_stage), (100,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Squat Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()