import cv2
import numpy as np
from ultralytics import YOLO

# โหลดโมเดล YOLOv8-Pose (nano model เร็วสุด)
model = YOLO('yolov8n-pose.pt')

def calculate_angle(a, b, c):
    """
    คำนวณมุมระหว่าง 3 จุด (a=Hip, b=Knee, c=Ankle)
    รับค่าเป็นพิกัด (x, y) โดยตรง
    """
    a = np.array(a) # Hip
    b = np.array(b) # Knee
    c = np.array(c) # Ankle
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

cap = cv2.VideoCapture(0)

# ตัวแปรสำหรับการนับ
squat_count = 0
current_stage = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip ภาพ (Optional)
    # frame = cv2.flip(frame, 1)
    
    # --- ใช้ YOLO ทำนายผล ---
    # verbose=False เพื่อไม่ให้รก Terminal
    results = model(frame, verbose=False) 
    
    # ดึงภาพที่ YOLO วาดโครงกระดูกมาให้แล้ว (สะดวกกว่าวาดเอง)
    annotated_frame = results[0].plot()

    # ดึงข้อมูล Keypoints
    # results[0].keypoints.data อยู่ในรูป Tensor (Batch, Points, 3) -> (x, y, conf)
    keypoints = results[0].keypoints.data
    
    # เช็คว่าเจอคนไหม (ถ้าเจอ keypoints จะไม่ว่าง)
    if keypoints.shape[1] > 0:
        # เลือกคนแรกที่เจอ (Index 0)
        person = keypoints[0].cpu().numpy()

        # --- Mapping จุดของ YOLOv8 (COCO format) ---
        # 11: Left Hip, 12: Right Hip
        # 13: Left Knee, 14: Right Knee
        # 15: Left Ankle, 16: Right Ankle
        
        # ดึงค่า Confidence ของสะโพกเพื่อเลือกข้าง
        left_hip_conf = person[11][2]
        right_hip_conf = person[12][2]
        
        side = ""
        hip, knee, ankle = [], [], []
        display_coords = (0,0)

        # --- Logic เลือกข้างอัตโนมัติ (Auto-Select Side) ---
        if left_hip_conf > right_hip_conf:
            side = "LEFT"
            hip = person[11][:2]   # เอาแค่ x, y
            knee = person[13][:2]
            ankle = person[15][:2]
            # พิกัดสำหรับแสดงตัวเลขมุม (เอาตำแหน่งเข่า)
            display_coords = (int(knee[0]), int(knee[1]))
        else:
            side = "RIGHT"
            hip = person[12][:2]
            knee = person[14][:2]
            ankle = person[16][:2]
            display_coords = (int(knee[0]), int(knee[1]))

        # ต้องเช็คว่าค่าความมั่นใจของจุดหลักๆ สูงพอไหม (กันค่านั่วๆ ตอนไม่เห็นขา)
        if hip[0] != 0 and knee[0] != 0 and ankle[0] != 0:
            
            # คำนวณองศา
            angle = calculate_angle(hip, knee, ankle)
            
            # --- Logic การนับ (เหมือนเดิม) ---
            if angle > 160:
                current_stage = "up"
            if angle < 90 and current_stage == 'up':
                current_stage = "down"
                squat_count += 1
                print(f"Squat Count: {squat_count}")

            # แสดงผลมุมที่หัวเข่า
            cv2.putText(annotated_frame, str(int(angle)), 
                        display_coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # แสดงสถานะว่าจับด้านไหนอยู่
            cv2.putText(annotated_frame, f"Side: {side}", (annotated_frame.shape[1] - 200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- UI แสดงผล ---
    # ใช้ annotated_frame แทน frame เดิม เพราะมีเส้น Skeleton แล้ว
    cv2.rectangle(annotated_frame, (0,0), (225,73), (245,117,16), -1)
    
    cv2.putText(annotated_frame, 'REPS', (15,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(annotated_frame, str(squat_count), (10,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.putText(annotated_frame, 'STAGE', (100,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(annotated_frame, str(current_stage), (100,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Squat Detection (YOLOv8)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()