import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE
LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
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


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


squat_count = 0
squat_down = False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        status = "Stand"

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            landmarks = results.pose_landmarks.landmark

     
            hip = [landmarks[LEFT_HIP].x,
                    landmarks[LEFT_HIP].y]
            knee = [landmarks[LEFT_KNEE].x,
                    landmarks[LEFT_KNEE].y]
            ankle = [landmarks[LEFT_ANKLE].x,
                    landmarks[LEFT_ANKLE].y]

  
            angle = calculate_angle(hip, knee, ankle)

       
            if angle < 90:
                status = "Squat Down"
                squat_down = True
            else:
                status = "Stand Up"
                if squat_down:  
                    squat_count += 1
                    squat_down = False

        
            knee_coords = get_real_coords(landmarks[mp_pose.PoseLandmark.LEFT_KNEE], width, height)
            cv2.putText(frame, str(int(angle)), 
                        knee_coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

      
        cv2.putText(frame, status, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    
        cv2.putText(frame, f"Count: {squat_count}", (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

 
        cv2.imshow("Squat Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()




