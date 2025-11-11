import mediapipe as mp
import math
import cv2
import numpy as np
import json

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 

## to_px - convert pose landmarks to coordinate system with provided width and height
def to_px(lm, w , h):
    return np.array([lm.x * w, lm.y * h], dtype=float)

## angle_between - give the angle between two vectors.
def angle_between(v1, v2):
    v1 = np.asarray(v1, dtype = float)
    v2 = np.asarray(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    c = np.clip(np.dot(v1,v2)/(n1 * n2), -1.0,1.0)
    return math.degrees(math.acos(c))

## angle_3pt - give the angle of two line segments with b as the point connecting both
def angle_3pt(a,b,c):
    return angle_between(a-b,c-b)

# Start opencv video capture
cap = cv2.VideoCapture(0)

# Begin Pose Tracking
with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.1) as pose:
    # Continuous frame updating
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert colorspace to work with landmark processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            h, w, _ = image.shape
            lms = results.pose_landmarks.landmark

            # landmark indices
            NOSE = 0                        # nose (head)
            L_SH, L_EL, L_WR = 11, 13, 15   # left shoulder, elbow, wrist
            R_SH, R_EL, R_WR = 12, 14, 16   # right shoulder, elbow, wrist
            L_HP, L_KN, L_AN = 23, 25, 27   # left hip, knee, ankle
            R_HP, R_KN, R_AN = 24, 26, 28   # right hip, knee ankle
            
            # head
            head = to_px(lms[NOSE], w, h)

            # left arm
            L_sh = to_px(lms[L_SH], w, h)
            L_el = to_px(lms[L_EL], w, h)
            L_wr = to_px(lms[L_WR], w, h)

            # right arm
            R_sh = to_px(lms[R_SH], w, h)
            R_el = to_px(lms[R_EL], w, h)
            R_wr = to_px(lms[R_WR], w, h)

            # left leg
            L_hp = to_px(lms[L_HP], w, h)
            L_kn = to_px(lms[L_KN], w, h)
            L_an = to_px(lms[L_AN], w, h)

            # right leg
            R_hp = to_px(lms[R_HP], w, h)
            R_kn = to_px(lms[R_KN], w, h)
            R_an = to_px(lms[R_AN], w, h)

            # compute angles
            L_angle = angle_3pt(L_sh, L_el, L_wr)
            R_angle = angle_3pt(R_sh, R_el, R_wr)

            # Display angles in window
            if L_angle:
                cv2.putText(image, f"L: {L_angle:.1f}°", (int(L_el[0]+20), int(L_el[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            if R_angle:
                cv2.putText(image, f"R: {R_angle:.1f}°", (int(R_el[0]+20), int(R_el[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            
            # Covnert to JSON string
            # Make sure to change the relative path to the correct position data.
            path = "./position_data.json"
            json_obj = {
                "headPosition":{
                    "x":head[0],
                    "y":h - head[1],
                    "z":float(0)
                },
                "rightShoulderPosition":{
                    "x":R_sh[0],
                    "y":h - R_sh[1],
                    "z":float(0)
                },
                "leftShoulderPosition":{
                    "x":L_sh[0],
                    "y":h - L_sh[1],
                    "z":float(0)
                },
                "rightElbowPosition":{
                    "x":R_el[0],
                    "y":h - R_el[1],
                    "z":float(0)
                },
                "leftElbowPosition":{
                    "x":L_el[0],
                    "y":h - L_el[1],
                    "z":float(0)
                },
                "rightWristPosition":{
                    "x":R_wr[0],
                    "y":h - R_wr[1],
                    "z":float(0)
                },
                "leftWristPosition":{
                    "x":L_wr[0],
                    "y":h - L_wr[1],
                    "z":float(0)
                },
                "rightHipPosition":{
                    "x":R_hp[0],
                    "y":h - R_hp[1],
                    "z":float(0)
                },
                "leftHipPosition":{
                    "x":L_hp[0],
                    "y":h - L_hp[1],
                    "z":float(0)
                },
                "rightKneePosition":{
                    "x":R_kn[0],
                    "y":h - R_kn[1],
                    "z":float(0)
                },
                "leftKneePosition":{
                    "x":L_kn[0],
                    "y":h - L_kn[1],
                    "z":float(0)
                },
                "rightAnklePosition":{
                    "x":R_an[0],
                    "y":h - R_an[1],
                    "z":float(0)
                },
                "leftAnklePosition":{
                    "x":L_an[0],
                    "y":h - L_an[1],
                    "z":float(0)
                }
            }
            json_str = json.dumps(json_obj)

            # Try-except to prevent simultaneous file access.
            try:
                with open(path, "w") as f:
                    f.write(json_str)
            except:
                pass     

        # Update window        
        cv2.imshow('Elbow Angle Display', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Cleanup        
cap.release()
cv2.destroyAllWindows()
