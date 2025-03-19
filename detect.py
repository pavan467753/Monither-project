import os
import cv2
import numpy as np
import time
import pygame
from collections import Counter, deque

# Initialize pygame for sound alerts **only if not running inside Docker**
if not os.environ.get("DOCKER"):
    pygame.mixer.init()
    BUZZER_SOUND = "alert.mp3"  # Ensure this file is in the working directory
else:
    BUZZER_SOUND = None  # No sound in Docker

# Memory for gender smoothing
person_memory = {}
HISTORY_LENGTH = 10  # Frames for smoothing

# Memory for gender detection delay
last_gender_update_time = {}
last_detected_gender = {}
GENDER_UPDATE_DELAY = 5  # 5 seconds delay before changing gender prediction

# Buzzer timing memory
last_alert_time = 0  # Stores the last time the buzzer was played
ALERT_INTERVAL = 60  # Play buzzer only once every 60 seconds
BUZZER_DURATION = 5  # Play buzzer for 5 seconds

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

bodyProto = "MobileNetSSD_deploy.prototxt"
bodyModel = "MobileNetSSD_deploy.caffemodel"
bodyNet = cv2.dnn.readNetFromCaffe(bodyProto, bodyModel)

def apply_night_vision(frame):
    """Apply night vision effect if the input video is dark."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 50:  # Threshold for darkness
        return cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    return frame

def smooth_gender(memory, person_id, detected_gender):
    """Apply majority voting to smooth gender predictions."""
    if person_id not in memory:
        memory[person_id] = deque(maxlen=HISTORY_LENGTH)
    
    memory[person_id].append(detected_gender)
    gender_counts = Counter(memory[person_id])
    return gender_counts.most_common(1)[0][0]

def detect_faces(frame):
    """Detect faces and return bounding boxes."""
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append((x1, y1, x2, y2))

    return faceBoxes

def detect_bodies(frame):
    """Detect bodies and return bounding boxes for all people."""
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    
    bodyNet.setInput(blob)
    detections = bodyNet.forward()
    bodyBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        label = int(detections[0, 0, i, 1])
        
        if confidence > 0.4 and label == 15:  # Class label 15 is 'person'
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bodyBoxes.append((x1, y1, x2, y2))
    
    return bodyBoxes

def predict_gender(region):
    """Predict gender using the gender classification model."""
    blob = cv2.dnn.blobFromImage(region, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    return genderList[genderPreds[0].argmax()]

def detect_faces_and_bodies(frame):
    """Detect faces and bodies, predict gender, apply DBSCAN for clustering, and trigger alerts."""
    global last_alert_time
    frame = apply_night_vision(frame)
    resultImg = frame.copy()
    faceBoxes = detect_faces(frame)
    bodyBoxes = detect_bodies(frame)
    gender_count = Counter()
    detected_people = []
    
    male_positions = []
    female_positions = []

    for bodyBox in bodyBoxes:
        bx1, by1, bx2, by2 = bodyBox
        body_region = frame[by1:by2, bx1:bx2]
        
        if body_region.shape[0] > 0 and body_region.shape[1] > 0:
            person_id = tuple(bodyBox)
            detected_gender = predict_gender(body_region)
            
            current_time = time.time()
            if person_id not in last_detected_gender or (current_time - last_gender_update_time.get(person_id, 0)) >= GENDER_UPDATE_DELAY:
                last_detected_gender[person_id] = detected_gender
                last_gender_update_time[person_id] = current_time
            
            smoothed_gender = smooth_gender(person_memory, person_id, last_detected_gender[person_id])
            gender_count[smoothed_gender] += 1
            
            color = (0, 255, 0) if smoothed_gender == 'Male' else (255, 0, 255)
            cv2.putText(resultImg, f'{smoothed_gender}', (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(resultImg, (bx1, by1), (bx2, by2), color, 2)

    return resultImg
