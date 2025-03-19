from flask import Flask, render_template, request, Response, redirect, url_for, flash
import os
import cv2
import csv
import copy
import argparse
import numpy as np
import mediapipe as mp
import pygame
from collections import deque
import time
from utils import CvFpsCalc
from model.keypoint_classifier import KeyPointClassifier
from detect import detect_faces_and_bodies, detect_bodies  # Import body detection

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flash messages

# Initialize pygame only if not running inside Docker
if not os.environ.get("DOCKER"):
    pygame.mixer.init()
    BUZZER_SOUND = "alert.mp3"  # Path to the buzzer sound file
else:
    BUZZER_SOUND = None  # No sound in Docker

def play_buzzer():
    if BUZZER_SOUND:
        pygame.mixer.music.load(BUZZER_SOUND)
        pygame.mixer.music.play()

# Global variables for video stream and state
cap = None
stream_active = False

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', streaming=stream_active)

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Start the video stream."""
    global cap, stream_active

    ip_address = request.form.get('ip_address')
    if not ip_address:
        flash("Invalid IP address. Please enter a valid IP.", "error")
        return redirect(url_for('index'))

    video_url = f"{ip_address}/video"
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        flash("Failed to open video stream. Check the IP address.", "error")
        return redirect(url_for('index'))

    stream_active = True

    def generate_frames():
        """Stream frames with gender detection and gesture recognition."""
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        
        keypoint_classifier = KeyPointClassifier()
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = [row[0] for row in csv.reader(f) if row]
        
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue  # Skip if no frame is captured
            
            resultImg = detect_faces_and_bodies(frame)  # Face & body detection
            body_boxes = detect_bodies(frame)  # Get body bounding boxes
            if resultImg is None:
                continue

            debug_image = copy.deepcopy(resultImg)
            image = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    
                    # Check if the hand is above head level using body bounding box
                    for (bx1, by1, bx2, by2) in body_boxes:
                        head_level = by1  # The top y-coordinate of the body box
                        hand_y_positions = [y for _, y in landmark_list]
                        if min(hand_y_positions) < head_level:  # If hand is above the head
                            if 0 <= hand_sign_id < len(keypoint_classifier_labels):
                                label = keypoint_classifier_labels[hand_sign_id]
                                play_buzzer()
                            else:
                                label = ""
                            debug_image = draw_info_text(debug_image, label, landmark_list[0])
            
            _, buffer = cv2.imencode('.jpg', debug_image)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    """Stop the video stream and reset the state."""
    global cap, stream_active

    if cap:
        cap.release()
        cap = None  # Reset the video capture object

    stream_active = False
    flash("Webcam stopped successfully.", "info")
    return redirect(url_for('index'))

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[int(landmark.x * image_width), int(landmark.y * image_height)] for landmark in landmarks.landmark]

def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    temp_landmark_list = [[x - base_x, y - base_y] for x, y in landmark_list]
    temp_landmark_list = [coord for point in temp_landmark_list for coord in point]
    max_value = max(map(abs, temp_landmark_list))
    return [n / max_value for n in temp_landmark_list]

def draw_info_text(image, hand_sign_text, position):
    cv2.putText(image, "Gesture: " + hand_sign_text, (position[0], position[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)  # ✅ CORRECT

