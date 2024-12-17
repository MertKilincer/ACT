import cv2
import streamlit as st
import numpy as np
import torch
from collections import Counter
import pandas as pd
from ultralytics import YOLO
import joblib
import math
import time

# Load the YOLO model
model = YOLO("yolo11x-pose.pt")
model.to('cpu')

# Load the ML model and encoder
ml_model = joblib.load("models/mlp_model.joblib")
encoder = joblib.load("models/label_encoder.joblib")
standardScaler = joblib.load("models/standard_scaler.joblib")

# Streamlit app title and subheader
st.title("ACT")
st.subheader("Pose Estimation and Classification Demo")

# Placeholders for UI components
frame_placeholder = st.empty()
majority_prediction_placeholder = st.empty()

# Function to calculate angles between three points
def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = torch.dot(v1, v2)
    magnitude_v1 = torch.norm(v1)
    magnitude_v2 = torch.norm(v2)
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return None
    angle_rad = torch.acos(dot_product / (magnitude_v1 * magnitude_v2))
    return math.degrees(angle_rad.item())

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return torch.norm(p1 - p2).item()

# Preprocess the image to extract keypoint-based features
def preprocess_image(image):
    results = model.predict(image, imgsz=320, conf=0.5)
    angles_per_image = []
    joint_triplets = [
        (5, 7, 9), (6, 8, 10), (5, 11, 13), (6, 12, 14),
        (7, 5, 11), (8, 6, 12), (11, 13, 15), (12, 14, 16)
    ]
    joint_labels = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
                    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
                    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
                    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]

    for img_idx in range(len(results)):
        if len(results[img_idx]) == 0:
            continue
        image_angles = {"Image": img_idx + 1}

        for triplet in joint_triplets:
            p1 = results[img_idx].keypoints.xy[0][triplet[0]]
            p2 = results[img_idx].keypoints.xy[0][triplet[1]]
            p3 = results[img_idx].keypoints.xy[0][triplet[2]]
            angle = calculate_angle(p1, p2, p3)
            image_angles[f"{joint_labels[triplet[0]]}-{joint_labels[triplet[1]]}-{joint_labels[triplet[2]]}"] = angle

        for i in range(len(results[img_idx].keypoints.xy[0])):
            for j in range(i + 1, len(results[img_idx].keypoints.xy[0])):
                p1 = results[img_idx].keypoints.xy[0][i]
                p2 = results[img_idx].keypoints.xy[0][j]
                distance = calculate_distance(p1, p2)
                image_angles[f"{joint_labels[i]}-{joint_labels[j]}_distance"] = distance

        for i in range(len(results[img_idx].keypoints.xy[0])):
            p1 = results[img_idx].keypoints.xy[0][i]
            y_pos = p1[1].item()
            x_pos = p1[0].item()
            image_angles[f"{joint_labels[i]}_position_y"] = y_pos
            image_angles[f"{joint_labels[i]}_position_x"] = x_pos

        angles_per_image.append(image_angles)

    df = pd.DataFrame(angles_per_image)
    dropped_columns = ['Left Shoulder-Left Elbow-Left Wrist',
                       'Right Shoulder-Right Elbow-Right Wrist',
                       'Left Shoulder-Left Hip-Left Knee',
                       'Right Shoulder-Right Hip-Right Knee',
                       'Left Elbow-Left Shoulder-Left Hip',
                       'Right Elbow-Right Shoulder-Right Hip', 
                       'Left Hip-Left Knee-Left Ankle',
                       'Right Hip-Right Knee-Right Ankle', 'Image']
    df = df.drop(columns=dropped_columns)
    return df

# Real-time video feed with prediction
if st.button("Start Video & Predict"):
    cap = cv2.VideoCapture(0)
    time.sleep(3) 
    if not cap.isOpened():
        st.write("Error: Could not open camera")
        st.stop()

    frame_count = 0
    predictions = []
    start_time = time.time()
    while time.time() - start_time < 10:
        ret, frame = cap.read()

        if not ret:
            st.write("Error: Could not read frame")
            break

        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_data = preprocess_image(frame_rgb)

        # Prediction
        if not processed_data.empty:
            scaled_data = standardScaler.transform(processed_data)
            prediction = ml_model.predict(scaled_data)
            predictions.append(prediction[0])

        # Display frame
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        frame_count += 1

    cap.release()

    # Majority Prediction
    if predictions:
        prediction_counts = Counter(predictions)
        majority_label, majority_count = prediction_counts.most_common(1)[0]
        majority_percentage = (majority_count / len(predictions)) * 100

        majority_prediction_placeholder.markdown(f"""
        <div style="background-color: #f4f4f4; padding: 20px; border-radius: 10px; border: 2px solid #ddd; box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
            <h3 style="text-align: center; color: #3E4E5E;">Majority Prediction</h3>
            <p style="font-size: 18px; text-align: center; color: #3E4E5E; font-weight: bold;">
                {encoder.inverse_transform([majority_label])[0]} ({majority_percentage:.2f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.write("No predictions made during the recording.")
