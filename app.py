import cv2
import streamlit as st
import numpy as np
import torch
from collections import deque
import pandas as pd
from ultralytics import YOLO
import joblib
import math

# Load the YOLO model
model = YOLO("yolo11x-pose.pt")
model.to('cpu')

# Load the ML model and encoder
ml_model = joblib.load("models/mlp_model.joblib")
encoder = joblib.load("models/label_encoder.joblib")
standardScaler = joblib.load("models/standard_scaler.joblib")
# Initialize video capture
cap = cv2.VideoCapture(0)

# Streamlit app title and subheader
st.title("ACT")
st.subheader("Pose Estimation and Classification Demo")

# Placeholder for frames
frame_placeholder = st.empty()

# Create a placeholder for dynamic prediction display
prediction_placeholder = st.empty()

stop_button_pressed = st.button("Stop")
restart_button_pressed = st.button("Restart")

# Function to restart the video capture
def restart_video_capture():
    global cap
    cap.release()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.write("Error: Could not open camera")
        st.stop()

# Restart video capture if the restart button is pressed
if restart_button_pressed:
    restart_video_capture()

def calculate_angle(p1, p2, p3):
    # Create vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate dot product and magnitudes
    dot_product = torch.dot(v1, v2)
    magnitude_v1 = torch.norm(v1)
    magnitude_v2 = torch.norm(v2)
    
    # Prevent division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return None  # Undefined angle
    
    # Calculate angle in radians
    angle_rad = torch.acos(dot_product / (magnitude_v1 * magnitude_v2))
    # Convert to degrees
    return math.degrees(angle_rad.item())

# Function to calculate Euclidean distance between two points (p1, p2)
def calculate_distance(p1, p2):
    return torch.norm(p1 - p2).item()

# Function to preprocess the image and extract angles and positions
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
            print(f"No keypoints detected in image {img_idx + 1}")
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
        # Draw keypoints and connections on the image
        for i, keypoint in enumerate(results[img_idx].keypoints.xy[0]):
            x, y = keypoint[0].item(), keypoint[1].item()
            # Draw keypoints as circles
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circles
            # Add label text next to the keypoint
            cv2.putText(image, joint_labels[i], (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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

# Real-time frame processing and classification
while cap.isOpened() and not stop_button_pressed:
    ret, frame = cap.read()

    if not ret:
        st.write("Exercise Ended")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    processed_data = preprocess_image(frame_rgb)
    #scaled_data = standardScaler.fit_transform(processed_data)
    prediction = ml_model.predict(processed_data)
    predicted_class = encoder.inverse_transform(prediction)
    
        # Update the prediction markdown below the frame
    prediction_placeholder.markdown(f"""
    <div style="background-color: #f4f4f4; padding: 20px; border-radius: 10px; border: 2px solid #ddd; box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        <h3 style="text-align: center; color: #3E4E5E;">Predicted Class</h3>
        <p style="font-size: 18px; text-align: center; color: #3E4E5E; font-weight: bold;">
            {predicted_class[0]}
        </p>
    </div>
    """, unsafe_allow_html=True)
    print(predicted_class[0])

    # Display the frame in Streamlit
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # Check for stop button press or 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
