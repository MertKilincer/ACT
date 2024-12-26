from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil

import torch
import math
import csv

from ultralytics import YOLO
import torch
import math
import csv
import pandas as pd
import joblib
import pandas as pd
import os 
import numpy as np

import uvicorn
from mako.template import Template
from mako.lookup import TemplateLookup

from PIL import Image
from fastapi.responses import JSONResponse

import io

from fastapi.middleware.cors import CORSMiddleware


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


def is_standing(dataset, threshold=130):
    p1 = torch.tensor([dataset['Left Hip_position_x'], dataset['Left Hip_position_y']])
    p2 = torch.tensor([dataset['Left Knee_position_x'], dataset['Left Knee_position_y']])
    p3 = torch.tensor([dataset['Left Ankle_position_x'], dataset['Left Ankle_position_y']])
    angle = calculate_angle(p1, p2, p3)
    if angle is None:
        return None
    return angle > threshold

def is_elbow_straight(dataset, threshold=160):
    p1 = torch.tensor([dataset['Left Shoulder_position_x'], dataset['Left Shoulder_position_y']])
    p2 = torch.tensor([dataset['Left Elbow_position_x'], dataset['Left Elbow_position_y']])
    p3 = torch.tensor([dataset['Left Wrist_position_x'], dataset['Left Wrist_position_y']])
    angle = calculate_angle(p1, p2, p3)
    if angle is None:
        return None
    return angle > threshold

def is_elbow_ninety(dataset, threshold1= 60 , threshold2 = 120):
    p1 = torch.tensor([dataset['Left Shoulder_position_x'], dataset['Left Shoulder_position_y']])
    p2 = torch.tensor([dataset['Left Elbow_position_x'], dataset['Left Elbow_position_y']])
    p3 = torch.tensor([dataset['Left Wrist_position_x'], dataset['Left Wrist_position_y']])
    angle = calculate_angle(p1, p2, p3)
    if angle is None:
        return None
    return angle > threshold1 and angle < threshold2


def is_shoulder_air(dataset, threshold=160):
    p1 = torch.tensor([dataset['Left Hip_position_x'], dataset['Left Hip_position_y']])
    p2 = torch.tensor([dataset['Left Shoulder_position_x'], dataset['Left Shoulder_position_y']])
    p3 = torch.tensor([dataset['Left Elbow_position_x'], dataset['Left Elbow_position_y']])
    angle = calculate_angle(p1, p2, p3)
    if angle is None:
        return None
    return angle > threshold

def is_shoulder_adjoining(dataset, threshold=30):
    p1 = torch.tensor([dataset['Left Hip_position_x'], dataset['Left Hip_position_y']])
    p2 = torch.tensor([dataset['Left Shoulder_position_x'], dataset['Left Shoulder_position_y']])
    p3 = torch.tensor([dataset['Left Elbow_position_x'], dataset['Left Elbow_position_y']])
    angle = calculate_angle(p1, p2, p3)
    if angle is None:
        return None
    return angle < threshold

def preprocess_image(image_path):

    model = YOLO("yolo11x-pose.pt")

    model.to('cuda')        
    results = model.predict(image_path, imgsz=320, conf=0.5)

    # List to store calculated angles and distances for each image
    angles_per_image = []

    # Define joint triplets for angle calculations
    joint_triplets = [
        (5, 7, 9),  # Left Elbow
        (6, 8, 10), # Right Elbow
        (5, 11, 13),# Left Hip
        (6, 12, 14),# Right Hip
        (7, 5, 11), # Left Shoulder
        (8, 6, 12), # Right Shoulder
        (11, 13, 15),# Left Knee
        (12, 14, 16) # Right Knee
    ]

    # Joint labels for reference
    joint_labels = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
                "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
                "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
                "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]
    # Iterate over each image in the batch

    for img_idx in range(len(results)):  # Iterate over images (batch size)
        keypoints = results[img_idx].keypoints.xy[0]
        if len(results[img_idx]) == 0:
            print(f"No keypoints detected in image {img_idx + 1}")
            return None
        if len(results[img_idx]) > 1:
            print(f"Multiple detections in image {img_idx + 1}")
            # find the larger bounding box
            max_area = 0
            max_idx = 0
            for i in range(len(results[img_idx])):
                # turn the tensors to floats                
                print(results[img_idx].keypoints.xy[0][i])
                area = results[img_idx].keypoints.xy[0][i][1].item() * results[img_idx].keypoints.xy[0][i][0].item()
                if area > max_area:
                    max_area = area
                    max_idx = i

            keypoints = results[img_idx].keypoints.xy[max_idx]

            
                        
        image_angles = {"Image": img_idx + 1}
        
        # Calculate angles and distances for each joint triplet
        for triplet in joint_triplets:
            p1 = keypoints[triplet[0]]  # Correctly access the joint coordinates
            p2 = keypoints[triplet[1]]
            p3 = keypoints[triplet[2]]
            
            # Make sure the coordinates are tensors and have the shape [2] (x, y)
            angle = calculate_angle(p1, p2, p3)
            image_angles[f"{joint_labels[triplet[0]]}-{joint_labels[triplet[1]]}-{joint_labels[triplet[2]]}"] = angle
        
        # Calculate and add distances for each joint pair
        for i in range(len(keypoints)):
            for j in range(i + 1, len(keypoints)):
                p1 = keypoints[i]
                p2 = keypoints[j]
                distance = calculate_distance(p1, p2)
                image_angles[f"{joint_labels[i]}-{joint_labels[j]}_distance"] = distance

        #add the positions of the joints
        for i in range(len(keypoints)):
            p1 = keypoints[i]
            y_pos = p1[1]
            # convert tensor to float
            y_pos = y_pos.item()

            x_pos = p1[0]
            # convert tensor to float
            x_pos = x_pos.item()

            image_angles[f"{joint_labels[i]}_position_y"] = y_pos
            image_angles[f"{joint_labels[i]}_position_x"] = x_pos
            
        
        angles_per_image.append(image_angles)

    # return angles_per_image as a dataframe
    df = pd.DataFrame(angles_per_image)
    dropped_columns = ['Left Shoulder-Left Elbow-Left Wrist',
    'Right Shoulder-Right Elbow-Right Wrist',
    'Left Shoulder-Left Hip-Left Knee',
    'Right Shoulder-Right Hip-Right Knee',
    'Left Elbow-Left Shoulder-Left Hip',
    'Right Elbow-Right Shoulder-Right Hip', 'Left Hip-Left Knee-Left Ankle',
    'Right Hip-Right Knee-Right Ankle', 'Image']
    df = df.drop(columns = dropped_columns)


    hand_crafted_features = pd.DataFrame()

    hand_crafted_features["is_standing"] = df.apply(is_standing, axis=1)
    
    hand_crafted_features["is_elbow_straight"] = df.apply(is_elbow_straight, axis=1)

    hand_crafted_features["is_elbow_ninety"] = df.apply(is_elbow_ninety, axis=1)

    hand_crafted_features["is_shoulder_air"] = df.apply(is_shoulder_air, axis=1)

    hand_crafted_features["is_shoulder_adjoining"] = df.apply(is_shoulder_adjoining, axis=1)


    for column in hand_crafted_features.columns:
        hand_crafted_features[column] = hand_crafted_features[column].apply(lambda x: 0.5 if x is None else 0 if x == False else 1)

    added_joints = ["Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow","Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]

    # put these positions and distances between them to a hand_crafted_features dataframe
    for joint in added_joints:
        hand_crafted_features[f"{joint}_position_x"] = df[f"{joint}_position_x"]
        hand_crafted_features[f"{joint}_position_y"] = df[f"{joint}_position_y"]

    # put distances between each joint to the dataframe
    for i in range(len(added_joints)):
        for j in range(i + 1, len(added_joints)):
            hand_crafted_features[f"{added_joints[i]}-{added_joints[j]}_distance"] = df[f"{added_joints[i]}-{added_joints[j]}_distance"]

    # scale data using standard scaler saved in models folder
    standardScaler = joblib.load("models/standard_scaler.joblib")
    hand_crafted_features = standardScaler.transform(hand_crafted_features)
    print(hand_crafted_features)
    return hand_crafted_features

def predict_image(image_folder, model_path, encoder_path):
    # Load the model
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    # Load the image
    image_angles = preprocess_image(image_folder)
    if image_angles is None:
        return None

    # Make predictions
    predictions = model.predict(image_angles)
    print(predictions)
    if predictions is None:
        return None
    # decode the predictions with using the encoder
    predictions = predictions.argmax(axis=1)
    predictions = encoder.inverse_transform(predictions)
    return predictions



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
template_lookup = TemplateLookup(directories=['templates'])


UPLOAD_FOLDER = "static/images/"

@app.get("/", response_class=HTMLResponse)
async def index():
    template = template_lookup.get_template("index.html")
    return HTMLResponse(content=template.render(), status_code=200)


@app.post("/predict/")
async def predict(request):
    try:
        #use the post request to get the image
        print(request)
        prediction = predict_image("static/images/captured_image.png", "models/best_mlp_model.joblib", "models/label_encoder.joblib")
        if prediction is None:
            print("prediction is none")
            return JSONResponse(status_code=500, content={"error": "No keypoints detected in the image"})
        print(prediction)  

        #write prediction to the static/prediction.txt file
        with open("static/prediction.txt", "w") as file:
            file.write(prediction[0])
            
        return JSONResponse(content={"prediction": prediction[0]})
    except Exception as e:
        # Handle errors gracefully
        print(e)
        return JSONResponse(status_code=500, content={"error": f"{e}"})

    
@app.post("/uploadimg")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to the server
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_location, "wb") as image_file:
            image_file.write(await file.read())

        # Perform prediction on the saved file
        prediction = predict_image(
            image_folder=file_location,
            model_path="models/best_mlp_model.joblib",
            encoder_path="models/label_encoder.joblib",
        )
        if not prediction:
            return JSONResponse(status_code=500, content={"error": "No prediction available"})

        # Return the prediction and image path
        return JSONResponse(content={"prediction": prediction[0], "image_path": f"/static/images/{file.filename}"})

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    
@app.get("/upload/")
async def upload_file(image_data: str):
    try:  
        # Save the image to the server
        file_location = os.path.join(UPLOAD_FOLDER, image_data)

        # Perform prediction on the saved file
        prediction = predict_image(
            image_folder=file_location,
            model_path="models/best_mlp_model.joblib",
            encoder_path="models/label_encoder.joblib",
        )   
        if not prediction:
            return JSONResponse(status_code=500, content={"error": "No prediction available"})

        return JSONResponse(content={"prediction": prediction[0], "image_path": file_location})

    except Exception as e:
        # Handle errors gracefully
        print(e)
        return JSONResponse(status_code=500, content={"error": f"{e}"})
    

@app.get("/prediction")
async def get_prediction():
    try:
        # Read the prediction from the file
        with open("static/prediction.txt", "r") as file:
            prediction = file.read()
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        # Handle errors gracefully
        print(e)
        return JSONResponse(status_code=500, content={"error": f"{e}"})

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

