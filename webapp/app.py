from flask import Flask,render_template,Response
import random
import os
import sys
import cv2
import json
import base64
from time import sleep
from models.fnn import PoseFFNN
from helper import center_chest, get_pose_names
import mediapipe as mp
import torch

sys.path.append('..')

app = Flask(__name__)
camera=cv2.VideoCapture(0)

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
mediaPipe = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
poses = get_pose_names('clean')

# Define a new instance of the model
model = PoseFFNN(input_dim=69, output_dim=82)

# Load the saved model parameters into the new model
model.load_state_dict(torch.load('models/fnn_parameters.pth'))
model.to(device)

def generate_frames():
    while True:
            
        # read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def get_prediction():

    ret, frame = camera.read()

    if ret:
        imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mediaPipe.process(imageRGB)
        if results.pose_landmarks:
            featureVector = center_chest(results.pose_landmarks.landmark)
            featureVector = torch.Tensor(featureVector.flatten()).to(device)
            output = model(featureVector.to(device))
            probs = torch.nn.functional.softmax(output, dim=0)
            pred = output.argmax(dim=0, keepdim=True)
            confidence = str(round(probs[pred].item(), 2))
            label = poses[pred]
            return Response(label + ' ' + confidence, mimetype='text/html')
        else:
            return Response("None", mimetype='text/html')
    else:
        return Response("None", mimetype='text/html')

if __name__ == "__main__":
    app.run(debug=True)