from flask import Flask, jsonify, request, render_template
import random
import os
import cv2
import mediaPipe
import torch
# TODO: add import for model class
from IPython.display import clear_output, Image
from models.fnn import PoseFFNN
from helper import plot, get_pose_names, center_chest

app = Flask(__name__)

@app.route('/')
def index():
    filename = get_pose()
    return render_template('index.html', filename=filename)

def get_prediction(img):
    # TODO: 
    return "", ""

@app.route('/predict', methods=['POST'])
def predict():
    # we will get the file from the request
    #file = request['imgfile']
    #class_id, class_name = get_prediction(img=file)
    # TODO: update to return score, once we have a calculation for that
    POSE_NAME = "POSE NAME HERE"
    POSE_SCORE = "POSE SCORE HERE"
    return jsonify({'pose_name': POSE_NAME, 'pose_score': POSE_SCORE})

@app.route('/video_predict')
def video_predict():
    # create the model
    PATH_TO_SAVED_MODEL = ''
    model = torch.load(PATH_TO_SAVED_MODEL)

    # get poses
    # Get a list of all the pose names
    poses = get_pose_names()
    # Open the default camera (usually the webcam)
    cap = cv2.VideoCapture(0)

    # Loop through each frame from the camera
    i = 0
    while True:

        # Read the current frame
        ret, frame = cap.read()

        # If the frame was read successfully
        if ret:
            i += 1
            if i % 3 == 0:
                clear_output(wait=True)
                imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mediaPipe.process(imageRGB)
                if results.pose_landmarks:
                    featureVector = center_chest(results.pose_landmarks.landmark)
                    plot(featureVector)
                    featureVector = torch.Tensor(featureVector.flatten())
                    pred = model(featureVector).argmax(dim=0, keepdim=True)
                    pose_name = poses[pred]
                    print(pose_name)
                    break
                else:
                    print('bad image')

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'pose_name': pose_name})


@app.route('/get_pose')
def get_pose():
    # randomly generate a number between 0 and 81
    pose = random.randint(0, 81)
    # get that yoga pose from the dataset, return the filename
    poses = os.listdir('poses')
    filename = "/confusion_matrix_lr.png" # TODO: change this lol
    return filename

if __name__ == "__main__":
    app.run(debug=True)