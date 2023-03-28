import matplotlib.pyplot as plt
import numpy as np
import copy
import ast
import cv2
import os
from matplotlib import cm


def cartesianToSpherical(cartesianCoordinate):

    assert cartesianCoordinate.shape[0] == 3, "Make sure your input is a numpy array of length 3."
    assert cartesianCoordinate.dtype == np.dtype('float64'), "Make sure all inputs are floats!"
    
    x = cartesianCoordinate[0]
    y = cartesianCoordinate[1]
    z = cartesianCoordinate[2]

    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / rho)
    phi = np.arctan2(y, x)

    return np.array([rho, theta, phi])

    #################################################################################


def sphericalToCartesian(sphericalCoordinate):

    assert sphericalCoordinate.shape[0] == 3, "Make sure your input is a numpy array of length 3."
    assert sphericalCoordinate.dtype == np.dtype('float64'), "Make sure all inputs are floats!"
    
    rho = sphericalCoordinate[0]
    theta = sphericalCoordinate[1]
    phi = sphericalCoordinate[2]

    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)

    return np.array([x, y, z])

    #################################################################################


# This function takes a landmark and an index, and returns a simple [x, y, z] numpy array of the location.
def getLocation(landmark, index):

    return np.array([landmark[index].x, landmark[index].y, landmark[index].z])

    #################################################################################


def center_chest(pose):

    CHEST_LOCATION_CARTESIAN = (getLocation(pose, 11) + getLocation(pose, 12)) / 2.0
    indexesToUse = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    formattedRow = np.empty(0)

    for index in indexesToUse:
        newTriple = getLocation(pose, index) - CHEST_LOCATION_CARTESIAN
        newTriple[1] *= -1
        formattedRow = np.append(formattedRow, newTriple)

    formattedRow.shape = 23, 3
    return formattedRow

    #################################################################################


def rotate_pose(pose):

    flip = 1
    if pose[1][2] < pose[2][2]:
        flip *= -1

    WAIST_CENTER = (pose[13] + pose[14]) / 2.0
    ogTheta = np.arctan2(WAIST_CENTER[0], WAIST_CENTER[1])

    # create the rotation matrix for x-axis
    theta_x = np.arctan2(pose[1][1], pose[1][0])
    R_x = np.array([[np.cos(theta_x), -np.sin(theta_x), 0],
                  [np.sin(theta_x), np.cos(theta_x), 0],
                  [0, 0, 1]])
    
    # rotate the point
    newPose = pose @ R_x

    # create the rotation matrix for z-axis
    theta_z = np.arctan2(newPose[1][0], newPose[1][2])
    R_z = np.array([[np.cos(theta_z), 0, np.sin(theta_z)],
                    [0, 1, 0],
                    [-np.sin(theta_z), 0, np.cos(theta_z)]])

    # rotate the point around z-axis
    newPose = newPose @ R_z

    WAIST_CENTER = (newPose[13] + newPose[14]) / 2.0
    newTheta = np.arctan2(WAIST_CENTER[0], WAIST_CENTER[1])

    # create the rotation matrix for x-axis
    theta_x = ogTheta*flip - newTheta
    R_x = np.array([[np.cos(theta_x), -np.sin(theta_x), 0],
                  [np.sin(theta_x), np.cos(theta_x), 0],
                  [0, 0, 1]])

    # rotate the point
    newPose = newPose @ R_x

    return newPose

    #################################################################################


# Plot feature vector as a 3d array.
def plot(pose):

    pose.shape = (23, 3)

    fig = plt.figure(figsize = (6, 6))
    ax = plt.axes(projection='3d')

    ax.scatter3D(pose[:,0], pose[:,1], pose[:,2], c = "white", s = 2.5, alpha = 1)
    ax.scatter3D(pose[:,0], pose[:,1], pose[:,2], c = "black", s = 4.5, alpha = 0.5)
    ax.scatter3D(pose[0,0], pose[0,1], pose[0,2], c = "black", s = 800, alpha = 0.5)
    ax.scatter3D(pose[0,0], pose[0,1], pose[0,2], c = "black", s = 600, alpha = 0.75)
    ax.scatter3D(pose[0,0], pose[0,1], pose[0,2], c = "black", s = 600, alpha = 0.75)

    halfConnections = [(0, 1), (0, 2), (1, 2), (13, 14)]
    for connection in halfConnections:
        rowA = pose[connection[0]]
        rowB = pose[connection[1]]
        mat = np.transpose([rowA, rowB])
        ax.plot3D(mat[0], mat[1], mat[2], c = "#444444", lw = 2, alpha = 1)

    leftHalfConnections = [(1, 3), (3, 5), (5, 7), (7, 9), (5, 9), (5, 11),
                           (1, 13), (13, 15), (15, 17), (17, 19), (19, 21), (17, 21)]
    
    for connection in leftHalfConnections:
        rowA = pose[connection[0]]
        rowB = pose[connection[1]]
        mat = np.transpose([rowA, rowB])
        ax.plot3D(mat[0], mat[1], mat[2], c = "#00CCCC", lw = 2.5, alpha = 1)
    
    rightHalfConnections = [(2, 4), (4, 6), (6, 8), (8, 10), (6, 10), (6, 12),
                            (2, 14), (14, 16), (16, 18), (18, 20), (20, 22), (18, 22)]
    for connection in rightHalfConnections:
        rowA = pose[connection[0]]
        rowB = pose[connection[1]]
        mat = np.transpose([rowA, rowB])
        ax.plot3D(mat[0], mat[1], mat[2], c = "#EEA000", lw = 2.5, alpha = 1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.view_init(90, -90)
    plt.show()

    #################################################################################


def yogaToVector(myYogaPath, mediaPipe):

    myYogaFolders = os.listdir(myYogaPath)

    for folder in myYogaFolders:

        print(folder)
        rawVectors = []
        fixedVectors = []

        posePath = os.path.join(myYogaPath, folder)
        myYogaImages = os.listdir(posePath)

        for imageName in myYogaImages:

            imagePath = os.path.join(posePath, imageName)
            
            imageRGB = None
            try:
                image = cv2.imread(imagePath)
                imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                continue

            results = mediaPipe.process(imageRGB)
            if results.pose_landmarks:
                featureVector = center_chest(results.pose_landmarks.landmark)
                fixedVector = rotate_pose(featureVector)
                rawVectors.append(featureVector.flatten())
                fixedVectors.append(fixedVector.flatten())

        # Set the seed value for the random number generator to ensure consistent results
        np.random.seed(42)

        # Shuffle the indices of the images to create train, test, and validation sets
        num_images = len(rawVectors)
        shuffled_indices = np.random.permutation(num_images)
        train_indices = shuffled_indices[:int(num_images*0.6)]
        test_indices = shuffled_indices[int(num_images*0.6):int(num_images*0.8)]
        val_indices = shuffled_indices[int(num_images*0.8):]

        raw_train = np.array(rawVectors)[train_indices]
        raw_val = np.array(rawVectors)[val_indices]
        raw_test = np.array(rawVectors)[test_indices]

        fixed_train = np.array(fixedVectors)[train_indices]
        fixed_val = np.array(fixedVectors)[val_indices]
        fixed_test = np.array(fixedVectors)[test_indices]

        # Save the raw and fixed vectors to their respective train, test, and validation folders
        np.save(f'data/raw/train/{folder}.npy', raw_train, allow_pickle=True)
        np.save(f'data/raw/val/{folder}.npy', raw_val, allow_pickle=True)
        np.save(f'data/raw/test/{folder}.npy', raw_test, allow_pickle=True)
        
        np.save(f'data/fixed/train/{folder}.npy', fixed_train, allow_pickle=True)
        np.save(f'data/fixed/val/{folder}.npy', fixed_val, allow_pickle=True)
        np.save(f'data/fixed/test/{folder}.npy', fixed_test, allow_pickle=True)

    #################################################################################


def get_pose_names() -> list:

    # open the text file for reading
    with open('poses.txt', 'r') as f:
        # read the contents of the file as a string
        contents = f.read()

    # parse the string as a Python list using ast.literal_eval()
    return ast.literal_eval(contents)

    #################################################################################