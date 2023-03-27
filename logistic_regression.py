from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from dataloader.dataloader import load_data, PoseLoader
from helper import get_pose_names
import numpy as np
import argparse

# Add command line arguments for filter type and percentile
parser = argparse.ArgumentParser(description='Logistic Regression Model')
parser.add_argument('--data_type', type=str, default='filtered', help='(raw, fixed, or filtered) if filtered include percentile')
parser.add_argument('--percentile', type=int, default=85, help='percentile (default: 85)')
args = parser.parse_args()

print("------------------------------------")
print("Beginning Logistic regression model.")
print("------------------------------------")

# get a list of pose names from the helper function
poseNames = get_pose_names()

# load data using the load_data function from dataloader.py
filteredTrain, filteredVal, filteredTest = load_data(poseNames, args.data_type, args.percentile)

# create instances of the PoseLoader class for the train, validation, and test datasets
train_data = PoseLoader(filteredTrain, 'train')
val_data = PoseLoader(filteredVal, 'val')
test_data = PoseLoader(filteredTest, 'test')

# get feature matrices and label vectors for each dataset
X_train = np.concatenate((train_data.get_features(), val_data.get_features()), axis=0)
X_test = test_data.get_features()
y_train = np.concatenate((train_data.get_labels(), val_data.get_labels()), axis=0)
y_test = test_data.get_labels()

# print sizes of training and testing data
print("X train size:", X_train.shape)
print("Y train size:", y_train.shape)
print("X test size: ", X_test.shape)
print("Y test size: ", y_test.shape)

# initialize logistic regression model with balanced class weights and 10,000 maximum iterations
logistic_model = LogisticRegression(class_weight="balanced", max_iter=10000)

# fit logistic regression model on the training data
logistic_model.fit(X_train, y_train)

# make predictions on the training and testing data
train_predictions = logistic_model.predict(X_train)
test_predictions  = logistic_model.predict(X_test)

# calculate balanced accuracy scores for the training and testing data
train_accuracy = balanced_accuracy_score(y_true = y_train,
                                         y_pred = train_predictions)
test_accuracy  = balanced_accuracy_score(y_true = y_test,
                                         y_pred = test_predictions)

# print the balanced accuracy scores for the training and testing data
print('train accuracy :', train_accuracy)
print('test accuracy :', test_accuracy)
