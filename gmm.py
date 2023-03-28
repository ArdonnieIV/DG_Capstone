from sklearn.mixture import GaussianMixture
from sklearn.metrics import balanced_accuracy_score
from dataloader.dataloader import load_data, PoseLoader
from helper import get_pose_names
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Add command line arguments for filter type and percentile
parser = argparse.ArgumentParser(description='Logistic Regression Model')
parser.add_argument('--data_type', type=str, default='filtered', help='(raw, fixed, or filtered) if filtered include percentile')
parser.add_argument('--percentile', type=int, default=85, help='percentile (default: 85)')
args = parser.parse_args()

print("---------------------------------")
print("Beginning Gaussian mixture model.")
print("---------------------------------")

# retrieve train/test split from helper functions as well
print("Loading data...")

# get a list of pose names from the helper function
poseNames = get_pose_names()

# load data using the load_data function from dataloader.py
filteredTrain, filteredVal, filteredTest = load_data(poseNames, args.data_type, args.percentile)

# create instances of the PoseLoader class for the train, validation, and test datasets
train_data = PoseLoader(filteredTrain, 'train')
val_data = PoseLoader(filteredVal, 'val')
test_data = PoseLoader(filteredTest, 'test')

# get feature matrices and label vectors for each dataset
X_train = train_data.get_features()
X_valid = val_data.get_features()
X_test = test_data.get_features()
y_train = train_data.get_labels()
y_valid = val_data.get_labels()
y_test = test_data.get_labels()

# print sizes of training and testing data
print("X train size:", X_train.shape)
print("Y train size:", y_train.shape)
print("X test size: ", X_test.shape)
print("Y test size: ", y_test.shape)

print("Data loaded!")

num_classes = np.unique(y_train).shape[0]
print("There are", num_classes, "different classes in our data.")

print("")
print("Beginning fitting of GMMs.")

train_accuracy = []
test_accuracy = []
valid_accuracy = []
for num_components in range(1, 11):
    print("Now fitting with", num_components, "components.")
    classGMMs = np.empty(num_classes, dtype = GaussianMixture)
    for i in tqdm(range(num_classes)):
        classGMMs[i] = GaussianMixture(n_components = num_components, 
                                             max_iter = 1000, 
                                             covariance_type = 'diag',
                                             n_init = 3,
                                             random_state = 0)

        dataIndices = np.argwhere(y_train == i).ravel()
        classGMMs[i].fit(X_train[dataIndices])

    print("Fitting completed!")
    print("Calculating accuracy...")

    def predict(sample):
        # Your implementation goes here
        scores = np.empty(num_classes)
        for i in range(num_classes):
            scores[i] = classGMMs[i].score(np.array([sample])).sum()
        return np.argmax(scores)


    # testing
    train_predictions = np.zeros(y_train.shape[0])
    for i, sample in tqdm(enumerate(X_train)):
        train_predictions[i] = predict(sample)

    test_predictions = np.zeros(y_test.shape[0])
    for i, sample in tqdm(enumerate(X_test)):
        test_predictions[i] = predict(sample)
        
    valid_predictions = np.zeros(y_valid.shape[0])
    for i, sample in tqdm(enumerate(X_valid)):
        valid_predictions[i] = predict(sample)
        
    train_accuracy.append(balanced_accuracy_score(y_true = y_train,
                                             y_pred = train_predictions))
    test_accuracy.append(balanced_accuracy_score(y_true = y_test,
                                         y_pred = test_predictions))
    valid_accuracy.append(balanced_accuracy_score(y_true = y_valid,
                                         y_pred = valid_predictions))
    
    print(train_accuracy[len(train_accuracy)-1])
    print(test_accuracy[len(test_accuracy)-1])
    print(valid_accuracy[len(valid_accuracy)-1])
    

print("Train Accuracy:", train_accuracy)
print("Test Accuracy: ", test_accuracy)
print("Validation Accuracy: ", valid_accuracy)

fig, ax = plt.subplots(figsize = (12, 6))

num_components_x = range(1, 11)
ax.plot(num_components_x, train_accuracy, color = "orange", label = "Training Accuracy")
ax.plot(num_components_x, valid_accuracy, color = "blue", label = "Validation Accuracy")
ax.set_xlabel("Number of Components")
ax.set_ylabel("Accuracy")
ax.legend()
plt.show()