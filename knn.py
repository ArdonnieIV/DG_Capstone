from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import balanced_accuracy_score
from dataloader.dataloader import DataLoader
import numpy as np

# retrieve train/test split from helper functions as well
X_train = None
y_train = None

X_test = None
y_test = None

X_valid = None
y_valid = None

dl = DataLoader()
X_train, y_train, X_test, y_test, X_valid, y_valid = dl.get_train_test_split()

print("X train size:", X_train.shape)
print("Y train size:", y_train.shape)
print("X test size: ", X_test.shape)
print("Y test size: ", y_test.shape)
print("X valid size: ", X_valid.shape)
print("Y valid size: ", y_valid.shape)

print("Beginning fitting!")
print("------------------")

uniformTrainRes  = []
uniformValidRes  = []
uniformTestRes   = []

distanceTrainRes = []
distanceValidRes = []
distanceTestRes  = []

for k in range(1,11):
    for weight in ['uniform', 'distance']:
        print("Now fitting model with k = " + (str)(k) + " and " + weight + "weighting.")
        model = knn(n_neighbors = k, weights = weight)
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        test_predictions  = model.predict(X_test)
        valid_predictions  = model.predict(X_valid)
        
        train_accuracy = balanced_accuracy_score(y_true = y_train,
                                         y_pred = train_predictions)
        test_accuracy  = balanced_accuracy_score(y_true = y_test,
                                                 y_pred = test_predictions)
        valid_accuracy  = balanced_accuracy_score(y_true = y_valid,
                                                 y_pred = valid_predictions)
        
        if (weight == 'uniform'):
            uniformTrainRes.append(train_accuracy)
            uniformValidRes.append(valid_accuracy)
            uniformTestRes.append(test_accuracy)
        else:
            distanceTrainRes.append(train_accuracy)
            distanceValidRes.append(valid_accuracy)
            distanceTestRes.append(test_accuracy) 
            
        print("   Training accuracy:  ", train_accuracy)
        print("   Validation accuracy:", valid_accuracy)
        print("   Testing accuracy:   ", test_accuracy)


print("Uniform Weighting Train Accuracy:     ", uniformTrainRes)
print("Uniform Weighting Validation Accuracy:", uniformValidRes)
print("Uniform Weighting Testing Accuracy:   ", uniformTestRes)
print("")
print("Distance Weighting Train Accuracy:     ", distanceTrainRes)
print("Distance Weighting Validation Accuracy:", distanceValidRes)
print("Distance Weighting Testing Accuracy:   ", distanceTestRes)