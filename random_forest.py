from sklearn.ensemble import RandomForestClassifier as rf
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


print("")
print("Now fitting models:")
      
logistic_model_entropy = rf(criterion = 'entropy')
logistic_model_gini    = rf(criterion = 'gini')

logistic_model_entropy.fit(X_train, y_train)
logistic_model_gini.fit(X_train, y_train)

print("Done!")
print("Now predicting models:")

train_predictions_entropy  = logistic_model_entropy.predict(X_train)
test_predictions_entropy   = logistic_model_entropy.predict(X_test)
valid_predictions_entropy  = logistic_model_entropy.predict(X_valid)

train_predictions_gini  = logistic_model_gini.predict(X_train)
test_predictions_gini   = logistic_model_gini.predict(X_test)
valid_predictions_gini  = logistic_model_gini.predict(X_valid)

train_accuracy_entropy = balanced_accuracy_score(y_true = y_train,
                                         y_pred = train_predictions_entropy)
test_accuracy_entropy  = balanced_accuracy_score(y_true = y_test,
                                         y_pred = test_predictions_entropy)
valid_accuracy_entropy  = balanced_accuracy_score(y_true = y_valid,
                                         y_pred = valid_predictions_entropy)

train_accuracy_gini = balanced_accuracy_score(y_true = y_train,
                                         y_pred = train_predictions_gini)
test_accuracy_gini  = balanced_accuracy_score(y_true = y_test,
                                         y_pred = test_predictions_gini)
valid_accuracy_gini  = balanced_accuracy_score(y_true = y_valid,
                                         y_pred = valid_predictions_gini)

print("Training UAR with information entropy criterion:  ", train_accuracy_entropy)
print("Validation UAR with information entropy criterion:", valid_accuracy_entropy)
print("Testing UAR with information entropy criterion:   ", test_accuracy_entropy)

print("Training UAR with Gini impurity criterion:  ", train_accuracy_gini)
print("Validation UAR with Gini impurity criterion:", valid_accuracy_gini)
print("Testing UAR with Gini impurity criterion:   ", test_accuracy_gini)