from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from dataloader.dataloader import DataLoader
import numpy as np




# retrieve train/test split from helper functions as well
X_train = None
y_train = None

X_test = None
y_test = None

dl = DataLoader()
X_train, y_train, X_test, y_test = dl.get_train_test_split()

print("X train size:", X_train.shape)
print("Y train size:", y_train.shape)
print("X test size: ", X_test.shape)
print("Y test size: ", y_test.shape)

logistic_model = LogisticRegression(class_weight="balanced", max_iter=10000)
logistic_model.fit(X_train, y_train)

train_predictions = logistic_model.predict(X_train)
test_predictions  = logistic_model.predict(X_test)

train_accuracy = balanced_accuracy_score(y_true = y_train,
                                         y_pred = train_predictions)
test_accuracy  = balanced_accuracy_score(y_true = y_test,
                                         y_pred = test_predictions)

print(train_accuracy)
print(test_accuracy)