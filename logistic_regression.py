from sklearn.linear_model import LogisticRegression
from dataloader.dataloader import DataLoader




# retrieve train/test split from helper functions as well
X_train = None
y_train = None

X_test = None
y_test = None

dl = DataLoader()
X_train, y_train, X_test, y_test = dl.get_train_test_split()

logistic_model = LogisticRegression(class_weight="balanced", max_iter=1000)
logistic_model.fit(X_train, y_train)

train_accuracy = logistic_model.score(X_train, y_train)
test_accuracy  = logistic_model.score(X_test, y_test)

print(train_accuracy)
print(test_accuracy)