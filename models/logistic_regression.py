from sklearn.linear_model import LogisticRegression
import HelperFunctions


# retrieve these from helper functions
labels = None
features = None

# retrieve train/test split from helper functions as well
X_train = None
y_train = None

X_test = None
y_test = None

logistic_model = LogisticRegression(class_weight="balanced", max_iter=1000)
logistic_model.fit(X_train, y_train)

train_accuracy = logistic_model.score(X_train, y_train)
test_accuracy  = logistic_model.score(X_test, y_test)