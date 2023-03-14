from sklearn.mixture import GaussianMixture
from sklearn.metrics import balanced_accuracy_score
from dataloader.dataloader import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

print("---------------------------------")
print("Beginning Gaussian mixture model.")
print("---------------------------------")


# retrieve train/test split from helper functions as well
print("Loading data...")

X_train = None
y_train = None

X_test = None
y_test = None

dl = DataLoader()
X_train, y_train, X_test, y_test = dl.get_train_test_split()

print("Data loaded!")

num_classes = np.unique(y_train).shape[0]
print("There are", num_classes, "different classes in our data.")

print("")
print("Beginning fitting of GMMs.")

train_accuracy = []
test_accuracy = []
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

        
    train_accuracy.append(balanced_accuracy_score(y_true = y_train,
                                             y_pred = train_predictions))
    test_accuracy.append(balanced_accuracy_score(y_true = y_test,
                                         y_pred = test_predictions))
    
    print(train_accuracy[len(train_accuracy)-1])
    print(test_accuracy[len(test_accuracy)-1])
    

print("Train Accuracy:", train_accuracy)
print("Test Accuracy: ", test_accuracy)

fig, ax = plt.subplots(figsize = (12, 6))

num_components_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ax.plot(num_components_x, train_accuracy, color = "orange", label = "Training Accuracy")
ax.plot(num_components_x, test_accuracy, color = "blue", label = "Test Accuracy")
ax.set_xlabel("Number of Components")
ax.set_ylabel("Accuracy")
ax.legend();
plt.show()