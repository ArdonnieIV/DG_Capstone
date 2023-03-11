import os
import numpy as np
import math


class DataLoader():
    def __init__(self):
        self.num_features = 69 # can change this!

        myPath = os.getcwd()
        self.myDataPath = os.path.join(myPath, 'data')
        self.myDataFiles = os.listdir(self.myDataPath)


        # read in data from .npy files
        self.data = {}
        for file in self.myDataFiles:
            posePath = os.path.join(self.myDataPath, file)
            self.data[file[:-5]] = np.load(posePath, allow_pickle=True)

    def get_train_test_split(self, train_split=.8):
        """Returns data in format training_data, testing_data"""

        train_data_size = 0
        test_data_size  = 0

        X_train = None
        X_test  = None

        y_train = None
        y_test  = None
        
        for i, pose in enumerate(self.data):
            train_class_amount = math.floor(self.data[pose].shape[0] * train_split)
            train_data_size += train_class_amount
            test_class_amount = self.data[pose].shape[0] - train_class_amount
            test_data_size  += test_class_amount
            if i == 0:
                X_train = self.data[pose][0:train_class_amount, :]
                X_test = self.data[pose][train_class_amount:-1, :]

                y_train = np.array([i for j in range(train_class_amount)])
                y_test  = np.array([i for j in range(test_class_amount)])
            else:
                X_train = np.concatenate((X_train, self.data[pose][0:train_class_amount, :]), axis=0)
                X_test = np.concatenate((X_test, self.data[pose][train_class_amount:-1, :]), axis=0)

                y_train = np.concatenate((y_train, np.array([i for j in range(train_class_amount)])))
                y_test  = np.concatenate((y_test, np.array([i for j in range(test_class_amount)])))
        
        return X_train, y_train, X_test, y_test

dl = DataLoader()
X_train, y_train, X_test, y_test = dl.get_train_test_split()
print("Number of training points = " + str(X_train.shape[0]))
print("Number of testing points = "+ str(X_test.shape[0]))