from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy


def load_data(allPoses, type, percentile=None):
    """
    Load the data for the specified pose type and return it split into train, validation, and test sets.

    Parameters:
    allPoses (list): A list of strings representing the names of the yoga poses.
    type (str): A string representing the type of data to load, either 'filtered', 'fixed', or 'raw'.
    percentile (int): If type is 'filtered', the percentile used to create the filter.

    Returns:
    A tuple containing the train, validation, and test sets for the specified pose type.
    """
    train = {}
    val = {}
    test = {}

    validType = ['raw', 'fixed', 'filtered']

    if type not in validType:
        # Throw an error if the specified type is not valid
        raise ValueError(f"Invalid type '{type}' specified.")

    # Iterate over each pose and load the corresponding data
    for pose in allPoses:

        if type == 'filtered':
            # Load filtered data from corresponding folder
            train[pose] = np.load(f'data/fixed/train/{pose}.npy', allow_pickle=True)
            val[pose] = np.load(f'data/fixed/val/{pose}.npy', allow_pickle=True)
            test[pose] = np.load(f'data/fixed/test/{pose}.npy', allow_pickle=True)

        else:
            # Load fixed or raw data from corresponding folders
            train[pose] = np.load(f'data/{type}/train/{pose}.npy', allow_pickle=True)
            val[pose] = np.load(f'data/{type}/val/{pose}.npy', allow_pickle=True)
            test[pose] = np.load(f'data/{type}/test/{pose}.npy', allow_pickle=True)

    if type == 'filtered':
        train = create_filter(train, percentile)

    return train, val, test

    #################################################################################


class PoseLoader(Dataset):

    def __init__(self, data, dataset_type: str = 'Train', oneHot=False):

        # Save the type of dataset (train, validation or test) in the class variable
        self.dataset_type = dataset_type

        # Flattened (23, 3) cartesian pose estimation
        # Define the length of the input as 23 x 3 = 69
        self.input_length = 69 
        
        # Number of poses
        # Get the length of the input data dictionary to find the number of poses
        self.output_length = len(data)
        
        # Get a list of all the pose names from the dictionary keys
        self.poseNameList = list(data.keys())
        
        # Create a combined matrix of all the pose estimations
        self.combined_matrix = np.concatenate(list(data.values()), axis=0)
        
        # Create a label matrix that assigns a unique label for each pose
        self.label_matrix = np.hstack([[i]*len(data[k]) for i,k in enumerate(self.poseNameList)])

        # One hot also implies both x and y are tensors instead of numpy arrays
        if oneHot:
            # Convert to tensor
            self.combined_matrix = torch.from_numpy(self.combined_matrix).float()
            # Convert to one hot tensors
            self.label_matrix = one_hot(torch.from_numpy(self.label_matrix).long(), 
                                        num_classes=self.output_length).float()

    def get_features(self):
        # Return the feature matrix
        return self.combined_matrix
    
    def get_labels(self):
        # Return the label matrix
        return self.label_matrix
    
    def __len__(self) -> int:
        # Assert that the length of the combined matrix and label matrix are equal
        assert(len(self.combined_matrix) == len(self.label_matrix))
        # Return the length of the combined matrix
        return len(self.combined_matrix)
    
    def __getitem__(self, idx) -> dict:

        # Create a dictionary to return the input and label for the given index
        return_dict = {
            'input': self.combined_matrix[idx],
            'label': self.label_matrix[idx]
        }

        # Return the dictionary
        return return_dict
    
    #################################################################################
    

def get_mahalanobis_distance(allVectors, pose):

    matrix = copy.deepcopy(allVectors[pose])

    # Calculate the covariance matrix
    cov = np.cov(matrix.T)

    # Add a small amount of regularization to the diagonal of the covariance matrix
    cov += 1e-8 * np.identity(cov.shape[0])

    # Calculate the inverse of the covariance matrix
    inv_cov = np.linalg.inv(cov)

    # Calculate the mean of each column of the matrix
    column_means = np.mean(matrix, axis=0)

    # Calculate the Mahalanobis distance for each row
    return np.sqrt(np.sum(np.dot((matrix - column_means), inv_cov) * (matrix - column_means), axis=1))
    
    #################################################################################

def view_get_mahalanobis_distance(allVectors):

    all_distances = np.concatenate([get_mahalanobis_distance(allVectors, pose) for pose in allVectors])
    
    # Plot a histogram of the distances to visualize the distribution
    plt.hist(all_distances, bins=75)
    plt.xlabel("Mahalanobis Distance")
    plt.ylabel("Frequency")
    plt.show()

def find_threshold(allVectors, threshold_percentile):

    # Concatenate the Mahalanobis distance of all vectors with respect to their group
    all_distances = np.concatenate([get_mahalanobis_distance(allVectors, pose) for pose in allVectors])
    
    # Calculate the percentile threshold for the Mahalanobis distance distribution
    threshold = np.percentile(all_distances, threshold_percentile)

    return threshold

    #################################################################################


def get_bad_indicies(allVectors, threshold):

    allBads = {}
    for pose in allVectors:

        distances = get_mahalanobis_distance(allVectors, pose)

        # Find the indices of rows with distance greater than a threshold
        allBads[pose] = np.argwhere(distances > threshold)

    return allBads

    #################################################################################


def remove_rows(matrix, indices):
    """
    Returns a copy of matrix with the rows at the given indices removed.
    """
    matrix = copy.deepcopy(matrix)
    return np.delete(matrix, indices, axis=0)

    #################################################################################


def create_filter(allVectors, percentile):

    # Find the threshold values for each percentile
    threshold = find_threshold(allVectors, percentile)

    # Find the outlier indicies of poses in feature matrix
    allBads = get_bad_indicies(allVectors, threshold)

    # Create a filtered version of the training data for each percentile
    allFiltered = {}

    for pose in allVectors:
        # Remove the rows that exceed the threshold value for the given percentile
        allFiltered[pose] = remove_rows(allVectors[pose], allBads[pose])
    
    # Save the filtered data to files
    return allFiltered

    #################################################################################