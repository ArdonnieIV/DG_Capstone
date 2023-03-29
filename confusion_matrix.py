from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def confusion_matrix(y_test, test_predictions):
    print("Confusion Matrix: ")
    plt.figure(figsize=(70,60))
    cf = confusion_matrix(y_test, test_predictions)
    sns.heatmap(cf, annot=True)
    plt.savefig('confusion_matrix_lr.png')


    pose_counts = defaultdict(int)
    sorted_y_test = y_test
    sorted_y_test.sort()
    for point in sorted_y_test:
        pose_counts[point] += 1
    pose_counts = np.array(list(pose_counts.values()))
    balanced_cf = cf/pose_counts

    pose2idx = {pose: idx for pose, idx in zip(os.listdir('data'), range(len(os.listdir('data'))))}

    balanced_cf = np.round(balanced_cf,decimals = 2)
    balanced_cf.set_xlabels(list(pose2idx.keys()))
    balanced_cf.set_ylabels(list(pose2idx.keys()))
    sns.heatmap(balanced_cf, annot=True)
    plt.savefig('balanced_confusion_matrix_lr.png')
