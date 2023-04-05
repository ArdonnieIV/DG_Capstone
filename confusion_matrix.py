from sklearn.metrics import confusion_matrix
from helper import get_pose_names
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_confusion_matrix(y_test, test_predictions):

    print("Confusion Matrix: ")

    # Index into this list for label names
    poses = get_pose_names()

    # Create the confusion matrix
    cm = confusion_matrix(y_test, test_predictions)

    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a figure and axis object with seaborn
    fig, ax = plt.subplots(figsize=(70, 60))
    plt.style.use('dark_background')

    # Create a heatmap of the normalized confusion matrix
    sns.heatmap(cm_norm, annot=True, cmap='mako', ax=ax, fmt='.2f', xticklabels=poses, yticklabels=poses)

    # Set the axis labels and title
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Normalized confusion matrix')

    # Show the plot
    plt.show()

    return cm_norm

def print_worst_cm(cm, total):

    print(f'Top {total} worst cases from Confusion Matrix.\n')
    poses = get_pose_names()

    allBadScores = {}
    for i in range(82):
        for j in range(82):
            if i != j:
                if cm[i][j]:
                    allBadScores[(i, j)] = cm[i][j]

    allBadScores = {k: v for k, v in sorted(allBadScores.items(), key=lambda item: item[1], reverse=True)}

    print('percent\ttrue'.ljust(77), 'pred')
    for i, pair in enumerate(allBadScores.keys()):

        print(f'{allBadScores[pair]:.2f}\t{poses[pair[0]].ljust(70)}{poses[pair[1]]}')

        if i==total:
            break
