import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
from sklearn.metrics import*
import cv2
import os
import itertools



# confusion matrixをプロットし画像として保存する関数
def plot_confusion_matrix(cm, classes, output_file,
                          normalize=False,
                          cmap=matplotlib.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Confusion matrix [Normalization]'
        # print("Normalized confusion matrix")
    else:
        title = 'Confusion matrix'
        # print('Confusion matrix, without normalization')

    # print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


