import h5py
import numpy as np
import os
import glob
import cv2
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from global_var import *

warnings.filterwarnings('ignore')

test_size = 0.10
seed      = 9
train_path = "dataset/train"
test_path  = "dataset/test"
h5_data    = 'output/data.h5'
h5_labels  = 'output/labels.h5'
scoring    = "accuracy"

def train_model(tree):
    h5f_data  = h5py.File(h5_data, 'r')
    h5f_label = h5py.File(h5_labels, 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string   = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels   = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()

    print("[STATUS] features shape: {}".format(global_features.shape))
    print("[STATUS] labels shape: {}".format(global_labels.shape))

    print("[STATUS] training started...")

    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features), np.array(global_labels), test_size=test_size, random_state=seed)

    print("[STATUS] splitted train and test data...")
    print("Train data  : {}".format(trainDataGlobal.shape))
    print("Test data   : {}".format(testDataGlobal.shape))
    print("Train labels: {}".format(trainLabelsGlobal.shape))
    print("Test labels : {}".format(testLabelsGlobal.shape))

    tree.fit(trainDataGlobal, trainLabelsGlobal)

def test_model(tree):
    train_labels = os.listdir(train_path)
    train_labels.sort()

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for file in glob.glob(test_path + "/*.jpg"):
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        prediction = tree.predict(global_feature.reshape(1,-1))[0]

        cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()