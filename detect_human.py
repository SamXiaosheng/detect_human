import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import hog

TRAIN_LIST_PATH = '../datasets/INRIAPerson/train_64x128_H96/'
IMAGE_PATH = '../datasets/INRIAPerson/96X160H96/Train/'
POS_FILENAME = 'pos.lst'
NEG_FILENAME = 'neg.lst'


class HumanDetector:

    # Classifier
    clf = svm.SVC()

    # Descriptor
    hog = hog.Hog()

    # Feature list
    feature_vec = []

    # Classification list
    target_vec = []

    def __init__(self):
        pass

    def build_features(self, pos_list, neg_list):
        for pos in pos_list:
            # Detect features with HoG
            print 'pos', pos
            img = cv2.imread(pos, cv2.IMREAD_GRAYSCALE)
            img = np.float32(img)
            features = self.hog.descript(img)
            plt.figure()
            plt.imshow(img)
            plt.figure()
            plt.imshow(self.hog.hog_show())
            plt.show()
            print features
            self.feature_vec.append(features)
            self.target_vec.append(1)

        for neg in neg_list:
            # Detect features with HoG
            print 'neg', neg
            img = cv2.imread(neg, cv2.IMREAD_GRAYSCALE)
            img = np.float32(img)
            features = self.hog.descript(img)
            plt.figure()
            plt.imshow(img)
            plt.figure()
            plt.imshow(self.hog.hog_show())
            plt.show()
            self.feature_vec.append(features)
            self.target_vec.append(0)

    # Use the feature vectors to train the SVM
    def train(self, X=feature_vec, y=target_vec):
        # convert the lists to array for training
        X = np.array(X)
        y = np.asarray(y)
        print X
        print y

        self.clf.fit(X, y)

    def test(self, img):
        img = np.float32(img)
        features = self.hog.descript(img)
        prediction = self.clf.predict(features)
        return prediction


# MAIN
human_detector = HumanDetector()

pos_list = []
neg_list = []
# Getting the list of positive samples
with open(TRAIN_LIST_PATH + POS_FILENAME, 'r') as f:
    while True:
        line = f.readline()
        # Zero length indicates EOF
        if len(line) == 0:
            break

        # get filename without return
        filename = line.rpartition('/')[2].rstrip()
        pos_list.append(IMAGE_PATH + '/pos/' + filename)

# Getting the list of negative samples
with open(TRAIN_LIST_PATH + NEG_FILENAME, 'r') as f:
    while True:
        line = f.readline()
        # Zero length indicates EOF
        if len(line) == 0:
            break

        # get filename without return
        filename = line.rpartition('/')[2].rstrip()
        neg_list.append(IMAGE_PATH + '/neg/' + filename)

# Training
# test one image for now
human_detector.build_features(pos_list[:1], neg_list[:1])
# human_detector.train()

# Test an image
# test_img = '../datasets/INRIAPerson/96X160H96/Train/pos/crop_000010a.png'
# img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
# print 'is it a human?', human_detector.test(img)[0]

# search image for human using sliding window approach
# Then use non-maximal suppression to get one highest scoring detection
