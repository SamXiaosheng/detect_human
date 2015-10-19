import cv2
import numpy as np
from sklearn import svm
import hog


class HumanDetector:

    # Classifier
    clf = svm.LinearSVC()  # one versus rest train only one class
    # clf = svm.SVC(kernel='linear')
    # Descriptor
    hog = hog.Hog()
    # hog = cv2.HOGDescriptor()

    # Feature list
    feature_vec = []

    # Classification list
    target_vec = []

    def __init__(self):
        pass

    def build_features(self, pos_list, neg_list, show=False):
        for pos in pos_list:
            # Detect features with HoG
            # print 'pos', pos
            img = cv2.imread(pos, cv2.IMREAD_GRAYSCALE)
            features = self.hog.compute(img)
            if(show):
                cv2.imshow('Image', np.uint8(img))
                cv2.imshow('Hog Visual', np.uint8(self.hog.hog_show()))
                cv2.waitKey(0)
            # print features
            self.feature_vec.append(features)
            self.target_vec.append(1)

        for neg in neg_list:
            # Detect features with HoG
            # print 'neg', neg
            img = cv2.imread(neg, cv2.IMREAD_GRAYSCALE)
            features = self.hog.compute(img)
            if(show):
                cv2.imshow('Image', np.uint8(img))
                cv2.imshow('Hog Visual', np.uint8(self.hog.hog_show()))
                cv2.waitKey(0)
            self.feature_vec.append(features)
            self.target_vec.append(0)

    # Use the feature vectors to train the SVM
    def train(self, X=feature_vec, y=target_vec):
        # convert the lists to array for training
        X = np.asarray(X)
        y = np.asarray(y)
        self.clf.fit(X, y)

    def test(self, img):
        features = np.asarray(self.hog.compute(img))
        prediction = self.clf.predict(features.ravel())
        return prediction
