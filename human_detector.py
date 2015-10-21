import cv2
import numpy as np
from sklearn import svm
import hog


class HumanDetector:
    """HumanDetector object

    Uses a Linear SVM and a HoG descriptor to detect humans.
    clf (object) linear svm object
    hog (object) HoG descriptor object
    feature_vec (list) list of HoG features for each sample
    target_vec (list) list of the class for each sample [human or not]
    """
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
        """this function creates the feature vector

        given a list of positive and negative samples.
        """
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
        """Trains the SVM."""
        # convert the lists to array for training
        X = np.asarray(X)
        y = np.asarray(y)
        self.clf.fit(X, y)

    def test(self, img):
        """Tests a sample using the trained SVM."""
        features = np.asarray(self.hog.compute(img))
        prediction = self.clf.predict(features.ravel())
        return prediction
