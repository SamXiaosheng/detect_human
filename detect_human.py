import cv2
import numpy as np
from sklearn import svm
import hog
import pickle
import sys

TRAIN_LIST_PATH = '../datasets/INRIAPerson/train_64x128_H96/'
TEST_LIST_PATH = '../datasets/INRIAPerson/test_64x128_H96/'
IMAGE_PATH = '../datasets/INRIAPerson/96X160H96/Train/'
TEST_IMAGE_PATH = '../datasets/INRIAPerson/70X134H96/Test/'
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

    def build_features(self, pos_list, neg_list, show=False):
        for pos in pos_list:
            # Detect features with HoG
            # print 'pos', pos
            img = cv2.imread(pos, cv2.IMREAD_GRAYSCALE)
            img = np.float32(img)
            features = self.hog.descript(img)
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
            img = np.float32(img)
            features = self.hog.descript(img)
            if(show):
                cv2.imshow('Image', np.uint8(img))
                cv2.imshow('Hog Visual', np.uint8(self.hog.hog_show()))
                cv2.waitKey(0)
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
loading = False

if(str(sys.argv[1]) == '-l'):
    loading = True

# TRAINING
human_detector = HumanDetector()

if loading:
    with open('human_detector.pkl', 'rb') as input:
        human_detector.clf = pickle.load(input)
else:
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
    print 'pos list length', len(pos_list)
    print 'neg list length', len(neg_list)
    human_detector.build_features(pos_list[:10], neg_list[:10], True)
    human_detector.train()
    with open('human_detector.pkl', 'wb') as output:
        pickle.dump(human_detector.clf, output, pickle.HIGHEST_PROTOCOL)

# TESTING
# Test the classifier
pos_test_list = []
with open(TEST_LIST_PATH + POS_FILENAME, 'r') as f:
        while True:
            line = f.readline()
            # Zero length indicates EOF
            if len(line) == 0:
                break

            # get filename without return
            filename = line.rpartition('/')[2].rstrip()
            pos_test_list.append(TEST_IMAGE_PATH + '/pos/' + filename)

neg_test_list = []
with open(TEST_LIST_PATH + NEG_FILENAME, 'r') as f:
        while True:
            line = f.readline()
            # Zero length indicates EOF
            if len(line) == 0:
                break

            # get filename without return
            filename = line.rpartition('/')[2].rstrip()
            neg_test_list.append(TEST_IMAGE_PATH + '/neg/' + filename)

TP = 0.0
FN = 0.0
FP = 0.0
TN = 0.0
for pos in pos_test_list[:1]:
    img = cv2.imread(pos, cv2.IMREAD_GRAYSCALE)
    # print human_detector.test(img)[0]
    # cv2.imshow('Image', np.uint8(img))
    # cv2.waitKey(0)
    if (human_detector.test(img)[0] == 1):
        TP += 1.
    else:
        FN = FN + 1.

for neg in neg_test_list[:1]:
    img = cv2.imread(neg, cv2.IMREAD_GRAYSCALE)
    if (human_detector.test(img)[0] == 1):
        FP = FP + 1.
    else:
        TN = TN + 1.

print 'neg', len(neg_test_list)
print 'pos', len(pos_test_list)
print '          pos   |  neg'
print '------------------------'
print 'pos |    {}     |  {}   '.format(TP, FN)
print '------------------------'
print 'neg |    {}     |  {}   '.format(FP, TN)
print '------------------------'

precision = TP/(TP + FP)
accuracy = (TP + TN) / (TP + TN + FP + FN)
print 'Precision:', precision
print 'Accuracy:', accuracy


# search image for human using sliding window approach
# Then use non-maximal suppression to get one highest scoring detection
