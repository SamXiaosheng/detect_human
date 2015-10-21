# ======================================================
# This script will load or train a human detector and
# evaluate the classifer.
#
# Author: Robert Pham
#
# ======================================================
import pickle
import sys
import human_detector
import cv2

TRAIN_LIST_PATH = '../datasets/INRIAPerson/train_64x128_H96/'
TEST_LIST_PATH = '../datasets/INRIAPerson/test_64x128_H96/'
IMAGE_PATH = '../datasets/INRIAPerson/96X160H96/Train/'
TEST_IMAGE_PATH = '../datasets/INRIAPerson/70X134H96/Test/'
POS_FILENAME = 'pos.lst'
NEG_FILENAME = 'neg.lst'


# MAIN
loading = False

if(str(sys.argv[1]) == '-l'):
    loading = True

# TRAINING
print 'Training...'
human_detector = human_detector.HumanDetector()

if loading:
    print 'loaded pre-trained human detector'
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
    human_detector.build_features(pos_list, neg_list, False)
    human_detector.train()
    with open('human_detector.pkl', 'wb') as output:
        pickle.dump(human_detector.clf, output, pickle.HIGHEST_PROTOCOL)

# TESTING
# Test the classifier
print '\nTesting...'
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

print 'pos', len(pos_test_list)
print 'neg', len(neg_test_list)

TP = 0.0
FN = 0.0
FP = 0.0
TN = 0.0
for pos in pos_test_list:
    img = cv2.imread(pos, cv2.IMREAD_GRAYSCALE)
    # print human_detector.test(img)[0]
    # cv2.imshow('Image', np.uint8(img))
    # cv2.waitKey(0)
    if (human_detector.test(img)[0] == 1):
        TP += 1.
    else:
        FN = FN + 1.

for neg in neg_test_list:
    img = cv2.imread(neg, cv2.IMREAD_GRAYSCALE)
    if (human_detector.test(img)[0] == 1):
        FP = FP + 1.
    else:
        TN = TN + 1.

print '\nClassifier Results...'
print '          pos    |  neg'
print '------------------------'
print 'pos |    {}     |  {}   '.format(TP, FN)
print '------------------------'
print 'neg |    {}     |  {}   '.format(FP, TN)
print '------------------------'

precision = TP/(TP + FP)
accuracy = (TP + TN) / (TP + TN + FP + FN)
print 'Precision:', precision
print 'Accuracy:', accuracy
