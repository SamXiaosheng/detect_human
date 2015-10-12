import cv2
# import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

TRAIN_LIST_PATH = '../datasets/INRIAPerson/train_64x128_H96/'
IMAGE_PATH = '../datasets/INRIAPerson/96X160H96/Train/'
POS_FILENAME = 'pos.lst'
NEG_FILENAME = 'neg.lst'


class Hog:

    # Private variables for Hog
    __bin_n = 16  # Number of bins

    # histogram
    __hist = []

    __bin_cells = 1
    __mag_cells = 1

    def descript(self, img):
        cv2.imshow('image', img)

        # get the gradients
        gx = cv2.filter2D(img, -1, np.mat('-1 0 1'))
        gy = cv2.filter2D(img, -1, np.mat('-1; 0.; 1.'))
        #print 'gx', gx
        mag, ang = cv2.cartToPolar(gx, gy)  # angles 0 - 2pi

        #print 'ang shape:', ang.shape

        # quantize from 0...16
        bins = np.int32(self.__bin_n * ang / (2*np.pi))
        #print 'bin shape', bins.shape

        # separate image into cells
        rows, cols = img.shape
        numcells_h = cols/8
        numcells_v = rows/8
        cells = np.array(np.hsplit(bins, numcells_h))
        mcells = np.array(np.hsplit(mag, numcells_h))
        #print 'cells shape', cells.shape
        self.__bin_cells = np.empty((numcells_h,numcells_v,8,8))
        self.__mag_cells = np.empty((numcells_h,numcells_v,8,8))
        #print 'self.__bin_cells', self.__bin_cells[:][:][:][:].shape
        for i in range(0,numcells_h):
            #print 'what', np.array(np.vsplit(cells[:][i], 20)).shape
            self.__bin_cells[i][:][:][:] = np.array(np.vsplit(cells[:][i], numcells_v))
            self.__mag_cells[i][:][:][:] = np.array(np.vsplit(mcells[:][i], numcells_v))
        #print 'bin cells', self.__bin_cells[10][19]
        #print 'mag cells', self.__mag_cells[10][19]

        # Calculate the block orientations
        i = 0
        #print self.__bin_cells[i:2+i, i:2+i].ravel().shape
        hist = []

        for i in range(0,numcells_h-1):
            for j in range(0,numcells_v-1):
                # print i
                # 50% Overlap
                hists = np.bincount(np.int32(self.__bin_cells[i:2+i, j:2+j].ravel()),
                                    weights=np.int32(self.__mag_cells[i:2+i, j:2+j].ravel()), 
                                    minlength=self.__bin_n)
                # print hists
                # make the 1D vector. 
                hist.append(hists)
        # print np.array(hist).ravel().shape #20*12*16
        # store stuff for visuals (cell orientations)
        hist = np.array(hist).ravel()
        self.__hist = hist
        return hist

        cv2.imshow('mag', mag)


class HumanDetector:

    # Classifier
    clf = svm.SVC()

    # Descriptor
    hog = Hog()

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
            print features
            self.feature_vec.append(features)
            self.target_vec.append(1)

        for neg in neg_list:
            # Detect features with HoG
            print 'neg', neg
            img = cv2.imread(neg, cv2.IMREAD_GRAYSCALE)
            img = np.float32(img)
            features = self.hog.descript(img)
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
human_detector.build_features(pos_list[:1], neg_list[:1])  # test one image for now
human_detector.train()

# Test an image
test_img = '../datasets/INRIAPerson/96X160H96/Train/pos/crop_000010a.png'
img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
print 'is it a human?', human_detector.test(img)[0]
