import cv2
# import matplotlib.pyplot as plt
import numpy as np

POS_LIST_PATH = '../datasets/INRIAPerson/train_64x128_H96/'
IMAGE_PATH = '../datasets/INRIAPerson/96X160H96/Train/pos/'
POS_FILENAME = 'pos.lst'


def hog():
    pass

# Training
with open(POS_LIST_PATH + POS_FILENAME, 'r') as f:
    while True:
        line = f.readline()
        # Zero length indicates EOF
        if len(line) == 0:
            break

        # get filename without return
        filename = line.rpartition('/')[2].rstrip()
        img = cv2.imread(IMAGE_PATH + filename, 1)
        print img
        cv2.imshow('image', img)
        cv2.waitKey(0)

print 'File closed', f.closed
