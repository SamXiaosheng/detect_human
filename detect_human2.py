# ======================================================
# This script will load a trained human detector and
# search for pedestrians using a sliding window approach
#
# Author: Robert Pham
#
# ======================================================
import pickle
import human_detector
import cv2
import numpy as np
from non_max_suppression import non_max_suppression


def process_img(img, detector):
    """Search for pedestrians using a sliding window approach."""
    # will need to process with different window sizes
    boxes = []  # top-left and bottom right coords
    print 'processing...'
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    rows, cols = img.shape

    # test image as a whole
    if detector.test(img)[0] == 1:
        print 'here'
        cv2.rectangle(out, (0, 0), (cols, rows),
                      (0, 255, 0), 3)
        return out

    # test image with windows and get boxes
    # 50% overlap
    win_size = np.array([160, 90])
    windows = [win_size*2, win_size*3, win_size*4]
    for window in windows:
        # cv2.imshow("window", np.zeros(window))
        for i in range(0, rows - window[0], window[0]/4):
            for j in range(0, cols - window[1], window[1]/4):
                roi = img[i:window[0]+i, j:window[1]+j]
                # cv2.imshow("ROI", roi)
                # cv2.waitKey(1000)
                if (detector.test(roi)[0] == 1):
                    # human so draw a box
                    boxes.append((j, i, window[1]+j, window[0]+i))

    # suppress boxes
    boxes = non_max_suppression(boxes, .2)

    # draw the boxes
    for box in boxes:
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]),
                      (0, 255, 0), 1)
    return out

# -------------------------------------------------
#
# MAIN
#
# -------------------------------------------------
TEST_PATH = '../datasets/INRIAPerson/Test/'
POS_FILENAME = 'pos.lst'

# TRAINING
print 'Training...'
human_detector = human_detector.HumanDetector()

print 'loaded pre-trained human detector'
with open('human_detector.pkl', 'rb') as input:
    human_detector.clf = pickle.load(input)

# Get list of images
print '\nTesting...'
pos_test_list = []
with open(TEST_PATH + POS_FILENAME, 'r') as f:
        while True:
            line = f.readline()
            # Zero length indicates EOF
            if len(line) == 0:
                break

            # get filename without return
            filename = line.rpartition('/')[2].rstrip()
            pos_test_list.append(TEST_PATH + '/pos/' + filename)

# Search image for human using sliding window approach
for pos in pos_test_list:
    img = cv2.imread(pos, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Image', img)

    # decide if you want to process this image or not
    c = cv2.waitKey(1000)
    if c == ord('q'):
        # end processing
        break
    elif c == ord('s'):
        # skip this image
        continue

    # process
    out = process_img(img, human_detector)
    cv2.imshow('detection', out)
# Then use non-maximal suppression to get one highest scoring detection
