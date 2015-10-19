import numpy as np


def non_max_suppression(boxes, overlapThresh=0.3):
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return []
    # print 'boxes', boxes
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    picks = []  # boxes that will be used

    # Create vectors of coords
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    ids = np.argsort(y2)  # get indexes
    while len(ids) > 0:
        # get last
        i = ids[-1]
        picks.append(i)

        # compare square with largest y2 (lowest square in image)
        # with all other squares to find
        # smallest coords as shown as astricks
        # top-left = (x1,y1)
        # bottom right = (x2,y2)
        #        ---------
        #        |        |
        #        |  *-----|---|
        #        |  |     |   |
        #        ---------*   |
        #           |         |
        #           -----------
        xx1 = np.maximum(x1[i], x1[ids[:-1]])
        yy1 = np.maximum(y1[i], y1[ids[:-1]])
        xx2 = np.minimum(x2[i], x2[ids[:-1]])
        yy2 = np.minimum(y2[i], y2[ids[:-1]])

        # get width of the new box
        # if the no box was created (no overlap)
        # the area will be zero
        # list of widths and heights
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # print 'w', w
        # list of overlaps
        overlap = (w * h) / area[ids[:-1]]

        # delete all other boxes that overlap the picked box
        ids = ids[overlap < overlapThresh]

    return boxes[picks].astype('int')
