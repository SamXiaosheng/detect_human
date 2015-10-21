# ======================================================
# This script tests the non_max_suppression function
#
# Author: Robert Pham
#
# ======================================================
from non_max_suppression import non_max_suppression

# testing box in box
boxes = []
boxes.append((1, 1, 20, 20))
boxes.append((5, 5, 10, 10))
boxes.append((100, 100, 130, 130))
print boxes
picked = non_max_suppression(boxes)
print picked
