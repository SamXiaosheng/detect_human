import numpy as np
import cv2


class Hog:
    # Works for 96X160 images
    # Private variables for Hog
    __bin_n = 16  # Number of bins

    # histogram
    __hist = []

    __ahist = 0

    def hog_show(self):
        cols, rows, bins = self.__ahist.shape
        print rows, cols
        img = np.zeros((rows*8, cols*8))
        print img.shape
        for i in range(0, cols-1):
            for j in range(0, rows-1):
                # get the angles
                angle_weights = self.__ahist[i][j][:]
                # normalize to 255
                angle_weights = angle_weights / np.amax(self.__ahist) * 255
                # print angle_weights

                # draw the line
                angle = 0
                x = (1, 4)
                y = (7, 4)
                cell = np.zeros((8, 8))
                for angle_weight in angle_weights:
                    myline = np.zeros((8, 8))
                    cv2.line(myline, x, y, (angle_weight, 0, 0), 1)
                    M = cv2.getRotationMatrix2D((4, 4), angle, 1)
                    myline = cv2.warpAffine(myline, M, (8, 8))
                    cell[cell < myline] = myline[cell < myline]
                    # img[j*8:(j+1)*8, i*8:(i+1)*8] =\
                    #    cv2.addWeighted(img[j*8:(j+1)*8, i*8:(i+1)*8], 0.5,
                    #                    cell, 0.5, 0.0)
                    img[j*8:(j+1)*8, i*8:(i+1)*8] = cell

                    angle += 180/self.__bin_n

        return img

    def descript(self, img):
        # get the gradients
        gx = cv2.filter2D(img, -1, np.mat('-1 0 1'))
        gy = cv2.filter2D(img, -1, np.mat('-1; 0.; 1.'))
        # print 'gx', gx
        mag, ang = cv2.cartToPolar(gx, gy)  # angles 0 - 2pi
        ang[ang >= np.pi] = ang[ang >= np.pi] - np.pi
        # plt.imshow(ang)
        # plt.show()
        # print 'ang shape:', ang.shape

        # quantize from 0...16
        bins = np.int32(self.__bin_n * ang / (np.pi))
        # print 'bin shape', bins.shape

        # separate image into cells
        rows, cols = img.shape
        numcells_h = cols/8
        numcells_v = rows/8
        cells = np.array(np.hsplit(bins, numcells_h))
        mcells = np.array(np.hsplit(mag, numcells_h))
        # print 'cells shape', cells.shape
        bin_cells = np.empty((numcells_h, numcells_v, 8, 8))
        mag_cells = np.empty((numcells_h, numcells_v, 8, 8))
        # print 'self.__bin_cells', self.__bin_cells[:][:][:][:].shape
        for i in range(0, numcells_h):
            # print 'what', np.array(np.vsplit(cells[:][i], 20)).shape
            bin_cells[i][:][:][:] = np.array(np.vsplit(cells[:][i],
                                             numcells_v))
            mag_cells[i][:][:][:] = np.array(np.vsplit(mcells[:][i],
                                             numcells_v))
        # print 'bin cells', self.__bin_cells[10][19]
        # print 'mag cells', self.__mag_cells[10][19]

        # Calculate the block orientations
        # i = 0
        # print self.__bin_cells[i:2+i, i:2+i].ravel().shape
        hist = []
        self.__ahist = np.empty((numcells_h, numcells_v, self.__bin_n))
        for i in range(0, numcells_h-1):
            for j in range(0, numcells_v-1):
                # print i
                # 50% Overlap
                hists = np.bincount(
                    np.int32(bin_cells[i:2+i, j:2+j].ravel()),
                    weights=np.int32(mag_cells[i:2+i, j:2+j].ravel()),
                    minlength=self.__bin_n)
                # print 'hists', hists
                self.__ahist[i][j][:] = np.array(hists)
                # make the 1D vector.
                hist.append(hists)
        # print np.array(hist).ravel().shape #20*12*16
        # store stuff for visuals (cell orientations)
        hist = np.array(hist).ravel()
        return hist
