import numpy as np
import cv2 as cv
import computeHomography
from displayMatches import showInliers

RATIO = 0.75
RANSAC_EPSILON = 4.0
N_RANSAC_ITR = 1000


class panoramaTools:
    '''
    This class contains methods that call each other in orderd to build a stitched panorama.
    '''

    def __init__(self, imgs_list):
        # Image list
        self.imlist = sorted(imgs_list)
        # Image counter
        self.num_of_imgs = len(imgs_list)
        '''
        pick a coordinate system in which we would like the panorama to be rendered.
        We will (somewhat arbitrarily) choose this coordinate system to be the coordinate system of
        the middle frame Im in our image sequence.
        For that we devide the image list into two groups left and right in relation to the center image 
        '''
        self.left_list, self.right_list, self.center_im = [], [], None
        self.centerImageIdx = self.num_of_imgs//2
        self.centerImage = self.imlist[self.centerImageIdx]
        # We load each image into the array and resize it
        for i in range(self.num_of_imgs):
            self.imlist[i] = cv.imread(self.imlist[i], 1)
            '''
            The reason we resize our image is that for some photos that were taken by our phones, the size was too big and we had to wait longer than usual,
            We felt that resizing the image, although this might create artifacts and inaccurate results, will still be a good representation that we were happy of.
            '''
            self.imlist[i] = cv.resize(self.imlist[i], (600, 600))
        # Create left and right list
        for i in range(self.num_of_imgs):
            if (i <= self.centerImageIdx):
                self.left_list.append(self.imlist[i])
            else:
                self.right_list.append(self.imlist[i])
        # Saving one image for inliners and outliers gives a general idea of our implementation
        print("Saving demo image with inliners and outliners")
        if self.num_of_imgs >= 2:
            self.displayMatches(self.imlist[0], self.imlist[1])

    def buildPanorama(self):
        # The building process of the panorama is done by taking the left list and building it, then stitching the result with the right list.
        print("building left")
        self.buildLeft()
        print("building right")
        self.buildRight()
        # Remove the excess black border
        self.removeBlackBorder()
        # Save the panorama
        cv.imwrite("resultPanorama.jpg", self.result)
        # Display the panorama
        cv.imshow("Panorama_image.jpg", self.result)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def buildLeft(self):
        # Stich the images in the left list
        img1 = self.left_list[0]
        for img2 in self.left_list[1:]:
            # Compute the homography from the two images
            H = computeHomography.generateHomography(img1, img2)
            # Transofrm the coordinate system
            xh, dsize, offsetx, offsety = computeHomography.fixLeft(H, img1)
            # Stitch the two images together
            img1 = self.warpLeft(xh, dsize, offsetx, offsety, img1, img2)
        print("done left")
        self.result = img1

    def buildRight(self):
        for img in self.right_list:
            H = computeHomography.generateHomography(self.result, img)
            dsize = computeHomography.fixRight(H, img, self.result)
            self.result = self.warpRight(H, dsize, img)
        print("done right")

    def warpLeft(self, H, size, offsetx, offsety, img1, img2):
        img1 = cv.warpPerspective(img1, H, size)
        # Fix img2 onto img1 using the update coordinate system
        img1[offsety:img2.shape[0] + offsety, offsetx:img2.shape[1] + offsetx] = img2
        return img1

    def warpRight(self, H, size, img):
        tmp = cv.warpPerspective(img, H, size)
        i1y, i1x = self.result.shape[:2]
        '''
        We wanted to get rid of the blacks between stitching, therefore we took the average of all nearby valus, this method will
        dispose of the blacks between each image because there is more color than black pixels (thresholding).
        '''
        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    # Check for blacks
                    if (np.array_equal(self.result[j, i], np.array([0, 0, 0])) and np.array_equal(tmp[j, i],
                                                                                                  np.array([0, 0, 0]))):
                        # Keep black
                        tmp[j, i] = [0, 0, 0]
                    else:
                        # Take the value from the current panorama
                        if (np.array_equal(tmp[j, i], [0, 0, 0])):
                            tmp[j, i] = self.result[j, i]
                        else:
                            if not np.array_equal(self.result[j, i], [0, 0, 0]):
                                bl, gl, rl = self.result[j, i]
                                tmp[j, i] = [bl, gl, rl]
                except:
                    print("passed")
                    pass

        return tmp
    # Display the matches
    def displayMatches(self, img1, img2):
        vis = showInliers(img1, img2)
        cv.imwrite("inliers.jpg", vis)

    def removeBlackBorder(self):
        gray = cv.cvtColor(self.result, cv.COLOR_BGR2GRAY)
        (_, mask) = cv.threshold(gray, 1.0, 255.0, cv.THRESH_BINARY)

        # findContours destroys input
        temp = mask.copy()
        contours = cv.findContours(temp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # sort contours by largest first (if there are more than one)
        contours = sorted(contours, key=lambda contour: len(contour), reverse=True)
        roi = cv.boundingRect(contours[0])

        # use the roi to select into the original 'stitched' image
        self.result = self.result[roi[1]:roi[3], roi[0]:roi[2]]
