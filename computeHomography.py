import math
import random
import ransac
import cv2 as cv
import numpy as np
RATIO = 0.7
TRESHOLD = 4.0


def applyHomography(H, srcCord):
    trgCord = []
    for i in range(srcCord.shape[0]):
        x_i, y_i = srcCord[i]
        x_Trans = H[0, 0] * x_i + H[0, 1] * y_i + H[0, 2]
        y_Trans = H[1, 0] * x_i + H[1, 1] * y_i + H[1, 2]
        z = H[2, 0] * x_i + H[2, 1] * y_i + H[2, 2]

        x_Trans = x_Trans / z
        y_Trans = y_Trans / z

        transPoint = [x_Trans, y_Trans]
        trgCord.append(transPoint)

    return trgCord


def pixelTransform(H, srcCord):
    x_i, y_i = srcCord
    x_Trans = H[0, 0] * x_i + H[0, 1] * y_i + H[0, 2]
    y_Trans = H[1, 0] * x_i + H[1, 1] * y_i + H[1, 2]
    z = H[2, 0] * x_i + H[2, 1] * y_i + H[2, 2]

    x_Trans = x_Trans / z
    y_Trans = y_Trans / z

    transPoint = (x_Trans, y_Trans)

    return transPoint


def generateHomography(imageA, imageB):
    '''
    Generate the keypoints and descriptors using OCV
    '''
    kp1, des1 = keypointsDescriptors(imageA)
    kp2, des2 = keypointsDescriptors(imageB)

    (filterMatches, DMatches) = matches(des1, des2)

    H = calculateHomography(filterMatches, kp1, kp2, DMatches)

    return H


def fixLeft(H, img):
    '''
    Correct the coordinates and size
    '''
    xh = np.linalg.inv(H)
    ds = np.dot(xh, np.array([img.shape[1], img.shape[0], 1]))
    ds = ds / ds[-1]
    f1 = np.dot(xh, np.array([0, 0, 1]))
    f1 = f1 / f1[-1]
    xh[0][-1] += abs(f1[0])
    xh[1][-1] += abs(f1[1])
    ds = np.dot(xh, np.array([img.shape[1], img.shape[0], 1]))
    offsety = abs(int(f1[1]))
    offsetx = abs(int(f1[0]))
    dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
    return xh, dsize, offsetx, offsety


def fixRight(H, img, currentpan):
    txyz = np.dot(H, np.array([img.shape[1], img.shape[0], 1]))
    txyz = txyz / txyz[-1]
    dsize = (int(txyz[0]) + currentpan.shape[1], int(txyz[1]) + currentpan.shape[0])
    return dsize


def calculateHomography(matches, keyPointsA, keyPointsB, DMatches):
    if len(matches) > 4:

        startPts = np.float32([keyPointsA[i] for (_, i) in matches])
        endPts = np.float32([keyPointsB[i] for (i, _) in matches])

        H, _ = cv.findHomography(endPts, startPts, cv.RANSAC, TRESHOLD)
        H = np.array(H)

        '''
        This is the findHomography that we created, it does work for some cases (like with backyard) but for others it does not, we are not sure why this happenes,
        leaving it here for future improvements
        '''
        # H, _ = ransac.ransacHomography(keyPointsA, keyPointsB, DMatches, 1000, 4)
        # H = np.array(H)

        return H
    return None


def keypointsDescriptors(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kps, des = sift.detectAndCompute(img, None)
    # From keypoint object to numpy array
    kps = np.float32([kp.pt for kp in kps])
    return (kps, des)


def matches(des1, des2):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, 2)
    filtered_matches = []
    allMatches = []
    # Filter using lowe's ratio
    for m, n in matches:
        if m.distance < RATIO*n.distance:
            filtered_matches.append((m.trainIdx, m.queryIdx))
            allMatches.append(m)
    return (filtered_matches, allMatches)
