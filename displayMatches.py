
import cv2
import numpy as np
import ransac
RATIO = 0.7
TRESHOLD = 4


def showInliers(imageA, imageB):

    (kpsA, featuresA) = detectAndDescribe(imageA)
    (kpsB, featuresB) = detectAndDescribe(imageB)

    # match features between the two images
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB)

    # if the match is None, then there aren't enough matched
    # keypoints to create a panorama
    if M is None:
        return None
    (matches, H, status, inliers) = M

    # check to see if the keypoint matches should be visualized
    vis = displayMatches(imageB, imageA, kpsA, kpsB, matches,
                         status, inliers)
    # return a tuple of the stitched image and the
    # visualization
    return vis

    # return the stitched image


def detectAndDescribe(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check to see if we are using OpenCV 3.X
    # detect and extract features from the image
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)

    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])

    # return a tuple of keypoints and features
    return (kps, features)


def matchKeypoints(kpsA, kpsB, featuresA, featuresB):
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    matchesForHomo = []

    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * RATIO:
            matches.append((m[0].trainIdx, m[0].queryIdx))
            matchesForHomo.append(m[0])

    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, TRESHOLD)
        H, inliers = ransac.ransacHomography(
            kpsA, kpsB, matchesForHomo, 1000, 4)
        # return the matches along with the homograpy matrix
        # and status of each matched point
        return (matches, H, status, inliers)

    # otherwise, no homograpy could be computed
    return None


def displayMatches(imageA, imageB, kpsA, kpsB, matches, status, inliers):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        # if it's a key point and inlier
        if s == 1 and (trainIdx, queryIdx) in inliers:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (50, 255, 255), 1)             # ( blue , green , red )
        # if it's key point and out lier
        elif s == 1:
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (255, 0, 0), 1)

    # return the visualization
    return vis
