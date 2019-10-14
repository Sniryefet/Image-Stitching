import math
import random

import cv2
import numpy as np

'''
This class was used to calculate the homography using ransac, but now it is used to create the inliners/outliers.
'''

def computeHomography(f1, f2, matches):

    num_matches = len(matches)

    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows, num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]

        (a_x, a_y) = f1[m.queryIdx]
        (b_x, b_y) = f2[m.trainIdx]

        A[i*2, 0] = a_x
        A[i*2, 1] = a_y
        A[i*2, 2] = 1
        A[i*2, 6] = -b_x * a_x
        A[i*2, 7] = -b_x * a_y
        A[i*2, 8] = -b_x

        A[i*2 + 1, 3] = a_x
        A[i*2 + 1, 4] = a_y
        A[i*2 + 1, 5] = 1
        A[i*2 + 1, 6] = -b_y * a_x
        A[i*2 + 1, 7] = -b_y * a_y
        A[i*2 + 1, 8] = -b_y

    U, s, Vt = np.linalg.svd(A)

    # s is a 1-D array of singular values sorted in descending order
    # U, Vt are unitary matrices
    # Rows of Vt are the eigenvectors of A^TA.
    # Columns of U are the eigenvectors of AA^T.

    H = np.eye(3)

    eig_shape = Vt.shape
    h = Vt[eig_shape[0]-1]
    H_len = len(H)
    for i in range(H_len):
        for j in range(H_len):
            H[i, j] = h[H_len * i + j]

    return H


def ransacHomography(f1, f2, matches,  nRANSAC, RANSACthresh):
    '''
    Loop that calculates H using RANSAC and inliners
    '''
    num_inliers = -1
    best_estimate = []
    for i in range(nRANSAC):
        matches_sub = np.random.choice(matches, 4, replace=False)
        H = computeHomography(f1, f2, matches_sub)

        inlier_indices = getInliers(f1, f2, matches, H, RANSACthresh)
        if (len(inlier_indices) > num_inliers):
            num_inliers = len(inlier_indices)
            best_estimate = inlier_indices
    # For matches
    inliers = []
    for i in range(len(inlier_indices)):
        inliers.append((matches[inlier_indices[i]].trainIdx, matches[inlier_indices[i]].queryIdx))


    H=leastSquaresFit(f1, f2, matches,  best_estimate)
    return H, inliers


def getInliers(f1, f2, matches, M, RANSACthresh):
    inlier_indices=[]

    for i in range(len(matches)):

        x_f1, y_f1=f1[matches[i].queryIdx]
        x_f2, y_f2=f2[matches[i].trainIdx]

        mat_x=np.zeros((3, 1))

        mat_x[2]=1
        mat_x[1]=y_f1
        mat_x[0]=x_f1

        y=np.dot(M, mat_x)

        if (np.sqrt(((y[0]/y[2])-x_f2)**2 + ((y[1]/y[2])-y_f2)**2) <= RANSACthresh):
            inlier_indices.append(i)

    return inlier_indices


def leastSquaresFit(f1, f2, matches,  inlier_indices):
    '''
    Compute a homography M using all inliers.
    Compute the transformation matrix from f1 to f2 using only the
    inliers and return it.
    '''
    M=np.eye(3)

    best=[]
    inlier_length=len(inlier_indices)
    for i in range(inlier_length):
        best.append(matches[inlier_indices[i]])
    M=computeHomography(f1, f2, best)

    return M
