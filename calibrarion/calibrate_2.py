import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# ret should be under 0.5 for best calib or >0.5>1.0 for good calib

#Compute mean of reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

with open('calib_result.txt', 'a') as the_file:
    the_file.write('Camera Matrix: \n')
    the_file.write(str(np.concatenate(mtx, axis=0).tolist()) + '\n')
    the_file.write('Distortion Coefficients: \n')
    the_file.write(str(np.concatenate(dist, axis=0).tolist()) + '\n')
    the_file.write('Rotation Vector: \n')
    the_file.write(str(np.concatenate(rvecs, axis=0).tolist()) + '\n')
    the_file.write('Translation Vector: \n')
    the_file.write(str(np.concatenate(tvecs, axis=0).tolist()) + '\n')
    the_file.write("Mean reprojection error:\n" + str(mean_error/len(objpoints)) + '\n')
    the_file.write('--------------------------------')
