import numpy as np
import cv2

def calculate (objp, imgp, grayp):
    print("calculating calibration result...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, grayp.shape[::-1], None, None)

    # Compute mean of reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
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
        the_file.write("Mean reprojection error:\n" + str(mean_error / len(objpoints)) + '\n')
        the_file.write('--------------------------------')


# Global variables
cap = cv2.VideoCapture(0)
global_count = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 75, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if global_count % 30 != 0:
        # Global count to skip frames
        global_count += 1
        continue


    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        frame = cv2.drawChessboardCorners(frame, (9, 6), corners2, ret)

    cv2.imshow('img', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        cv2.destroyAllWindows()
        calculate(objpoints, imgpoints, gray)
