# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
import math
import transformation

# custom imports
import rotation_matrix_util as rmu
import client


def handle_drone_directions(img, rect):
    ###### Face from center of frame ######

    if dlib.rectangle.contains(rect, int(img.shape[1] / 2), int(img.shape[0] / 2)) is False:

        directions = ["0", "0", "0", "0"]  # up, right, down, left
        # right/left cases
        if dlib.rectangle.left(rect) < int(img.shape[1] / 2) - dlib.rectangle.width(rect):
            directions[1] = "right"
        elif dlib.rectangle.right(rect) > int(img.shape[1] / 2) + dlib.rectangle.width(rect):
            directions[3] = "left"
        # up/down cases
        if dlib.rectangle.bottom(rect) < int(img.shape[0] / 2):
            directions[0] = "up"
        elif dlib.rectangle.top(rect) > int(img.shape[0] / 2):
            directions[2] = "down"

        final_directions = list(filter(lambda x: x != "0", directions))
        if len(final_directions) > 1:
            cv2.putText(img, 'Need to move drone {0} {1}'.format(final_directions[0], final_directions[1]),
                        (5, 200), font, 1, (255, 255, 255), 2)
        elif len(final_directions) == 1:
            cv2.putText(img, 'Need to move drone {0}'.format(final_directions[0]),
                        (5, 200), font, 1, (255, 255, 255), 2)
            ###### Face from center of frame ######

        return final_directions

def draw_lines(head_pose_data, img):

    # cv2.line(frame, head_pose[0], head_pose[1], (255, 0, 0), 2)
    cv2.line(img, tuple(head_pose_data[0][0][0].astype(int)), tuple(head_pose_data[0][1][0].astype(int)), (0, 0, 255))
    cv2.line(img, tuple(head_pose_data[0][1][0].astype(int)), tuple(head_pose_data[0][2][0].astype(int)), (0, 0, 255))
    cv2.line(img, tuple(head_pose_data[0][2][0].astype(int)), tuple(head_pose_data[0][3][0].astype(int)), (0, 0, 255))
    cv2.line(img, tuple(head_pose_data[0][3][0].astype(int)), tuple(head_pose_data[0][0][0].astype(int)), (0, 0, 255))
    cv2.line(img, tuple(head_pose_data[0][4][0].astype(int)), tuple(head_pose_data[0][5][0].astype(int)), (0, 0, 255))
    cv2.line(img, tuple(head_pose_data[0][5][0].astype(int)), tuple(head_pose_data[0][6][0].astype(int)), (0, 0, 255))
    cv2.line(img, tuple(head_pose_data[0][6][0].astype(int)), tuple(head_pose_data[0][7][0].astype(int)), (0, 0, 255))
    cv2.line(img, tuple(head_pose_data[0][7][0].astype(int)), tuple(head_pose_data[0][4][0].astype(int)), (0, 0, 255))
    cv2.line(img, tuple(head_pose_data[0][0][0].astype(int)), tuple(head_pose_data[0][4][0].astype(int)), (0, 0, 255))
    cv2.line(img, tuple(head_pose_data[0][1][0].astype(int)), tuple(head_pose_data[0][5][0].astype(int)), (0, 0, 255))
    cv2.line(img, tuple(head_pose_data[0][2][0].astype(int)), tuple(head_pose_data[0][6][0].astype(int)), (0, 0, 255))
    cv2.line(img, tuple(head_pose_data[0][3][0].astype(int)), tuple(head_pose_data[0][7][0].astype(int)), (0, 0, 255))

def head_pose_estimate(frame, shape):

    size = frame.shape
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                            (shape[17][0], shape[17][1]),
                            (shape[21][0], shape[21][1]),
                            (shape[22][0], shape[22][1]),
                            (shape[26][0], shape[26][1]),
                            (shape[36][0], shape[36][1]),
                            (shape[39][0], shape[39][1]),
                            (shape[42][0], shape[42][1]),
                            (shape[45][0], shape[45][1]),
                            (shape[31][0], shape[31][1]),
                            (shape[30][0], shape[30][1]),
                            (shape[35][0], shape[35][1]),
                            (shape[48][0], shape[48][1]),
                            (shape[54][0], shape[54][1]),
                            (shape[57][0], shape[57][1]),
                            (shape[8][0], shape[8][1]),
                        ], dtype="double")

    # 3D model points. https://github.com/lincolnhard/head-pose-estimation
    model_points = np.array([
        (6.825897, 6.760612, 4.402142),
        (1.330353, 7.122144, 6.903745),
        (-1.330353, 7.122144, 6.903745),
        (-6.825897, 6.760612, 4.402142),
        (5.311432, 5.485328, 3.987654),
        (1.789930, 5.393625, 4.413414),
        (-1.789930, 5.393625, 4.413414),
        (-5.311432, 5.485328, 3.987654),
        (2.005628, 1.409845, 6.165652),
        (0.0, 1.409845, 6.165652),
        (-2.005628, 1.409845, 6.165652),
        (2.774015, -2.080775, 5.048531),
        (-2.774015, -2.080775, 5.048531),
        (0.000000, -3.116408, 6.097667),
        (0.000000, -7.415691, 4.070434)
    ])

    proj=10.0
    reproject_matrix = np.array([(proj, proj, proj),
                                 (proj, proj, -proj),
                                 (proj, -proj, -proj),
                                 (proj, -proj, proj),
                                 (-proj, proj, proj),
                                 (-proj, proj, -proj),
                                 (-proj, -proj, -proj),
                                 (-proj, -proj, proj)])

    #reproject_matrix = np.array([(0.0, 1.409845, 1000.0)])

    camera_matrix = np.array(
                         [[1065.998192050825, 0.0, 650.5364868504282],
                         [0.0, 1068.49376227235, 333.59792728394547],
                         [0.0, 0.0, 1.0]], dtype = "double"
                         )

    params = {}
    # print("Camera Matrix: \n {0}".format(camera_matrix))

    #dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    dist_coeffs = np.array([0.15166761845099383, 0.018422099273101917, -0.037319705442136746, 0.02048992115919593, -0.2574454398381191])
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)


    print("Rotation Vector: \n {0}".format(rotation_vector))
    params["rotation_vector"] = np.concatenate(rotation_vector, axis=0).tolist()

    params["euler_angles"] = {}

    # print("Rotation Euler angles (Radians): \n {0}".format(rmu.rotationMatrixToEulerAngles(cv2.Rodrigues(rotation_vector)[0])))
    # params["euler_angles"]["radians"] = rmu.rotationMatrixToEulerAngles(cv2.Rodrigues(rotation_vector)[0]).tolist()
    # or use this
    print("Rotation Euler angles (Radians): \n {0}".format(
        np.asarray(transformation.euler_from_matrix(cv2.Rodrigues(rotation_vector)[0], 'sxyz'))))
    params["euler_angles"]["radians"] = np.asarray(
        transformation.euler_from_matrix(cv2.Rodrigues(rotation_vector)[0], 'sxyz'))

    print("Rotation Euler angles (Degrees): \n {0}".format(rmu.rotationMatrixToEulerAngles(cv2.Rodrigues(rotation_vector)[0]) * (180/PI)))
    params["euler_angles"]["degrees"] = (rmu.rotationMatrixToEulerAngles(cv2.Rodrigues(rotation_vector)[0]) * (180/PI)).tolist()

    print("Translation Vector: \n {0}".format(translation_vector))
    params["translation_vector"] = np.concatenate(rotation_vector, axis=0).tolist()

    params["camera_position"] = -np.matrix(cv2.Rodrigues(rotation_vector)[0]).T * np.matrix(translation_vector)
    print("Camera Position: \n {0}".format(params["camera_position"]))


    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    (nose_end_point2D, jacobian) = cv2.projectPoints(reproject_matrix, rotation_vector, translation_vector, camera_matrix,
                                                     dist_coeffs, None, None, cv2.CALIB_FIX_ASPECT_RATIO)

    return [nose_end_point2D, params]


def execute(count, skip_frames):
    # loop over the frames from the video stream
    while True:

        count += 1
        if count > 1000000:
            count = 1
        if count % skip_frames == 0:
            continue
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 800 pixels, and convert it to
        # grayscale
        frame = vs.read()
        frame = imutils.resize(frame, height=720)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        #final directions
        final_directions = []
        distancei = 0

        # loop over the face detections
        for rect in rects:
            # calculate distance from camera
            w = dlib.rectangle.right(rect) - dlib.rectangle.left(rect)
            h = dlib.rectangle.bottom(rect) - dlib.rectangle.top(rect)
            distancei = (2 * math.pi * 180) / (w + h * 360) * 1000 + 3
            cv2.putText(frame, 'Distance = ' + str(distancei * 3) + ' cm', (5, 100), font, 1, (255, 255, 255), 2)

            # draw rectangle around face
            cv2.rectangle(frame, (dlib.rectangle.left(rect), dlib.rectangle.top(rect)),
                          (dlib.rectangle.right(rect), dlib.rectangle.bottom(rect)), (0, 255, 0), 2)

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # handle frame directions
            final_directions = handle_drone_directions(frame, rect)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # calculate head pose estimation
            head_pose = head_pose_estimate(frame, shape)
        # draw lines
        draw_lines(head_pose, frame)

        head_pose[1]["direction"] = final_directions
        #print(head_pose[2])
        if client_socket is not None:
            client_socket.socket_send(
                str.encode(str(head_pose[2])))

        # degree from centre of image
        frame_center = (int(frame.shape[1]/2), int(frame.shape[0]/2))

        cv2.circle(frame, frame_center, 1, (0, 0, 255), -1)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            if client_socket is not None:
                client_socket.socket_close()  # close socket and terminate C++ socket server
            break


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()
time.sleep(2.0)

FRAMES_SKIP = 50;
COUNT = 0

PI = math.pi
font = cv2.FONT_HERSHEY_SIMPLEX

# initiate socket
client_socket = None
try:
    client_socket = client.create_client_socket()
except ConnectionRefusedError:
    print("[INFO] Connection with C++ server refused.. proceeding without server")
    time.sleep(2.0)

try:
    execute(COUNT, FRAMES_SKIP)
except KeyboardInterrupt:
    if client_socket is not None:
        client_socket.socket_close()  # close socket and terminate C++ socket server
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()