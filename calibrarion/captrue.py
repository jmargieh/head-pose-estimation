import numpy as np
import cv2

# Global variables
cap = cv2.VideoCapture(0)
global_count = 0
cap_count = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.putText(frame, 'Images captured: ' + str(cap_count), (5, 100), font, 1, (255, 255, 255), 2)

    if global_count % 30 == 0 :
        cv2.imwrite('frame%d.jpg' % cap_count, frame)
        cap_count += 1
        print('image saved!')

    # Global count to skip frames
    global_count += 1

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()