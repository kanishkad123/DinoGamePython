# Followed blog : https://medium.com/@harshilp/playing-chromes-dinosaur-game-using-opencv-19b3cf9c3636

import numpy as np
import cv2
import math
import pyautogui

capture = cv2.VideoCapture(0)
while capture.isOpened():
    ret, frame = capture.read()
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    crop_image = frame[100:300, 100:300]

    # https://www.tutorialspoint.com/opencv/opencv_gaussian_blur.htm
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # https://www.geeksforgeeks.org/python-opencv-cv2-cvtcolor-method/ Change color after movement detection
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # https://stackoverflow.com/questions/48109650/how-to-detect-two-different-colors-using-cv2-inrange-in-python-opencv
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html to help with false detection
    kernel = np.ones((5, 5))
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
	# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

	# https://pythonexamples.org/python-opencv-cv2-find-contours-in-image/
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
		# Put a rectangle around the object and draw
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)
        hull = cv2.convexHull(contour)
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Fi convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # TODO: if there are birds in the game dino will not duck
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 255, 0], 2)

        # Press SPACE if condition is match
        if count_defects >= 4:
            pyautogui.press('space')
            cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)

    except:
        pass

    cv2.imshow("Gesture", frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
