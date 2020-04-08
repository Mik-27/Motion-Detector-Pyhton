from datetime import datetime

import cv2

first_frame = None
motion_list = [None, None]
times = []

# Capture the video from the system webcam
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

try:
    while True:
        check, frame = video.read()
        motion = 0
        # Grayscale the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur the frame to better spot the differences
        gary = cv2.GaussianBlur(gray, (21, 21), 0)

        # Storing the first frame
        if first_frame is None:
            first_frame = gray
            continue

        # Comparing the first frame with the current frame
        diff_frame = cv2.absdiff(first_frame, gray)

        # If the difference between first and current frame is greater than 30 the it will show white(255) color
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        # Dilating the Threshold frame
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        # Get contours of the threshold frame i.e. moving object
        (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 10000:
                continue

            motion = 1

            # Set the rectangle for the contours of moving objecct
            (x, y, w, h) = cv2.boundingRect(contour)

            # Display the rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        motion_list.append(motion)
        motion_list = motion_list[-2:]

        # Appending Start rime of motion
        if motion_list[-1] == 1 and motion_list[-2] == 0:
            times.append(datetime.now())

        # Appending End time of motion
        if motion_list[-2] == 1 and motion_list[-1] == 0:
            times.append(datetime.now())

        # Displaying each frame individually
        cv2.imshow("Gray Frame", gray)
        cv2.imshow("Difference Frame", diff_frame)
        cv2.imshow("Threshold Frame", thresh_frame)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        # 'q' to exit
        if key == ord("q"):
            if motion == 1:
                times.append(datetime.now())
            break
except cv2.error:
    print("Error")
except ValueError:
    print("Error")
finally:
    video.release()
    cv2.destroyAllWindows()
