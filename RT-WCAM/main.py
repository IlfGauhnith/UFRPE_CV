import cv2 
import numpy as np

cap = cv2.VideoCapture(0) 

if __name__ == "__main__":

    while True:
        _, frame = cap.read() 
        frame = cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # FAST
        ff_detector = cv2.FastFeatureDetector.create()
        key_points = ff_detector.detect(gray)

        # Canny
        canny = cv2.Canny(gray, 50, 150)

        # Hough Transform
        lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        
        # Drawing
        frame = cv2.drawKeypoints(frame, key_points, frame)
        if lines is not None:
            for i in range(0, len(lines)):
                print(lines[i])
                frame = cv2.line(frame, (lines[i][0][0], lines[i][0][1]), 
                                 (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 1, cv2.LINE_4)

        cv2.imshow("Result", frame) 

        if cv2.waitKey(1) and 0xFF == ord('q'):  
            break
