import cv2
import numpy as np
import math


class Converters:

    @staticmethod
    def unit_frame_test(image, converter):
        cv2.imshow("before", image)
        image = converter(image)
        cv2.imshow(
            f"after '{converter.__name__}' dtype='{image.dtype}' shape='{image.shape}'", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def remove_ground(image):
        res = np.zeros(image.shape[:2], np.uint8)

        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                blue = image[i, j, 0]
                green = image[i, j, 1]
                red = image[i, j, 2]

                if (green > red > blue):
                    res[i, j] = 0
                else:
                    res[i, j] = 255

        return res

    @staticmethod
    def laplacianGradient(image):
        laplacian = cv2.Laplacian(image, cv2.CV_8U)
        return laplacian

    @staticmethod
    def sobelGradient(image):
        sobelx = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=3)
        sobel_res = sobelx + sobely
        return sobel_res

    @staticmethod
    def cannyEdgeDetector(image, threshold1, threshold2):
        return cv2.Canny(image, threshold1=threshold1, threshold2=threshold2, L2gradient=True)

    @staticmethod
    def applyCircleHoughTransform(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=100, param2=30,
                                   minRadius=1, maxRadius=30)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                image = cv2.circle(image, center, 1, (0, 0, 0), 3)
                # circle outline
                radius = i[2]
                image = cv2.circle(image, center, radius, (0, 0, 0), 3)

        return image

    @staticmethod
    def applyLineHoughTransform(image, ccanny):
        lines = cv2.HoughLines(image, 1, np.pi / 180, 150, None, 0, 0)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(ccanny, pt1, pt2, (0, 0, 0), 30, cv2.FILLED)
        return ccanny

    @staticmethod
    def dilateAndErode(image):
        image = cv2.dilate(image,
                           kernel=cv2.getStructuringElement(
                               shape=cv2.MORPH_RECT, ksize=(3, 3)),
                           iterations=3)
        image = cv2.erode(image,
                          kernel=cv2.getStructuringElement(
                              shape=cv2.MORPH_RECT, ksize=(3, 3)),
                          iterations=1)
        return image

    @staticmethod
    def convert_detect(image):
        ground_removed = Converters.remove_ground(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel = Converters.sobelGradient(gray_image)

        res = sobel + ground_removed
        
        res = cv2.threshold(res, thresh=100, maxval=255,
                            type=cv2.THRESH_BINARY)[1]
        
        res = Converters.dilateAndErode(res)

        canny = Converters.cannyEdgeDetector(res, 50, 200)
        ccanny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        res = Converters.applyLineHoughTransform(canny, ccanny)

        res = Converters.applyCircleHoughTransform(res)

        contours, _ = cv2.findContours(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = cv2.drawContours(image, contours, -1, (0, 0, 255), 2) 

        return res


def write_video(video_filename: str, save_filename: str):

    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print("Error: Could not open input video file")
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out = cv2.VideoWriter(save_filename, fourcc, fps, frame_size)

    if not out.isOpened():
        print("Error: Could not open output video file")
        cap.release()
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        converted_frame = Converters.convert_detect(frame)
        out.write(converted_frame)

    print(f"Success in creating \"{save_filename}\"")
    cap.release()
    out.release()


if __name__ == "__main__":

    #test_frame = cv2.imread("DSP/test_frame.jpg")
    #Converters.unit_frame_test(test_frame, Converters.convert_detect)

    write_video(video_filename="DSP/f2002bvsg.mp4",
               save_filename="DSP/output.avi")
