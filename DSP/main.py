import cv2
import os
import numpy as np

class Converters:

    @staticmethod
    def unit_frame_test(image, converter):
        cv2.imshow("before", image)
        cv2.imshow(f"after '{converter.__name__}'", converter(image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def remove_ground(image):
        res = np.zeros(image.shape)

        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                blue = image[i, j, 0]
                green = image[i, j, 1]
                red = image[i, j, 2]

                if (green > red > blue):
                    res[i, j] = (0, 0, 0)
                else:
                    res[i, j] = (255, 255, 255)
                 
        return res

    @staticmethod
    def convert_detect(image):
        ground_removed = Converters.remove_ground(image)
        return ground_removed


def write_video(video_filename: str, save_filename: str):

    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print("Error: Could not open input video file")
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

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
    
    #test_frame = cv2.imread("C:\\Users\\lucas\\BCC\\CV\\DSP\\test_frame.jpg")
    #Converters.unit_frame_test(test_frame, Converters.remove_ground)

    write_video(video_filename="C:\\Users\\lucas\\BCC\\CV\\DSP\\f2002bvsg.mp4", save_filename="C:\\Users\\lucas\\BCC\\CV\\DSP\\output.mp4")
