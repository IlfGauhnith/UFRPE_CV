from main import Converters
import cv2

def write_video_debug(save_filename: str, fourcc, fps, frame_size, frames):
    out = cv2.VideoWriter(save_filename, fourcc, fps, frame_size)

    if not out.isOpened():
        print("Error: Could not open output video file")
        return

    while True:
        if not frames:
            break

        frame = frames.pop(0)
        out.write(frame)

    print(f"Success in creating \"{save_filename}\"")

    out.release()

def debug(video_filename: str):
    rmbg_frames = []
    dilate_erode_frames = []
    canny_frames = []
    hough_frames = []

    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print("Error: Could not open input video file")
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        ground_removed = Converters.remove_ground(frame)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel = Converters.sobelGradient(gray_image)
        res = sobel + ground_removed
        res = cv2.threshold(res, thresh=100, maxval=255,
                            type=cv2.THRESH_BINARY)[1]
        
        rmbg_frames.append(cv2.cvtColor(res, cv2.COLOR_GRAY2BGR))

        res = Converters.dilateAndErode(res)
        dilate_erode_frames.append(cv2.cvtColor(res, cv2.COLOR_GRAY2BGR))

        canny = Converters.cannyEdgeDetector(res, 50, 200)
        ccanny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        canny_frames.append(ccanny)

        res = Converters.applyLineHoughTransform(canny, ccanny)
        res = Converters.applyCircleHoughTransform(res)
        hough_frames.append(res)

    cap.release()

    write_video_debug("DSP/debug/rmbg.avi", fourcc, fps, frame_size, rmbg_frames)
    write_video_debug("DSP/debug/dilate_erode.avi", fourcc, fps, frame_size, dilate_erode_frames)
    write_video_debug("DSP/debug/canny.avi", fourcc, fps, frame_size, canny_frames)
    write_video_debug("DSP/debug/hough.avi", fourcc, fps, frame_size, hough_frames)

if __name__ == "__main__":
    debug(video_filename="DSP/f2002bvsg.mp4")
