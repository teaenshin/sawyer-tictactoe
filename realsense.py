import pyrealsense2 as rs
import numpy as np
import cv2


def get_center_white(color_image):
    hsl = cv2.cvtColor(color_image, cv2.COLOR_BGR2HLS)

    lower_hsl = np.array([0, 0, 200])  
    upper_hsl = np.array([180, 255, 255]) 

    # Threshold the image to get only cup colors
    mask = cv2.inRange(hsl, lower_hsl, upper_hsl)

    y_coords, x_coords = np.nonzero(mask)

    # If there are no detected points, exit
    if len(x_coords) == 0 or len(y_coords) == 0:
        print("No points detected. Is your color filter wrong?")
        return None

    # Calculate the center of the detected region by 
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    return center_x , center_y



# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite('imgs/cam2.jpg',color_image)
        center_coords = get_center_white(color_image)
        if center_coords:
            cv2.circle(color_image, center_coords, 5, (0, 0, 255), -1)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:
    pipeline.stop()