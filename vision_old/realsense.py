import pyrealsense2 as rs
import numpy as np
import cv2


def get_center_white(color_image):
    hsl = cv2.cvtColor(color_image, cv2.COLOR_BGR2HLS)

    lower_hsl = np.array([0, 0, 0])
    upper_hsl = np.array([180, 255, 35]) 

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

def get_whiteboard(color_image):
    hsl = cv2.cvtColor(color_image, cv2.COLOR_BGR2HLS)

    lower_hsl = np.array([0, 0, 0])  
    upper_hsl = np.array([180, 255, 35]) 

    # Threshold the image to get only cup colors
    mask = cv2.inRange(hsl, lower_hsl, upper_hsl)
    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there are no contours, return None
    if not contours:
        return None

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def crop_image(image, contour):
    
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image


def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
    else:
        center_x, center_y = 0, 0
    return center_x, center_y



ctx = rs.context()
if len(ctx.devices) > 0:

    for d in ctx.devices:

        print ('Found device: ', \

                d.get_info(rs.camera_info.name), ' ', \

                d.get_info(rs.camera_info.serial_number))

else:

    print("No Intel Device connected")
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
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
        print("hi")
        
        largest_contour = get_whiteboard(color_image)
        center_coords = get_contour_center(largest_contour)

        # If there is a largest contour, draw it on the image
        if largest_contour is not None:
            cv2.drawContours(color_image, [largest_contour], -1, (0, 255, 0), 3)
            # cropped_image = crop_image(color_image, largest_contour)
            # cv2.namedWindow('Cropped', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('Cropped', cropped_image)


        if center_coords:
            cv2.circle(color_image, center_coords, 5, (0, 0, 255), -1)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:
    pipeline.stop()