#!/usr/bin/env python
import cv2
import numpy as np
from PIL import Image

# Configure depth and color streams for the Real Sense Camera

### PROCESS BOARD FROM CAM ###

def get_whiteboard(color_image):
    '''
    Takes in a color_image input and detects the whiteboard by finding the largest contour by area.
    If no largest contour detected, will return None
    '''
    hls = cv2.cvtColor(color_image, cv2.COLOR_BGR2HLS)

    lower_hls = np.array([0, 0, 0])
    upper_hls = np.array([180, 255, 35])

    mask = cv2.inRange(hls, lower_hls, upper_hls)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there are no contours, return None
    if not contours:
        print("Error: there are no contours")
        return None

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

# def get_whiteboard(color_image):
#     '''
#     Takes in a color_image input and detects the whiteboard by finding the largest contour by area.
#     If no largest contour detected, will return None
#     '''
#     hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

#     lower_hsv = np.array([0, 0, 160])  
#     upper_hsv = np.array([180, 80, 255]) 

#     mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
#     # Create a binary mask for the white paper

#     cv2.imshow('mask', mask)
#     cv2.waitKey(0)
#     # Find contours in the masked image
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # If there are no contours, return None
#     if not contours:
#         print("Error: there are no contours")
#         return None

#     # Find the largest contour by area
#     largest_contour = max(contours, key=cv2.contourArea)
#     return largest_contour

# def get_contour_center(contour):
#     M = cv2.moments(contour)
#     if M["m00"] != 0:
#         center_x = int(M["m10"] / M["m00"])
#         center_y = int(M["m01"] / M["m00"])
#     else:
#         center_x, center_y = 0, 0
#     return center_x, center_y

def crop_image(image, contour):
    '''
    Crops image to a given contour
    '''
    whiteout(image, contour)
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def whiteout(image, contour):
    mask = np.zeros_like(image)
    # Draw the contour on the mask with white color and thickness of -1 (fill the contour)
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)
    # Convert the mask to grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Set pixels outside the contour to black in the original image
    image[mask == 0] = [255, 255, 255]

def getBoard(cropped_image, debug=False):
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # bi = cv2.bilateralFilter(gray, 5, 75, 75)

    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    thresh = 255 - thresh # invert so grid becomes white
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)

    ### Get largest contour, which should be the grid
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)
    # Create a mask image for visualization
    mask = cropped_image.copy()
    cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

    epsilon = 0.04 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    ### Get 4 corners of the grid
    corners = approx_polygon.reshape(-1, 2)

    ### warp grid into square
    # Define the four corners of the target square
    target_size = 300
    target_corners = np.array([ [0, 0], [target_size - 1, 0] , [0, target_size - 1], [target_size - 1, target_size - 1] ],  dtype=np.float32) 
    corners = np.float32(corners) # convert to np.float32 for cv2.warpPerspective
    top_corners = sorted(corners, key=lambda x:x[1])[:2]
    top_corners = sorted(top_corners, key=lambda x:x[0])
    bottom_corners = sorted(corners, key=lambda x: x[1])[2:]
    bottom_corners = sorted(bottom_corners, key=lambda x:x[0])
    corners = np.array(top_corners + bottom_corners, dtype=np.float32) # (top left, top right, bottom left, bottom right)
    
    if debug:
        print('corners', corners)
        image_with_polygon = cropped_image.copy()
        c = 50
        for corner in corners:
            center = (int(corner[0]), int(corner[1]))
            cv2.circle(image_with_polygon, center, 5, (0, 0, c), -1)  # -1 fills the circle with the specified color
            c += 50
        cv2.circle(image_with_polygon, (10, 70), 5, (0, 255, 0), -1)  # -1 fills the circle with the specified color
        cv2.drawContours(image_with_polygon, [approx_polygon], -1, (0, 255, 0), 2)
        cv2.imshow('img with poly', image_with_polygon)
        cv2.waitKey(0)

    if corners.shape !=(4, 2):
        cv2.destroyAllWindows()
        print("BOARD NOT DETECTED")
        return None
    # Calculate the perspective transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(corners, target_corners)
    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(thresh, transformation_matrix, (target_size, target_size))
    _, warped_image = cv2.threshold(warped_image, 128, 255, cv2.THRESH_BINARY)
    
    if debug:
        cv2.imshow('warped_image after threshold', warped_image)
        cv2.waitKey(0)

    # cv2.destroyAllWindows() 

    return warped_image

### GET CELLS ###
def getGridCells(warped_grid, margin_percent=15):
    '''
    Input:
    warped_grid: image of the the tictactoe grid warped to top down view
    margin_percent=5: will crop out 10% off each side of each cell, to account for slight differences in grid sizes. want to crop out grid but not shape. 

    Output:
    cells: list of 9, cropped grid cells
    |_0_|_1_|_2_|
    |_3_|_4_|_5_|
    |_6_|_7_|_8_|
    '''

    # make sure wapred_image is binary
    # _, warped_grid = cv2.threshold(warped_grid, 128, 255, cv2.THRESH_BINARY)

    cell_height = warped_grid.shape[0] // 3
    cell_width = warped_grid.shape[1] // 3
    cells = []

    for row in range(3): 
        for col in range(3):
            # coords for top left corner    
            top_left_col = col * cell_width
            top_left_row = row * cell_height

            # Crop the image to the current cell
            cell_image = warped_grid[top_left_row: top_left_row + cell_height, top_left_col : top_left_col + cell_width]
            
            # Crop margin off cell
            crop_width = int(cell_image.shape[1] * margin_percent / 100)
            crop_height = int(cell_image.shape[0] * margin_percent / 100)
            cell_image = cell_image[crop_height:-crop_height, crop_width:-crop_width]

            # Add the cropped cell to the list
            cells.append(cell_image)

    # Create a new image by pasting the grid images
    # new_image = Image.new('RGB', (width, height))
    return cells


def get_line_params(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return (x1, y1, x2, y2)

def calculate_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d:
        xi = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        yi = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
        return int(xi), int(yi)
    else:
        return None

def getGridCellsRobust(warped_grid):
    blur = cv2.GaussianBlur(warped_grid, (3, 3), 0)
    top_bottom_margin = 30  # Margin from top and bottom
    left_right_margin = 30  # Margin from left and right        
    # Apply Canny edge detection
    edges = cv2.Canny(warped_grid, 50, 150, apertureSize=3)

    # Use Hough Transform to find lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

    blank_image = np.zeros(warped_grid.shape, warped_grid.dtype)
    vertical_lines = []
    horizontal_lines = []

    # Draw the lines on the image for visualization
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if abs(b) < 0.5 and (rho < left_right_margin and rho > -warped_grid.shape[1] + left_right_margin):
            append = True
            for prev_line in vertical_lines:
                if abs(prev_line[0][0]-rho) < 30:
                    append = False
            
            if append:
                vertical_lines.append(line)
                cv2.line(blank_image, (x1, y1), (x2, y2), (255, 255, 255), 4)

        elif abs(np.sin(theta)) > 0.5 and (rho > top_bottom_margin and rho < warped_grid.shape[0] - top_bottom_margin):
            append = True
            for prev_line in horizontal_lines:
                if abs(prev_line[0][0]-rho) < 30:
                    append = False
            
            if append:
                horizontal_lines.append(line)
                cv2.line(blank_image, (x1, y1), (x2, y2), (255, 255, 255), 4)

    horizontal_lines_params = [get_line_params(line[0][0], line[0][1]) for line in horizontal_lines]
    vertical_lines_params = [get_line_params(line[0][0], line[0][1]) for line in vertical_lines]

    # Calculate intersections
    intersections = []
    for h_line in horizontal_lines_params:
        for v_line in vertical_lines_params:
            intersect = calculate_intersection(h_line, v_line)
            if intersect:
                intersections.append(intersect)


    intersections.sort(key=lambda x: x[0])
    intersections.sort(key=lambda x:x[1])
    top_left = warped_grid[0:intersections[0][1], 0:intersections[0][0]]
    top_middle = warped_grid[0:intersections[0][1], intersections[0][0]:intersections[1][0]]
    top_right = warped_grid[0:intersections[1][1], intersections[1][0]:]
    middle_left = warped_grid[intersections[0][1]:intersections[2][1], 0:intersections[0][0]]
    middle_middle = warped_grid[intersections[0][1]:intersections[2][1], intersections[0][0]:intersections[1][0]]
    middle_right = warped_grid[intersections[1][1]:intersections[3][1], intersections[1][0]:]
    bottom_left = warped_grid[intersections[2][1]:, 0:intersections[2][0]]
    bottom_middle = warped_grid[intersections[2][1]:, intersections[2][0]:intersections[3][0]]
    bottom_right = warped_grid[intersections[2][1]:, intersections[3][0]:]



    cells = [top_left, top_middle, top_right, 
                middle_left, middle_middle, middle_right, 
                bottom_left, bottom_middle, bottom_right]
    
    # Display each segment in a separate window
    for idx, segment in enumerate(cells):
        window_name = f"Segment {idx+1}"
        cv2.imshow(window_name, segment)
        cv2.waitKey(0)  # Wait for a key press to show the next segment

    cv2.destroyAllWindows() 

    return cells

### PROCESS CELLS ###
def is_oval(contour, tolerance=0.2):
    """
    Determine if a contour is an oval.

    Args:
    contour (np.ndarray): The contour to be checked.
    tolerance (float): Tolerance for the approximation. Higher means less strict.

    Returns:
    bool: True if the contour is an oval, False otherwise.
    """
    # Approximate the contour and check if the shape is closed
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 5:  # Ovals will have many points in their approximation
        return False

    # Fit an ellipse to the contour and compare the shapes
    ellipse = cv2.fitEllipse(approx)
    ellipse_contour = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 1)

    # Compare the fitted ellipse with the original contour
    similarity = cv2.matchShapes(contour, ellipse_contour, 1, 0.0)

    return similarity < tolerance

def is_circle(contour, tolerance=0.2):
    """
    Determine if a contour is a circle.

    Args:
    contour (np.ndarray): The contour to be checked.
    tolerance (float): Tolerance for the circle approximation. Lower value means stricter circle detection.

    Returns:
    bool: True if the contour is a circle, False otherwise.
    """
    # Approximate the contour to a simpler shape
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 5 or len(approx) > 10:
        return False

    # Fit a circle to the contour
    (x, y), radius = cv2.minEnclosingCircle(approx)

    # Create a perfect circle contour for comparison
    center = (int(x), int(y))
    radius = int(radius)
    circle_contour = np.array([[[int(x + radius * np.cos(theta)), int(y + radius * np.sin(theta))]] for theta in np.linspace(0, 2 * np.pi, 360)], dtype=np.int32)

    # Compare the fitted circle with the original contour
    similarity = cv2.matchShapes(contour, circle_contour, 1, 0.0)

    return similarity < tolerance

def is_x(contour):
    ##TODO detect if a contour represents an X as opposed to a line, L, 3 lines, or box
    return True

def only_near_edges(contour, width, height):
    for point in contour:
        x , y = point[0]
        x_block = width*0.1
        y_block = height*0.1
        if x_block<x<width-x_block and y_block<y<height-y_block:
            return False
    return True

def identifyCell(cell):
    '''
    identifies if a cell is empty, 'X', or 'O'
    Input:
    cell: binary,thresholded image of a cell
    '''
    cell_area = cell.shape[0] * cell.shape[1]


    # Find contours in the thresholded image
    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the original image for visualization
    contour_image = np.zeros((cell.shape[0], cell.shape[1], 3))
    cv2.drawContours(contour_image, contours, 0, (255, 0, 0), thickness=cv2.FILLED)
    # cv2.imshow("contour Image", contour_image)
    # cv2.waitKey(0)

    if not contours:
        return ""
   
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)

        threshold_area = 30 # Adjust this if needed
        if area < threshold_area:
            return ""
        
        if is_circle(contour) or is_oval(contour):
            return "O"
        elif not only_near_edges(contour, cell.shape[1], cell.shape[0]):
            return "X"
        
    return ""

def get_state(cells):
    gamestate = np.array([None,None,None,
                        None, None,None,
                        None,None,None])
    for i in range(len(cells)):
        cell = cells[i]
        # cv2.imshow(f'cell {i}', cell)
        # cv2.waitKey(0)
        cell_type = identifyCell(cell)
        gamestate[i] = cell_type
    return gamestate

def main():
    #create a 1d array to hold the gamestate
    gamestate = np.array([None,None,None,
                        None, None,None,
                        None,None,None])

    # Read in video feed 
    # print("test if camera works: ")
    # getCamera()
    # print("camera works")
    img = cv2.imread('/home/cc/ee106a/fa23/class/ee106a-aem/sawyer-tictactoe/src/vision/src/imgs/cam6.jpg')
    # resized_frame = cv2.resize(img, (300, 300))  # TODO # Resize the image for consistency
    resized_frame = img
    ### Crop 
    whiteboard = get_whiteboard(resized_frame)
    cropped_image = crop_image(resized_frame, whiteboard)
    cv2.imshow('cropped', cropped_image)
    cv2.waitKey(0)

    warped_grid = getBoard(cropped_image)
    cv2.imshow('warped_grid', warped_grid)
    cv2.waitKey(0)
    cells = getGridCells(warped_grid)
    # getGridCellsRobust(warped_grid)
    gamestate = get_state(cells)
    print('gamestate', gamestate)
    
    

if __name__ == "__main__":
    main()


#### MISC ###
def getCamera():
    # define a video capture object 
    vid = cv2.VideoCapture(0) 
    if not vid.isOpened():
        print("Error could not open camera")
        exit()
    vid.set(cv2.CAP_PROP_FPS, 30) # set frame rate
    while(True): 
        
        # Capture the video frame 
        # by frame 
        ret, frame = vid.read() 

        if not ret:
            print("Error could not read frame")
            exit()
    
        # Display the resulting frame 
        cv2.imshow('frame', frame) 
        
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 


