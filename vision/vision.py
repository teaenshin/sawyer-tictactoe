import cv2
import numpy as np
from PIL import Image
import pyrealsense2 as rs

# Configure depth and color streams for the Real Sense Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline_started = False
###########

def get_whiteboard(color_image):
    '''
    Takes in a color_image input and detects the whiteboard by finding the largest contour by area.
    If no largest contour detected, will return None
    '''
    hsl = cv2.cvtColor(color_image, cv2.COLOR_BGR2HLS)

    lower_hsl = np.array([0, 0, 0])  
    upper_hsl = np.array([180, 255, 35]) 

    mask = cv2.inRange(hsl, lower_hsl, upper_hsl)
    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there are no contours, return None
    if not contours:
        return None

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
    else:
        center_x, center_y = 0, 0
    return center_x, center_y

def crop_image(image, contour):
    '''
    Crops image to a given contour
    '''
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def processBoard(debug=True):
    '''
    Identifies and crops to whiteboard. Identifies and crops to tictactoe grid in whiteboard.
    Returns a binary image of the grid warped to top down view. 
    '''
    img = cv2.imread('imgs/cam4.jpg')
    # resized_frame = cv2.resize(img, (300, 300))  # TODO # Resize the image for consistency
    resized_frame = img
    ### Crop 
    whiteboard = get_whiteboard(resized_frame)
    cropped_image = crop_image(resized_frame, whiteboard)
    # TODO: let user manually check if cropped image is correct

    ### Preproccess img
    # Convert to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # bi = cv2.bilateralFilter(gray, 5, 75, 75)

    _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    
    thresh = 255 - thresh # invert so grid becomes white

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
    # Draw the polygon on the original image
    image_with_polygon = cropped_image.copy()
    for corner in corners:
        cv2.circle(image_with_polygon, corner, 5, (0, 0, 255), -1)  # -1 fills the circle with the specified color
    # cv2.drawContours(image_with_polygon, [approx_polygon], -1, (0, 255, 0), 2)

    ### warp grid into square
    # Define the four corners of the target square
    target_size = 300
    target_corners = np.array([[0, 0], [0, target_size - 1], [target_size - 1, 0], [target_size - 1, target_size - 1], ], dtype=np.float32) # (top left, top right, bottom left, bottom right)
    corners = np.float32(corners) # convert to np.float32 for cv2.warpPerspective
    corners = sorted(corners, key=lambda x: x[1])
    corners = sorted(corners, key=lambda x: x[0]) # sort by row
    corners = np.array(corners)

    # TODO: figure out what to do when the grid is not detected
    assert corners.shape == (4, 2), "there should be 4 corners for the grid"
    # Calculate the perspective transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(corners, target_corners)
    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(thresh, transformation_matrix, (target_size, target_size))
    _, warped_image = cv2.threshold(warped_image, 128, 255, cv2.THRESH_BINARY)


    
    return warped_image
    # Display the resulting frame

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

def getCamera():
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Display the processed image with grid lines
        cv2.imshow('original', frame)
        cv2.imwrite('imgs/cam1.jpg',frame)
        # Set the frame rate you want to process (1 frame per second).
        fps = 10
        delay = int(1000 / fps)  # Delay in milliseconds
        if cv2.waitKey(delay) == ord('q'):
            print("Stopped video processing")
            break
    # Release the video capture and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()


def get_color_image():
    if not pipeline_started:
        pipeline.start(config)
        pipeline_started = True
    color_image = None
    i = 0
    while not color_image and i < 5:
        i+=1
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
    return color_image


def is_oval(contour, tolerance=0.1):
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

def is_circle(contour, tolerance=0.1):
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
    cv2.imshow("contour Image", contour_image)
    cv2.waitKey(0)


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

def getBoard(cropped_image):
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # bi = cv2.bilateralFilter(gray, 5, 75, 75)

    _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    
    thresh = 255 - thresh # invert so grid becomes white

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
    # Draw the polygon on the original image
    image_with_polygon = cropped_image.copy()
    for corner in corners:
        cv2.circle(image_with_polygon, corner, 5, (0, 0, 255), -1)  # -1 fills the circle with the specified color
    # cv2.drawContours(image_with_polygon, [approx_polygon], -1, (0, 255, 0), 2)

    ### warp grid into square
    # Define the four corners of the target square
    target_size = 300
    target_corners = np.array([[0, 0], [0, target_size - 1], [target_size - 1, 0], [target_size - 1, target_size - 1], ], dtype=np.float32) # (top left, top right, bottom left, bottom right)
    corners = np.float32(corners) # convert to np.float32 for cv2.warpPerspective
    corners = sorted(corners, key=lambda x: x[1])
    corners = sorted(corners, key=lambda x: x[0]) # sort by row
    corners = np.array(corners)
    board = [""]*9
    if corners.shape !=(4, 2):
        return board
    # Calculate the perspective transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(corners, target_corners)
    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(thresh, transformation_matrix, (target_size, target_size))
    _, warped_image = cv2.threshold(warped_image, 128, 255, cv2.THRESH_BINARY)

    cells = getGridCells(warped_image)

    
    for i in range(9):
        val = identifyCell(cells[i])
        board[i] = val
    return board


def main():
    #create a 1d array to hold the gamestate
    gamestate = np.array([None,None,None,
                        None, None,None,
                        None,None,None])

    # Read in video feed 
    warped_grid = processBoard()
    cells = getGridCells(warped_grid)
    for i in range(len(cells)):
        cell = cells[i]
        x = identifyCell(cell)
        print('cell type', x)
    # processCells(warped_board)
    # getCamera()
    
    '''
    if board is done drawing:
        while not isGameOver(state): 
            
    '''
    

if __name__ == "__main__":
    main()