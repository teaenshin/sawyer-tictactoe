import cv2
import numpy as np
from PIL import Image
'''
def identifyLines():
    cap = cv2.VideoCapture(0)
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
        
        # Preprocess the image (resize, convert to grayscale, etc.)
        resized_frame = cv2.resize(frame, (300, 300))  # Resize the image for consistency
        # Convert to HSV (Hue, Saturation, Value) color space to facilitate color filtering
        hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

        # Define range of black color in HSV
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 100])  # You might need to adjust these values

        # Threshold the HSV image to get only black colors
        black_mask = cv2.inRange(hsv, lower_black, upper_black)

        # Bitwise-AND mask and original image to isolate the color
        result = cv2.bitwise_and(resized_frame, resized_frame, mask=black_mask)
        cv2.imshow('result mask', result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        # print('edges', edges)

        # Find lines using the Hough Line Transform
        # HoughLinesP(lines, pixel accuracy, angle accuracy, threshold/min num points on line)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)  
        # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        # print('lines', lines)
    source ~ee106a/sawyer_setup.bash
        # Identify grid lines (vertical and horizontal)
        horizontal_lines = []
        vertical_lines = []
        # print('horizontal_lines', horizontal_lines)
        # print('vertical_lines', vertical_lines)
        # for x1,y1,x2,y2 in lines[0]: # TODO: for HoughLinesP
        #     cv2.line(resized_frame,(x1,y1),(x2,y2),(0,255,0),2)

        # for HoughLines() 
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if np.pi / 4 < theta < 3 * np.pi / 4:  # Horizontal lines
                    horizontal_lines.append(line)
                else:  # Vertical lines
                    vertical_lines.append(line)
        # Draw the grid lines on the image
        for line in horizontal_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for line in vertical_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cvsource ~ee106a/sawyer_setup.bash2.line(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display the processed image with grid lines
        cv2.imshow('original', frame)
        cv2.imshow('Tic-Tac-Toe Grid Detection', resized_frame)
        # cv2.waitKey(0)
        
        # Set the frame rate you want to process (1 frame per second).
        fps = 10
        delay = int(1000 / fps)  # Delay in milliseconds
        if cv2.waitKey(delay) == ord('q'):
            print("Stopped video processing")
            break
        
    # Release the video capture and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()
'''

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
    img = cv2.imread('../imgs/cam4.jpg')
    # resized_frame = cv2.resize(img, (300, 300))  # TODO # Resize the image for consistency
    resized_frame = img
    ### Crop 
    whiteboard = get_whiteboard(resized_frame)
    cropped_image = crop_image(resized_frame, whiteboard)
    # TODO: let user manually check if cropped image is correct

    ### Preproccess img
    # Convert to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    # bi = cv2.bilateralFilter(gray, 5, 75, 75)
    # cv2.imshow('bi',bi)

    _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", thresh)
    
    thresh = 255 - thresh # invert so grid becomes white
    #cv2.imshow('threshold', thresh)

    ### Get largest contour, which should be the grid
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)
    # Create a mask image for visualization
    mask = cropped_image.copy()
    cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
    cv2.imshow('largest contour', mask)

    epsilon = 0.04 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    ### Get 4 corners of the grid
    corners = approx_polygon.reshape(-1, 2)
    print(corners)
    # Draw the polygon on the original image
    image_with_polygon = cropped_image.copy()
    for corner in corners:
        cv2.circle(image_with_polygon, corner, 5, (0, 0, 255), -1)  # -1 fills the circle with the specified color
    # cv2.drawContours(image_with_polygon, [approx_polygon], -1, (0, 255, 0), 2)
    cv2.imshow('largest contour with corners ', image_with_polygon)

    ### warp grid into square
    # Define the four corners of the target square
    target_size = 300
    target_corners = np.array([[0, 0], [target_size - 1, 0], [target_size - 1, target_size - 1], [0, target_size - 1]], dtype=np.float32)
    corners = np.float32(corners) # convert to np.float32 for cv2.warpPerspective
    print('cornesrs float', corners)
    # TODO: figure out what to do when the grid is not detected
    assert corners.shape == (4, 2), "there should be 4 corners for the grid"
    # Calculate the perspective transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(corners, target_corners)
    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(thresh, transformation_matrix, (target_size, target_size))
    _, warped_image = cv2.threshold(warped_image, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow('warped_image', warped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    return warped_image
    # Display the resulting frame

def getGridCells(warped_grid, margin_percent=15):
    '''
    Input:
    warped_grid: image of the the tictactoe grid warped to top down view
    margin_percent=5: will crop out 10% off each side of each cell, to account for slight differences in grid sizes

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
            # cv2.imshow(f'cropped cell {row} {col} ', cell_image)
            # cv2.waitKey(0)

    # Create a new image by pasting the grid images
    # new_image = Image.new('RGB', (width, height))
    print
    return cells

def getCamera():
    print('getcamera')
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
        print('frame size', frame.shape)
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

def identifyCell(cell):
    '''
    identifies if a cell is empty, 'X', or 'O'
    Input:
    cell: binary,thresholded image of a cell
    '''
    cell_area = cell.shape[0] * cell.shape[1]
    print('cell area', cell_area)
    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, the cell is empty
    if not contours:
        return 'Empty'
    # Loop through each contour
   
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        print('area', area)

        # Define a threshold for contour area (adjust as needed)
        threshold_area = 100

        # If the contour area is above the threshold, consider it a shape
        if area > threshold_area:
            # Calculate the perimeter of the contour
            perimeter = cv2.arcLength(contour, True)

            # Approximate the contour to a polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Get the number of vertices in the polygon
            num_vertices = len(approx)

            # If it has 4 vertices, it's likely an 'X'
            if num_vertices == 4:
                print("X detected")
            # If it has more than 8 vertices, it's likely an 'O'
            elif num_vertices > 8:
                print("O detected")

    
     

def getGameState(cells):
    '''
    cells
    '''
    pass

def main():
    #create a 1d array to hold the gamestate
    gamestate = np.array([None,None,None,
                        None, None,None,
                        None,None,None])

    # Read in video feed 
    warped_grid = processBoard()
    # warped_grid = cv2.imread('../imgs/cell_1.png')
    cv2.imshow('warpedgrid', warped_grid)
    cells = getGridCells(warped_grid)
    for cell in cells:
        cv2.imshow('cell', cell)
        cv2.waitKey(0)
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