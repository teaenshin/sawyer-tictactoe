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
    
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def processBoard(debug=True):

    img = cv2.imread('imgs/cam4.jpg')
    # resized_frame = cv2.resize(img, (300, 300))  # TODO # Resize the image for consistency
    resized_frame = img
    ### Crop 
    whiteboard = get_whiteboard(resized_frame)
    cropped_image = crop_image(resized_frame, whiteboard)

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
    cv2.imshow('warped_image', warped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    return warped_image
    # Display the resulting frame

def getGridCells(warped_grid):
    # TODO: identify the 9 cells (just divide by 3 for now)
    
    # TODO: process each of the 9 cells and detect if theres is an X (bonus: detect if there is a O )
    cell_height = warped_grid.shape[0] // 3
    cell_width = warped_grid.shape[1] // 3

    for i in range(3):
        # Loop through each column
        for j in range(3):
            # Define the region for the current cell
            left = j * cell_width
            top = i * cell_height
            right = left + cell_width
            bottom = top + cell_height

            # Crop the image to the current cell
            cell_image = warped_grid.crop((left, top, right, bottom))

            # Add the cropped image to the list
            warped_grid.append(cell_image)

    # Create a new image by pasting the grid images
    # new_image = Image.new('RGB', (width, height))

    # for i in range(3):
    #     for j in range(3):
    #         new_image.paste(grid_images[i * 3 + j], (j * cell_width, i * cell_height))
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return




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

def main():
    #create a 1d array to hold the gamestate
    gamestate = np.array([None,None,None,
                        None, None,None,
                        None,None,None])

    # Read in video feed 
    warped_grid = processBoard()
    cells = getGridCells(warped_grid)
    print('cells', cells)
    # processCells(warped_board)
    # getCamera()
    
    '''
    if board is done drawing:
        while not isGameOver(state): 
            
    '''

if __name__ == "__main__":
    main()