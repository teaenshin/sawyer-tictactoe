import cv2
import numpy as np

# def find_corners(img):
#     """Finds harris corners
#     """
#     corners = cv2.cornerHarris(img, 5, 3, 0.1)
#     corners = cv2.dilate(corners, None)
#     corners = cv2.threshold(corners, 0.01 * corners.max(), 255, 0)[1]
#     corners = corners.astype(np.uint8)
#     _, labels, stats, centroids = cv2.connectedComponentsWithStats(
#         corners, connectivity=4)
#     # For some reason, stats yielded better results for
#     # corner detection than centroids. This might have
#     # something to do with sub-pixel accuracy. 
#     # Check issue #10130 on opencv
#     return stats 


# def contoured_bbox(img):
#     """Returns bbox of contoured image"""
#     contours, hierarchy = cv2.findContours(img, 1, 2)
#     # Largest object is whole image,
#     # second largest object is the ROI
#     sorted_cntr = sorted(contours, key=lambda cntr: cv2.contourArea(cntr))
#     return cv2.boundingRect(sorted_cntr[-2])


# def preprocess_input(img):
#     """Preprocess image to match model's input shape for shape detection"""
#     img = cv2.resize(img, (32, 32))
#     # Expand for channel_last and batch size, respectively
#     img = np.expand_dims(img, axis=-1)
#     img = np.expand_dims(img, axis=0)
#     return img.astype(np.float32) / 255

# def order_points(pts):
#     """
#     Helper function for four_point_transform.
#     Check pyimagesearch blog for an explanation on the matter
#     """
#     # Order: top-left, top-right, bottom-right and top-left
#     rect = np.zeros((4, 2), dtype=np.float32)
#     # top-left will have smallest sum, while bottom-right
#     # will have the largest one
#     _sum = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(_sum)]
#     rect[2] = pts[np.argmax(_sum)]
#     # top-right will have smallest difference, while
#     # bottom-left will have the largest one
#     _diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(_diff)]
#     rect[3] = pts[np.argmax(_diff)]
#     return rect

# def four_point_transform(img, pts):
#     """Returns 'bird view' of image"""
#     rect = order_points(pts)
#     tl, tr, br, bl = rect
#     # width of new image will be the max difference between
#     # bottom-right - bottom-left or top-right - top-left
#     widthA = np.linalg.norm(br - bl)
#     widthB = np.linalg.norm(tr - tl)
#     width = int(round(max(widthA, widthB)))
#     # Same goes for height
#     heightA = np.linalg.norm(tr - br)
#     heightB = np.linalg.norm(tl - bl)
#     height = int(round(max(heightA, heightB)))
#     # construct destination for 'birds eye view'
#     dst = np.array([
#         [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
#         dtype=np.float32)
#     # compute perspective transform and apply it
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(img, M, (width, height))
#     return warped

# def find_sheet_paper(frame, thresh, add_margin=True):
#     """Detect the coords of the sheet of paper the game will be played on"""
#     stats = find_corners(thresh)
#     # First point is center of coordinate system, so ignore it
#     # We only want sheet of paper's corners
#     corners = stats[1:, :2]
#     # print('corners shape', corners)
#     if corners.shape[0] < 4:
#         return None, None
#     corners = order_points(corners) # (n, 2)
#     # Get bird view of sheet of paper
#     paper = four_point_transform(frame, corners)
#     if add_margin:
#         paper = paper[10:-10, 10:-10]
#     return paper, corners


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

def processBoard(debug=True):

    print('processboard')
    # 1. Preproccess img
    img = cv2.imread('imgs/cam3.jpg')
    # cv2.imshow('original', img)
    resized_frame = cv2.resize(img, (300, 300))  # Resize the image for consistency
    # cv2.imshow('resized', resized_frame)
    # Convert to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    # bi = cv2.bilateralFilter(gray, 5, 75, 75)
    # cv2.imshow('bi',bi)
    _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh # invert so grid becomes white
    cv2.imshow('threshold', thresh)


    # 2. Get largest contour, which should be the grid
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)
    # Create a mask image for visualization
    mask = resized_frame.copy()
    # cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
    # cv2.imshow('largest contour', mask)
    epsilon = 0.04 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 3. Get 4 corners of the grid
    corners = approx_polygon.reshape(-1, 2)
    print(corners)
    # Draw the polygon on the original image
    image_with_polygon = resized_frame.copy()
    for corner in corners:
        cv2.circle(image_with_polygon, corner, 5, (0, 0, 255), -1)  # -1 fills the circle with the specified color
    # cv2.drawContours(image_with_polygon, [approx_polygon], -1, (0, 255, 0), 2)
    cv2.imshow('largest contour with corners ', image_with_polygon)

    
    # 4. warp grid into square
    # Define the four corners of the target square
    target_size = 300
    target_corners = np.array([[0, 0], [target_size - 1, 0], [target_size - 1, target_size - 1], [0, target_size - 1]], dtype=np.float32)
    corners = np.float32(corners) # convert to np.float32 for cv2.warpPerspective
    print('cornesrs float', corners)
    assert corners.shape == (4, 2)
    # Calculate the perspective transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(corners, target_corners)
    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(thresh, transformation_matrix, (target_size, target_size))
    cv2.imshow('warped_image', warped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    
    # Display the resulting frame

     
def preprocess(img):
    resized_frame = cv2.resize(img, (64, 64))  # Resize the image for consistency
    # Convert to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



def getGridCells(center, gridLen):
    '''
    center: (row, col)
    tictactoe trid is gridLen x gridLen

    returns (9,2) dim array with
    [[[top left], [top right], [bottom left], [bottom right]],
        
    ]

    '''

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
    processBoard()
    # getCamera()
    
    '''
    if board is done drawing:
        while not isGameOver(state): 
            
    '''

if __name__ == "__main__":
    main()