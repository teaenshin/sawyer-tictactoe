#!/usr/bin/env python

import rospy
from vision.msg import BoardData
from vision_utils import *

class VisionNode:
    ''' Publishes gamestate '''
    def __init__(self):
        rospy.init_node('vision_node')
        self.publisher = rospy.Publisher('board_data_topic', BoardData, queue_size=1)
        self.rate = rospy.Rate(1)  # 1 Hz
        self.whiteboard = None # whiteboard contour
        self.vid = cv2.VideoCapture(1) 
        if not self.vid.isOpened():
            print("Error could not open camera")
            exit()
        self.vid.set(cv2.CAP_PROP_FPS, 30) # set frame rate


    def setup_vision(self, debug=False):
        while True:
            cv2.destroyAllWindows()

            color_image = self.get_color_image()
            # color_image = cv2.imread('/home/cc/ee106a/fa23/class/ee106a-aem/sawyer-tictactoe/src/vision/src/imgs/cam3.jpg')
            self.whiteboard = get_whiteboard(color_image) # largest contour, used to identify whether board is obstructed
            
            if debug:
                copy = color_image.copy()
                cv2.drawContours(copy, self.whiteboard, -1 ,(255,255,255), 1)
                cv2.imshow('whiteboard contour', copy)
                cv2.waitKey(0)

            # manually check whether cropped and warped grid is correct
            cropped_image = crop_image(color_image, self.whiteboard)
            cv2.imshow("Cropped", cropped_image)
            print("Press y if it cropped whiteboard looks good. Press enter to refresh camera feed.")
            key = cv2.waitKey(0)
            if key != ord('y'):
                continue 
            warped_board = getBoard(cropped_image)
            cv2.imshow("Warped", warped_board)
            print("Press y if it warped whiteboard looks good. Press enter to refresh camera feed.")
            key = cv2.waitKey(0)
            if key == ord('y'):
                break 

    def get_color_image(self):
        ''' gets a color frame from camera feed '''
        ret, frame = self.vid.read() 
        if not ret:
            print("Error could not read frame")
        return frame

        
    def publish_board_data(self):
        while not rospy.is_shutdown():
            # cur_board = self.get_board()
            
            board_data = BoardData()
            cur_gamestate = self.get_gamestate()
            if cur_gamestate is None: # if full whiteboard not detected, don't publish
                print("Full whiteboard not detected")
                self.rate.sleep()
                continue
            board_data.data = cur_gamestate
            self.publisher.publish(board_data)
            rospy.loginfo("Publishing board data: %s", board_data.data)
            self.rate.sleep()

    
    def get_gamestate(self):
        color_image = self.get_color_image()

        # don't publish when board not detected/
        cur_whiteboard = get_whiteboard(color_image)
        print('og whiteboard contour area vs cur', cv2.contourArea(self.whiteboard), cv2.contourArea(cur_whiteboard))

        if cv2.contourArea(cur_whiteboard) / cv2.contourArea(self.whiteboard) < 0.7:
            return None
        
        cropped_image = crop_image(color_image, self.whiteboard)
        warped_board = getBoard(cropped_image)

        # cv2.imshow('warped board', warped_board)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cells = getGridCells(warped_board)
        gamestate = get_state(cells)
        print('gamestate', gamestate)
        assert gamestate is not None
        return gamestate



    

if __name__ == '__main__':
    node = VisionNode()
    node.setup_vision()

    try:
        node.publish_board_data()
    except rospy.ROSInterruptException: pass
