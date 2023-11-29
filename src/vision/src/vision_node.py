#!/usr/bin/env python

import rospy
from vision.msg import BoardData
from vision_utils import *

class VisionNode:
    def __init__(self):
        rospy.init_node('vision_node')
        self.publisher = rospy.Publisher('board_data_topic', BoardData, queue_size=1)
        self.rate = rospy.Rate(1)  # 1 Hz
        self.whiteboard = None
        self.vid = cv2.VideoCapture(0) 
        if not self.vid.isOpened():
            print("Error could not open camera")
            exit()
        self.vid.set(cv2.CAP_PROP_FPS, 30) # set frame rate


    def setup_vision(self):
        while True:
            color_image = self.get_color_image()
            self.whiteboard = get_whiteboard(color_image)
            cropped_image = crop_image(color_image, self.whiteboard)
            cv2.imshow("Cropped", cropped_image)
            print("Press y if it looks good")
            if cv2.waitKey(0) & 0xFF == ord('y'):
                break
        cv2.destroyAllWindows()

    def get_color_image(self):
        ret, frame = self.vid.read() 
        if not ret:
            print("Error could not read frame")
        return frame

        
    def publish_board_data(self):
        while not rospy.is_shutdown():
            board_data = BoardData()
            board_data.data = self.get_board()
            rospy.loginfo("Publishing board data: %s", board_data.data)
            self.publisher.publish(board_data)
            self.rate.sleep()


    def get_board(self):
        color_image = self.get_color_image()
        cropped_image = crop_image(color_image, self.whiteboard)
        return getBoard(cropped_image)
    

if __name__ == '__main__':
    print("hi")
    node = VisionNode()
    node.setup_vision()
    node.publish_board_data()
