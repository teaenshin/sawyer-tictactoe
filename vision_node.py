import rospy
import BOARD_DATA 
from vision.vision import *

class VisionNode:
    def __init__(self):
        rospy.init_node('vision_node')
        self.publisher = rospy.Publisher('board_data_topic', BOARD_DATA, queue_size=10)
        self.rate = rospy.Rate(1)  # 1 Hz
        self.whiteboard = None
        while True:
            color_image = get_color_image()
            self.whiteboard = get_whiteboard(color_image)
            cropped_image = crop_image(color_image, self.whiteboard)
            cv2.imshow("Cropped", cropped_image)
            user_input = input("Press Y and Enter if Good, any other key if not")
            if user_input == "Y":
                break
        

    def publish_board_data(self):
        while not rospy.is_shutdown():
            board_data = BOARD_DATA()
            board_data.data = self.get_board()
            rospy.loginfo("Publishing board data: %s", board_data.data)
            self.publisher.publish(board_data)
            self.rate.sleep()


    def get_board(self):
        color_image = get_color_image()
        cropped_image = crop_image(color_image, self.whiteboard)
        return getBoard(cropped_image)
    

if __name__ == '__main__':
    node = VisionNode()
    node.publish_board_data()
