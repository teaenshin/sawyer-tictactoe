import rospy
import BOARD_DATA 

class VisionNode:
    def __init__(self):
        rospy.init_node('vision_node')
        self.publisher = rospy.Publisher('board_data_topic', BOARD_DATA, queue_size=10)
        self.rate = rospy.Rate(1)  # 1 Hz
        #TODO init the center of the grid and the dimensions for cropping

    def publish_board_data(self):
        while not rospy.is_shutdown():
            board_data = BOARD_DATA()
            board_data.data = self.do_vision()
            rospy.loginfo("Publishing board data: %s", board_data.data)
            self.publisher.publish(board_data)
            self.rate.sleep()

    def do_vision(self):
        #TODO implement this function using vision.py
        return [""]*9

if __name__ == '__main__':
    node = VisionNode()
    node.publish_board_data()
