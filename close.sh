source ~ee106a/sawyer_setup.bash
source devel/setup.bash
catkin_make
rosrun vision gripper_close.py