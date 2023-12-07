source ~ee106a/sawyer_setup.bash
source devel/setup.bash
catkin_make
roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true