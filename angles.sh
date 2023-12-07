source ~ee106a/sawyer_setup.bash
source devel/setup.bash
catkin_make
rostopic echo robot/joint_states