FROM ros:lunar

RUN apt-get update && apt-get install -y build-essential python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /root

# catkin_make
# source devel/setup.bash
# export ROS_MASTER_URI=http://`hostname`:11311
# roslaunch quad_controller_rl rl_controller.launch