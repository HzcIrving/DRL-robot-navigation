# 用于与ROS系统进行通信和交互的Python库
import rospy 
# 用于在Python中创建和控制新进程的库。
import subprocess
from os import path 

# visualization_msgs: ROS中用于可视化的消息类型。
# - Marker: 用于在ROS中发布可视化标记的消息类型。 
# - MarkerArray: 用于在ROS中发布多个可视化标记的消息类型。
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray 

from numpy import inf
import numpy as np
import random
import math

# gazebo_msgs: ROS中与Gazebo仿真器通信的消息类型 
# - ModelState: 用于设置模型状态的消息类型  
from gazebo_msgs.msg import ModelState 

# 用于进行四元数计算的库 
from squaternion import Quaternion 

# Twist: 用于表示机器人运动命令的消息类型
from geometry_msgs.msg import Twist

# LaserScan：用于激光雷达扫描数据的消息类型。
# PointCloud2：用于点云数据的消息类型。
from sensor_msgs.msg import LaserScan, PointCloud2 

# pc2：用于处理点云数据的模块。
import sensor_msgs.point_cloud2 as pc2

# Odometry：用于机器人里程计信息的消息类型。
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 确保初始目标点不与目标障碍物产生交合
# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goalOK = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goalOK = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goalOK = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goalOK = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goalOK = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goalOK = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goalOK = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goalOK = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goalOK = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goalOK = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goalOK = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goalOK = False
    return goalOK


# Function to put the laser data in bins
def binning(lower_bound, data, quantity):
    width = round(len(data) / quantity)
    quantity -= 1
    bins = []
    for low in range(lower_bound, lower_bound + quantity * width + 1, width):
        bins.append(min(data[low:low + width]))
    return np.array([bins])


class GazeboEnv:
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, launchfile, height, width, nchannels):
        """ 
        初始化参数需要提供launch文件
        位置: TD3/asserts 
        """ 
        # 初始里程计信息
        self.odomX = 0
        self.odomY = 0

        self.goalX = 1
        self.goalY = 0.0

        # 边界 
        self.upper = 5.0
        self.lower = -5.0 
        
        # Velodyne数据 20Dim * 10m 
        # 最大视距为10m 
        self.velodyne_data = np.ones(20) * 10


        # ModelState() 
        # Robot r1的状态（四元数）、位置、Goalreaching distance
        self.set_self_state = ModelState()        
        self.set_self_state.model_name = 'r1'
        self.set_self_state.pose.position.x = 0.
        self.set_self_state.pose.position.y = 0.
        self.set_self_state.pose.position.z = 0.
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        
        # Create"gaps" for bining of laser data 
        # 角度范围  
        # 最左侧 [-pi/2+0.03, -pi/2+pi/laser_dim] 
        #               [-pi/2+pi/laser_dim, -pi/2+pi/laser_dim+pi_laser_dim] 
        #               .....
        #               [pi/2-pi/laser_dim, pi/2+0.03]
        self.gaps = [[-1.6, -1.57 + 3.14 / 20]] # 用于存储每个激光雷达扫描数据之间的角度间隔
        for m in range(19):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + 3.14 / 20])
        self.gaps[-1][-1] += 0.03

        # initialized 
        # launching simulator 
        port = '11311'
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node('gym', anonymous=True) 
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        # 使用Python中的subprocess库启动了一个roscore进程，并指定了端口号为11311。
        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        self.gzclient_pid = 0

        # Set up the ROS publishers and subscribers 
        # 发布机器人速度控制指令到'r1/cmd_vel' 
        self.vel_pub = rospy.Publisher('/r1/cmd_vel', Twist, queue_size=1) 
        
        """
        # 在Gazebo中，模型的状态包括其位置、姿态、线速度、角速度等信息。
        # 通过发布一个 ModelState 消息，可以更改模型的状态，这在仿真环境中非常有用。 
        下面这个语法，将一个包含有关Gazebo中模型状态的消息 ModelState 发布到主题 gazebo/set_model_state 上。这个 Publisher 的队列大小是10，这意味着如果发布者发布的消息数量超过10，则最旧的消息将被删除以保持队列的大小不超过10。
        """
        self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10) 
        
        # 物理引擎的暂停/启动 
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty) 
        
        # 物理引擎重启 
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        
        # 可视化不同的标记数据  
        topic = 'vis_mark_array'
        self.publisher = rospy.Publisher(topic, MarkerArray, queue_size=3)
        topic2 = 'vis_mark_array2'
        self.publisher2 = rospy.Publisher(topic2, MarkerArray, queue_size=1)
        topic3 = 'vis_mark_array3'
        self.publisher3 = rospy.Publisher(topic3, MarkerArray, queue_size=1)
        topic4 = 'vis_mark_array4'
        self.publisher4 = rospy.Publisher(topic4, MarkerArray, queue_size=1)
        
        # 订阅机器人的Velodyne激光雷达数据 
        # 数据会在接收到后传入 self.velodyne_callback回调函数
        self.velodyne = rospy.Subscriber('/velodyne_points', PointCloud2, self.velodyne_callback, queue_size=1)

    # POI检测 -> 20个点 
    # ->【 Return 20Dim self.velodyne_data 激光雷达数据】
    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(20) * 10 # 10m范围初始化 
        
        """
        这段代码是一个用于处理Velodyne激光雷达数据的回调函数。下面是对代码的解释：

        获取激光雷达数据：首先，通过pc2.read_points()函数从激光雷达消息中读取数据。这些数据以点云形式存在，包含每个点的(x, y, z)坐标信息。

        初始化激光雷达数据数组：创建一个长度为20的一维数组self.velodyne_data，并将每个元素初始化为10。这表示激光雷达的有效范围为10米。

        处理每个点的数据：对于每个点，进行以下操作：

        - 检查点的高度：通过判断data[i][2]是否大于-0.2来确定点是否有效。如果高度小于-0.2，表示该点无效，将跳过后续处理。
        - 计算角度和距离：通过点的(x, y)坐标计算与参考向量的夹角beta，以及点到原点的距离dist。
        - 将数据分配到对应的槽位：根据角度beta的范围，将距离dist分配给self.velodyne_data数组的相应槽位。通过遍历self.gaps列表，找到与当前点的角度beta对应的范围，然后更新对应槽位的距离值。这样，self.velodyne_data数组中的每个槽位代表了一个特定角度范围内的最小距离值。
        
        通过这段代码，可以将激光雷达的数据转化为一个长度为20的一维数组，每个元素表示对应角度范围内的最小距离值，用于描述环境的障碍物情况。
        """
        for i in range(len(data)):
            # 有效高度应该大于-0.2  
            if data[i][2] > -0.2: 
                
                # dot = x  
                # x轴方向投影 
                dot = data[i][0] * 1 + data[i][1] * 0
                
                # 计算角度和距离  
                # mag1 = sqrt(x^2 + y^2) 
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                # mag2 = sqrt(1) 
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                
                # beta = arccos(x/sqrt(x^2+y^2)) * sign (当在右边，data[i][1]) 
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])  # * -1
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                # 看角度落在哪个角度范围 
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break     
                    
    # 碰撞检测  
    # Detect a collision from laser data
    def calculate_observation(self, data):
        min_range = 0.3
        min_laser = 2
        done = False
        col = False

        for i, item in enumerate(data.ranges):
            if min_laser > data.ranges[i]:
                min_laser = data.ranges[i]
            if (min_range > data.ranges[i] > 0):
                done = True
                col = True
        return done, col, min_laser

    # Perform an action and read a new state
    def step(self, act):
        target = False
        
        # 依据强化学习决策来发布动作信息  
        # 速度、角度 
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = act[0]
        vel_cmd.angular.z = act[1]
        self.vel_pub.publish(vel_cmd)
        
        rospy.wait_for_service('/gazebo/unpause_physics')
        try: 
            # 暂停  该方法会向 /gazebo/unpause_physics service 发送请求，请求gazebo物理引擎暂停。
            self.unpause() 
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/r1/front_laser/scan', LaserScan, timeout=0.5)
            except:
                print("Attention !!! No radar scan data received ... ")

        """激光雷达数据更新"""
        laser_state = np.array(data.ranges[:])
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state] # 复制模式 或者(deepcopy也可)

        # 碰撞检测  
        done, col, min_laser = self.calculate_observation(data)

        dataOdom = None 
        while dataOdom is None:
            try:
                dataOdom = rospy.wait_for_message('/r1/odom', Odometry, timeout=0.5)
            except:
                print("Attention !!! No odom data received ... ")
            
        time.sleep(0.1) 
        rospy.wait_for_service('/gazebo/pause_physics')
        
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # Calculate robot heading from odometry data 
        # 航向以四元数形式返回 
        self.odomX = dataOdom.pose.pose.position.x
        self.odomY = dataOdom.pose.pose.position.y
        quaternion = Quaternion(
            dataOdom.pose.pose.orientation.w,
            dataOdom.pose.pose.orientation.x,
            dataOdom.pose.pose.orientation.y,
            dataOdom.pose.pose.orientation.z) 
        
        # 四元数 转化为 欧拉角
        euler = quaternion.to_euler(degrees=False) 
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        # Calculate the angle distance between the robots heading and heading toward the goal
        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY
        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skewY < 0:
            if skewX < 0:
                beta = -beta
            else:
                beta = 0 - beta
        beta2 = (beta - angle)
        if beta2 > np.pi:
            beta2 = np.pi - beta2
            beta2 = -np.pi - beta2
        if beta2 < -np.pi:
            beta2 = -np.pi - beta2
            beta2 = np.pi - beta2

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = act[0]
        vel_cmd.angular.z = act[1]
        self.vel_pub.publish(vel_cmd)

        # Publish visual data in Rviz  
        # 该段话讲述了一个名为publish_markers的函数，该函数的作用是发布在Rviz中显示的视觉信息，其中包括行动值和目标位置 
        # 创建一个圆柱体Marker，并将其发布到ROS话题
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "base_link" # 坐标系
        marker.type = marker.CYLINDER # 类型 
        marker.action = marker.ADD # Marker行为，ADD添加到场景中 
        marker.scale.x = 0.1 # 
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0 
        marker.color.r = 0.0
        marker.color.g = 1.0 # 绿色marker 
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0  
        marker.pose.position.x = self.goalX 
        marker.pose.position.y = self.goalY
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        # # 创建一个立方体Marker，并将其发布到ROS话题
        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "base_link"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(act[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        # # 创建一个立方体Marker，并将其发布到ROS话题
        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "base_link"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(act[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

        # 创建一个立方体Marker，并将其发布到ROS话题
        markerArray4 = MarkerArray()
        marker4 = Marker()
        marker4.header.frame_id = "base_link"
        marker4.type = marker.CUBE
        marker4.action = marker.ADD
        marker4.scale.x = 0.1  # abs(act2)
        marker4.scale.y = 0.1
        marker4.scale.z = 0.01
        marker4.color.a = 1.0
        marker4.color.r = 1.0
        marker4.color.g = 0.0
        marker4.color.b = 0.0
        marker4.pose.orientation.w = 1.0
        marker4.pose.position.x = 5
        marker4.pose.position.y = 0.4
        marker4.pose.position.z = 0
        markerArray4.markers.append(marker4)
        self.publisher4.publish(markerArray4)

        '''Bunch of different ways to generate the reward'''

        # reward = act[0]*0.7-abs(act[1])
        # r1 = 1 - 2 * math.sqrt(abs(beta2 / np.pi))
        # r2 = self.distOld - Dist
        r3 = lambda x: 1 - x if x < 1 else 0.0
        # rl = 0
        # for r in range(len(laser_state[0])):
        #    rl += r3(laser_state[0][r])
        # reward = 0.8 * r1 + 30 * r2 + act[0]/2 - abs(act[1])/2 - r3(min(laser_state[0]))/2
        reward = act[0] / 2 - abs(act[1]) / 2 - r3(min(laser_state[0])) / 2
        # reward = 30 * r2 + act[0] / 2 - abs(act[1]) / 2  # - r3(min(laser_state[0]))/2
        # reward = 0.8 * r1 + 30 * r2

        self.distOld = Dist

        # Detect if the goal has been reached and give a large positive reward
        if Dist < 0.3:
            target = True
            done = True
            self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
            reward = 80

        # Detect if ta collision has happened and give a large negative reward
        if col:
            reward = -100

        toGoal = [Dist, beta2, act[0], act[1]]
        state = np.append(laser_state, toGoal)
        return state, reward, done, target

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_world') 
        
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0., 0., angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        chk = False
        while not chk:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            chk = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odomX = object_state.pose.position.x
        self.odomY = object_state.pose.position.y

        self.change_goal()
        self.random_box()
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        data = None
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
            
        while data is None:
            try:
                data = rospy.wait_for_message('/r1/front_laser/scan', LaserScan, timeout=0.5)
            except:
                pass
            
        laser_state = np.array(data.ranges[:])
        laser_state[laser_state == inf] = 10
        laser_state = binning(0, laser_state, 20)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY

        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skewY < 0:
            if skewX < 0:
                beta = -beta
            else:
                beta = 0 - beta
        beta2 = (beta - angle)

        if beta2 > np.pi:
            beta2 = np.pi - beta2
            beta2 = -np.pi - beta2
        if beta2 < -np.pi:
            beta2 = -np.pi - beta2
            beta2 = np.pi - beta2

        toGoal = [Dist, beta2, 0.0, 0.0]
        state = np.append(laser_state, toGoal)
        return state

    # Place a new goal and check if its lov\cation is not on one of the obstacles
    def change_goal(self):
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        gOK = False

        while not gOK:
            self.goalX = self.odomX + random.uniform(self.upper, self.lower)
            self.goalY = self.odomY + random.uniform(self.upper, self.lower)
            gOK = check_pos(self.goalX, self.goalY)

    # Randomly change the location of the boxes in the environment on each reset to randomize the training environment
    def random_box(self):
        for i in range(4):
            name = 'cardboard_box_' + str(i) 
            
            x = 0
            y = 0
            chk = False
            while not chk:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                chk = check_pos(x, y)
                d1 = math.sqrt((x - self.odomX) ** 2 + (y - self.odomY) ** 2)
                d2 = math.sqrt((x - self.goalX) ** 2 + (y - self.goalY) ** 2)
                if d1 < 1.5 or d2 < 1.5:
                    chk = False
            box_state = ModelState()
            box_state.model_name = name 
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)
