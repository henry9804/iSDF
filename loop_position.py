import rospy
import moveit_commander
import time
from std_msgs.msg import Bool
import cv2


rospy.init_node('periodic_motion_controller')
move_group = moveit_commander.MoveGroupCommander('panda_manipulator')  # 로봇의 MoveGroup 이름

move_group.set_max_velocity_scaling_factor(0.1)  
move_group.set_max_acceleration_scaling_factor(0.1)  

move_group.set_goal_tolerance(0.03)  
period = 1.0

state_pub = rospy.Publisher('/robot/goal_reached',Bool,queue_size=10)
start_pub = rospy.Publisher('/isdf/start', Bool, queue_size =1)

joint_goals = [
    [0, -1.0, 0.0, -2.5, 0.0, 2.5, 1.0],
    [-0.3, -1.0, 0.0, -2.5, 0.0, 2.5, 1.0],
    [-0.6, -1.0, 0.0, -2.5, 0.0, 2.5, 1.0],
    [-0.9, -1.0, 0.0, -2.5, 0.0, 2.5, 1.0], 
    [-1.2, -1.0, 0.0, -2.5, 0.0, 2.5, 1.0],  
    [-1.5, -1.0, 0.0, -2.5, 0.0, 2.5, 1.0],
    [0, -1.0, 0.0, -2.5, 0.0, 2.5, 1.0],
]


tolerance = 0.05

while not rospy.is_shutdown():
        
    input("Press any key to start the robot motion...")
    for joint_goal in joint_goals:
        move_group.set_joint_value_target(joint_goal)

        # 이동 명령 실행 (비동기식)
        move_group.go(wait=False)

        rate = rospy.Rate(5)  

        while not rospy.is_shutdown():
            goal_reached = False
            current_joints = move_group.get_current_joint_values()

            goal_reached = all(
                abs(current - goal) < tolerance
                for current, goal in zip(current_joints, joint_goal)
            )

            state_pub.publish(goal_reached)


            if goal_reached:
                rospy.sleep(period)
                break

            rate.sleep()

        move_group.stop()
    new_start = True
    # input("Press any key to send the signal to view sdf...")  
    for _ in range(10):  # 5번 반복 발행
        print("start command publish")
        start_pub.publish(new_start)
        rospy.sleep(0.01)  # 0.1초 대기 (간격 조정 가능)

    
 
