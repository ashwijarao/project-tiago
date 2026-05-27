#!/usr/bin/env python3

import actionlib
import rospy
import tf.transformations

from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import String


class CollectionGoalRunner:
    def __init__(self):
        rospy.init_node("collection_goal_runner")

        self.target_frame = rospy.get_param("~target_frame", "map")
        self.goal_x = float(rospy.get_param("~goal_x", 0.0))
        self.goal_y = float(rospy.get_param("~goal_y", 0.0))
        self.goal_yaw = float(rospy.get_param("~goal_yaw", 0.0))
        self.timeout = float(rospy.get_param("~timeout_seconds", 120.0))

        self.state_pub = rospy.Publisher(
            "/collection_point/task_state",
            String,
            queue_size=5,
            latch=True
        )

        self.stop_publishers = [
            rospy.Publisher("/nav_vel", Twist, queue_size=1),
            rospy.Publisher("/mobile_base_controller/cmd_vel", Twist, queue_size=1)
        ]

        self.move_base_client = actionlib.SimpleActionClient(
            "/move_base",
            MoveBaseAction
        )

        rospy.on_shutdown(self.stop_robot)

    def build_goal(self):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.target_frame
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = self.goal_x
        goal.target_pose.pose.position.y = self.goal_y
        goal.target_pose.pose.position.z = 0.0

        q = tf.transformations.quaternion_from_euler(
            0.0,
            0.0,
            self.goal_yaw
        )

        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        return goal

    def publish_zero_velocity(self):
        zero_cmd = Twist()

        for _ in range(6):
            for publisher in self.stop_publishers:
                publisher.publish(zero_cmd)
            rospy.sleep(0.1)

    def stop_robot(self):
        self.move_base_client.cancel_all_goals()
        self.publish_zero_velocity()

    def run(self):
        rospy.loginfo("Collection point navigation task started.")
        self.state_pub.publish("collection point navigation started")

        rospy.loginfo("Waiting for move_base...")
        if not self.move_base_client.wait_for_server(rospy.Duration(30.0)):
            rospy.logerr("move_base is not available.")
            self.state_pub.publish("move_base unavailable")
            return

        rospy.loginfo(
            "Sending collection point goal: x=%.3f, y=%.3f, yaw=%.3f",
            self.goal_x,
            self.goal_y,
            self.goal_yaw
        )

        self.move_base_client.send_goal(self.build_goal())
        finished = self.move_base_client.wait_for_result(
            rospy.Duration(self.timeout)
        )

        if not finished:
            rospy.logwarn("Collection point navigation timed out.")
            self.move_base_client.cancel_goal()
            self.publish_zero_velocity()
            self.state_pub.publish("collection point timeout")
            return

        state = self.move_base_client.get_state()

        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo("Reached collection point")
            self.publish_zero_velocity()
            self.state_pub.publish("reached collection point")
        else:
            rospy.logwarn("Navigation failed. move_base state: %s", state)
            self.publish_zero_velocity()
            self.state_pub.publish("collection point navigation failed")


if __name__ == "__main__":
    try:
        runner = CollectionGoalRunner()
        runner.run()
    except rospy.ROSInterruptException:
        pass
