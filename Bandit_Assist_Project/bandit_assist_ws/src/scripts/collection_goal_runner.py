#!/usr/bin/env python3
import math
import actionlib
import rospy
import tf.transformations
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import LaserScan
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

        # ------------------------------------------------------------
        # OBJECT DETECTION (inline, placeholder)
        # -----------------------------------------------------------
        # Subscribes directly to the laser scan and tracks whether
        # something is currently closer than safe_stop_distance in
        # front of the robot. If an object is detected close while
        # navigating to the collection point, the move_base goal is
        # cancelled and zero velocity is published until the path is
        # clear, then the same goal is resent.
        #
        # TEAMMATE TODO: replace the body of scan_callback() with real
        # object detection (e.g. camera/YOLO based). Keep updating
        # self.object_close the same way (True/False) so the rest of
        # run() does not need to change.
        # ------------------------------------------------------------
        self.scan_topic = rospy.get_param("~scan_topic", "/scan_front_raw")
        self.safe_stop_distance = rospy.get_param("~safe_stop_distance", 0.6)
        self.front_angle_window = rospy.get_param("~front_angle_window", 0.5)
        self.object_close = False

        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)

        rospy.on_shutdown(self.stop_robot)

    def scan_callback(self, scan):
        """
        PLACEHOLDER object detection logic.

        Checks the minimum laser range within a forward-facing angle
        window. Sets self.object_close to True if something is closer
        than self.safe_stop_distance.

        TEAMMATE TODO: replace this body with real object detection
        (camera/YOLO/etc). Just keep setting self.object_close to
        True/False the same way.
        """
        min_dist = float("inf")

        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) <= self.front_angle_window:
                if r < min_dist:
                    min_dist = r

        was_close = self.object_close
        self.object_close = min_dist < self.safe_stop_distance

        if self.object_close and not was_close:
            rospy.logwarn("Object detected close to robot (dist=%.2fm)", min_dist)
            self.state_pub.publish("object detected - paused")
        elif was_close and not self.object_close:
            rospy.loginfo("Path clear - object no longer close")
            self.state_pub.publish("path clear - resuming")

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

        goal = self.build_goal()

        # OBJECT DETECTION: don't start moving if something is already
        # blocking the path right in front of the robot.
        if self.object_close:
            rospy.logwarn("Object detected near robot - waiting before sending goal.")
            self.state_pub.publish("object detected - waiting before start")
            rate = rospy.Rate(2)
            while self.object_close and not rospy.is_shutdown():
                rate.sleep()

        rospy.loginfo(
            "Sending collection point goal: x=%.3f, y=%.3f, yaw=%.3f",
            self.goal_x,
            self.goal_y,
            self.goal_yaw
        )
        self.move_base_client.send_goal(goal)

        # ------------------------------------------------------------
        # OBJECT DETECTION
        # ------------------------------------------------------------
        deadline = rospy.Time.now() + rospy.Duration(self.timeout)
        rate = rospy.Rate(5)
        finished = False

        while not rospy.is_shutdown():
            finished = self.move_base_client.wait_for_result(rospy.Duration(0.2))

            if finished:
                break

            if rospy.Time.now() > deadline:
                break

            if self.object_close:
                rospy.logwarn("Object detected during navigation - cancelling goal.")
                self.move_base_client.cancel_goal()
                self.publish_zero_velocity()

                while self.object_close and not rospy.is_shutdown():
                    rate.sleep()

                rospy.loginfo("Path clear - resending collection point goal.")
                self.move_base_client.send_goal(goal)
                # extend deadline a little to account for the pause
                deadline = rospy.Time.now() + rospy.Duration(self.timeout)

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
