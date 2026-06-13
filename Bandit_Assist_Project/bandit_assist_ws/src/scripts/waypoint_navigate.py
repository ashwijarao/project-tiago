#!/usr/bin/env python3

import math
import rospy
import actionlib
import tf

from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import String


# ---------------------------------------------------------------------------
# WAYPOINT DEFINITIONS
# Each waypoint uses map-frame coordinates: x, y, yaw
# ---------------------------------------------------------------------------

WAYPOINTS = {
    # Home position
    "home": (1.951, -0.994, 2.407),

    # Project collection points
    "collection_point_1": (-4.289, -2.195, -1.697),
    "collection_point_2": (-6.720, -0.814, 2.606),
    "collection_point_3": (-6.434, 1.506, -2.420),
    "collection_point_4": (-5.513, 5.990, 2.254),

    # New collection point 5
    "collection_point_5": (-6.222, -1.731, 0.410),
}


# ---------------------------------------------------------------------------
# Build MoveBaseGoal from x, y, yaw
# ---------------------------------------------------------------------------

def build_goal(x, y, yaw):
    goal = MoveBaseGoal()

    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.position.z = 0.0

    qx, qy, qz, qw = tf.transformations.quaternion_from_euler(0, 0, yaw)

    goal.target_pose.pose.orientation.x = qx
    goal.target_pose.pose.orientation.y = qy
    goal.target_pose.pose.orientation.z = qz
    goal.target_pose.pose.orientation.w = qw

    return goal


# ---------------------------------------------------------------------------
# Navigator node
# ---------------------------------------------------------------------------

class Navigator:
    def __init__(self):
        self.status_pub = rospy.Publisher(
            "/tiago_delivery/status",
            String,
            queue_size=10
        )

        self.explain_pub = rospy.Publisher(
            "/explain",
            String,
            queue_size=1
        )

        self.move_base = actionlib.SimpleActionClient(
            "move_base",
            MoveBaseAction
        )

        # ------------------------------------------------------------
        # OBJECT DETECTION 
        # -----------------------------------------------------------
        self.scan_topic = rospy.get_param("~scan_topic", "/scan_front_raw")
        self.safe_stop_distance = rospy.get_param("~safe_stop_distance", 0.6)
        self.front_angle_window = rospy.get_param("~front_angle_window", 0.5)
        self.object_close = False

        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)

        self.log("Waiting for move_base action server...")
        self.move_base.wait_for_server()
        self.log("move_base ready.")
        self.log("Navigator ready. Publish waypoint name to /tiago_delivery/waypoint_command")

        rospy.Subscriber(
            "/tiago_delivery/waypoint_command",
            String,
            self.on_command
        )

    def log(self, text):
        rospy.loginfo(text)
        self.status_pub.publish(String(data=text))

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
            self.log("Object detected close to robot (dist={:.2f}m)".format(min_dist))
        elif was_close and not self.object_close:
            self.log("Path clear - object no longer close")

    def on_command(self, msg):
        destination = msg.data.strip().lower()

        if destination not in WAYPOINTS:
            self.log("Unknown waypoint requested: {}".format(destination))
            return

        self.log("Command received: go to {}".format(destination))
        self.navigate_to(destination)

    def navigate_to(self, destination):
        x, y, yaw = WAYPOINTS[destination]
        goal = build_goal(x, y, yaw)

        # OBJECT DETECTION: don't start moving if something is already
        # blocking the path right in front of the robot.
        if self.object_close:
            self.log(
                "Object detected near robot - waiting before sending goal to {}".format(
                    destination
                )
            )
            rate = rospy.Rate(2)
            while self.object_close and not rospy.is_shutdown():
                rate.sleep()

        self.log(
            "Sending goal: x={:.3f}, y={:.3f}, yaw={:.3f}".format(
                x,
                y,
                yaw
            )
        )

        self.move_base.send_goal(goal)

        # ------------------------------------------------------------
        # OBJECT DETECTION: poll for result while watching for close
        # objects. If the path becomes blocked mid-navigation, cancel
        # the goal, wait until clear, then resend the same goal.
        # ------------------------------------------------------------
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            finished = self.move_base.wait_for_result(rospy.Duration(0.2))

            if finished:
                break

            if self.object_close:
                self.log(
                    "Object detected during navigation to {} - cancelling goal.".format(
                        destination
                    )
                )
                self.move_base.cancel_goal()

                while self.object_close and not rospy.is_shutdown():
                    rate.sleep()

                self.log("Path clear - resending goal to {}".format(destination))
                self.move_base.send_goal(goal)

        state = self.move_base.get_state()

        # actionlib GoalStatus.SUCCEEDED = 3
        if state == 3:
            self.log("Arrived at {}".format(destination))

            # Trigger explainer for home and collection points
            if destination == "home" or destination.startswith("collection_point"):
                self.explain_pub.publish(String(data=destination))
        else:
            self.log(
                "Failed to reach {}. move_base state: {}".format(
                    destination,
                    state
                )
            )


def main():
    rospy.init_node("tiago_navigator")
    Navigator()
    rospy.loginfo("Navigator running. Waiting for commands.")
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
