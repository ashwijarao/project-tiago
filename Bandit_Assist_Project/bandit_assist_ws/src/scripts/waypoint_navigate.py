#!/usr/bin/env python3

import rospy

import actionlib

import tf

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