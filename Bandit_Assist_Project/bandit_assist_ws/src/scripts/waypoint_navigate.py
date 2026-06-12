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