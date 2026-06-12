import rospy

import actionlib

from std_msgs.msg import String

from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal

LOCATIONS = {

    "home": {

        "script": "reached arena",

        "motion": "home",

    },

    "collection_point_1": {

        "script": "hey i have reached collection location please hand over the bag",

        "motion": "wave",

    },

    "collection_point_2": {

        "script": "hey i have reached collection location please hand over the bag",

        "motion": "wave",

    },

    "collection_point_3": {

        "script": "hey i have reached collection location please hand over the bag",

        "motion": "wave",

    },

    "collection_point_4": {

        "script": "hey i have reached collection location please hand over the bag",

        "motion": "wave",

    },

    "collection_point_5": {

        "script": "hey i have reached collection location please hand over the bag",

        "motion": "wave",

    },

}


def build_motion_goal(motion_name):
    
    goal = PlayMotionGoal()

    goal.motion_name = motion_name

    goal.skip_planning = False

    return goal


