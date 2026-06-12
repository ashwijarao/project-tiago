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

class Explainer:

    def __init__(self):

        self.tts_pub = rospy.Publisher(

            "/tts_web_command",

            String,

            queue_size=1

        )

        self.motion_client = actionlib.SimpleActionClient(

            "/play_motion",

            PlayMotionAction

        )

        rospy.loginfo("Waiting for play_motion action server...")

        connected = self.motion_client.wait_for_server(rospy.Duration(10.0))

        if connected:

            rospy.loginfo("play_motion ready.")

            self.motion_available = True

        else:

            rospy.logwarn(

                "play_motion not available. Speech will work, but arm motion will be skipped."

            )

            self.motion_available = False

        rospy.Subscriber(

            "/explain",

            String,

            self.on_arrival

        )

        rospy.loginfo("Explainer node started.")

    def say(self, text):

        rospy.loginfo("Speaking: {}".format(text))

        self.tts_pub.publish(String(data=text))

    def play_motion(self, motion_name):

        if not self.motion_available:

            rospy.logwarn("Skipping motion: {}".format(motion_name))

            return

        goal = build_motion_goal(motion_name)

        self.motion_client.send_goal_and_wait(goal)

    def on_arrival(self, msg):

        location = msg.data.strip().lower()

        if location not in LOCATIONS:

            rospy.loginfo(

                "No speech or motion needed for location: {}".format(location)

            )

            return

        script = LOCATIONS[location]["script"]

        motion_name = LOCATIONS[location]["motion"]

        rospy.loginfo(

            "Arrived at {}. Speaking and moving arm.".format(location)

        )

        self.say(script)

        rospy.sleep(1.0)

        self.play_motion(motion_name)

        rospy.sleep(1.0)

        if location.startswith("collection_point"):

            self.play_motion("home")

            rospy.sleep(1.0)

def main():

    rospy.init_node("explainer_node")

    Explainer()

    rospy.loginfo("Explainer running. Waiting for arrival signals.")

    rospy.spin()

if __name__ == "__main__":

    try:

        main()

    except rospy.ROSInterruptException:

        pass