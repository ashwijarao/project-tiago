# Bandit Assist: Software Architecture

TIAGo mobile manipulator system: recognise a known person, follow on voice
command, navigate to collection points, speak and gesture on arrival, then
return home.

## High level flow

1. The laptop microphone feeds `laptop_voice_bridge.py`, which sends recognised
   text over TCP port 55555 to the robot.
2. On the robot, `follow_me_brain.py` reads the RGB camera and laser scan. It
   locks onto the known face, waits for a voice trigger, then follows the person
   using the laser.
3. When a collection command arrives, `follow_me_brain.py` publishes a waypoint
   name to `/tiago_delivery/waypoint_command`.
4. `waypoint_navigate.py` receives that name and drives `move_base` to the
   matching waypoint, pausing and resuming around close obstacles.
5. On arrival it publishes the location to `/explain`, and `explainer_node.py`
   speaks a line and plays an arm motion.

## Nodes and interfaces

### Person recognition and following

**capture_faces.py** (dataset builder)
Subscribes: `/xtion/rgb/image_raw` (sensor_msgs/Image).
Writes grayscale 200x200 face crops to the dataset directory, with an
automatically incrementing index. Haar cascade detection, 40 captures per run.

**train_faces.py** (offline trainer)
Reads the face dataset directory. Writes `me_model.xml`, an OpenCV LBPH
recognizer. Requires the OpenCV contrib modules.

**follow_me_brain.py** (primary follower, runs on the robot)
Subscribes: `/xtion/rgb/image_raw`, `/scan` (sensor_msgs/LaserScan).
Publishes: `/mobile_base_controller/cmd_vel` (geometry_msgs/Twist),
`/tiago_hri/voice_command`, `/tiago_delivery/waypoint_command`.
Also runs a TCP server on port 55555 for the laptop voice bridge.
State machine: IDLE_LOOKING_FOR_FACE, then WAITING_FOR_VOICE_CMD, then
FOLLOWING_LASER. It recognises the LBPH trained face, gates on a voice trigger,
then follows the nearest consistent laser return. Voice tokens: "collection
point", "stop" or "halt", "follow me" or "move".

**follow_known_camera.py** (camera only follower, alternative)
Subscribes: `~image_topic` (default `/xtion/rgb/image_raw`), `~scan_topic`
(default `/scan_front_raw`).
Publishes: `~cmd_vel_topic` (default `/key_vel`), `~tts_topic`,
`/tiago_hri/status`.
SEARCHING then FOLLOWING state machine with a laser hard stop. Matches faces by
mean pixel difference against a faces folder, independent of the LBPH model.

**face_recognition_node.py** (dlib based greeter, alternative perception)
Subscribes: `/head_front_camera/rgb/image_raw`.
Publishes: `/tiago_hri/status`. Speaks via the `/tts` action.
Uses the face_recognition (dlib) library. Faces directory set by the
`~faces_directory` param.

**image_saver.py** (manual snapshot tool)
Press `s` to save a frame, `q` to quit.

### Navigation

**waypoint_navigate.py** (named waypoint navigator)
Subscribes: `/tiago_delivery/waypoint_command` (std_msgs/String), `~scan_topic`.
Publishes: `/tiago_delivery/status`, `/explain`. Action client to `move_base`.
Waypoints: home, collection_point_1 through collection_point_5, each a map frame
x, y, yaw. Laser logic cancels and resends the goal when an obstacle is close.

**collection_goal_runner.py** (single parametric goal runner)
Params: `~goal_x`, `~goal_y`, `~goal_yaw`, `~target_frame`, `~timeout_seconds`.
Publishes: `/collection_point/task_state`. Same laser pause and resume logic.

**explainer_node.py** (arrival behaviour)
Subscribes: `/explain` (std_msgs/String, the location name).
Publishes: `/tts_web_command`. Action client to `play_motion`.
Per location script plus an arm motion (wave, home).

### Voice

**voice_to_text.py** (ROS side microphone)
Publishes recognised speech to `/tiago_hri/voice_command` when a trigger word is
heard. Uses Google speech to text.

**laptop_voice_bridge.py** (laptop microphone, no ROS needed)
Sends recognised text over a TCP socket to the robot. Robot IP via the
`ROBOT_IP` environment variable, port 55555.

## Key ROS topics

`/xtion/rgb/image_raw` (sensor_msgs/Image): camera, consumed by
follow_me_brain, follow_known_camera, capture_faces.

`/scan` and `/scan_front_raw` (sensor_msgs/LaserScan): laser, consumed by the
followers and navigators.

`/mobile_base_controller/cmd_vel` (geometry_msgs/Twist): followers command the
base.

`/tiago_delivery/waypoint_command` (std_msgs/String): brain to
waypoint_navigate.

`/explain` (std_msgs/String): waypoint_navigate to explainer_node.

`/tts_web_command` and `/tts` (String / TtsAction): explainer and perception
drive speech.

TCP port 55555 (raw text): laptop_voice_bridge to follow_me_brain.

## Known architectural debt

Two dataset paths: `~/PAR_26/src/tiago_hri/...` used by capture, train and
brain, versus `~/Bandit_Assist/.../tiago_delivery/faces` used by
follow_known_camera. These should be unified.

Three recognition methods: the LBPH model, the dlib face_recognition library,
and the raw pixel difference match. Pick one.

Obstacle handling is laser only and is marked as a placeholder for future camera
or YOLO based detection.
