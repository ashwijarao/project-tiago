# Bandit Assist — Software Architecture

TIAGo mobile manipulator system: recognise a known person, follow on voice
command, navigate to collection points, speak/gesture on arrival, return home.

## High-level flow

```
                 ┌─────────────────┐
   laptop mic →  │ laptop_voice_    │  TCP :55555 (text)
                 │ bridge.py        │ ─────────────┐
                 └─────────────────┘               │
                                                   ▼
  RGB camera ──► follow_me_brain.py  ──────► /tiago_delivery/waypoint_command
  laser scan ──►  (face lock + voice                    │
                   gate + laser follow)                 ▼
                                            waypoint_navigate.py ──► move_base
                                                    │  (named waypoints,
                                                    │   laser pause/resume)
                                                    ▼
                                            /explain ──► explainer_node.py
                                                          (TTS + arm motion)
```

## Nodes and interfaces

### Person recognition & following

**capture_faces.py** — dataset builder
- Sub: `/xtion/rgb/image_raw` (`sensor_msgs/Image`)
- Out: grayscale 200×200 face crops to the dataset dir (auto-incrementing index)
- Haar cascade detection; 40 captures per run.

**train_faces.py** — offline trainer
- In: face dataset dir
- Out: `me_model.xml` (OpenCV LBPH recognizer). Requires `opencv-contrib-python`.

**follow_me_brain.py** — onboard "brain" (primary follower)
- Sub: `/xtion/rgb/image_raw`, `/scan` (`sensor_msgs/LaserScan`)
- Pub: `/mobile_base_controller/cmd_vel` (`geometry_msgs/Twist`),
  `/tiago_hri/voice_command`, `/tiago_delivery/waypoint_command`
- TCP server on `:55555` for the laptop voice bridge.
- State machine: `IDLE_LOOKING_FOR_FACE → WAITING_FOR_VOICE_CMD → FOLLOWING_LASER`.
- Recognises the LBPH-trained face, gates on a voice trigger, then follows the
  nearest consistent laser return. Voice tokens: `collection point`, `stop`/`halt`,
  `follow me`/`move`.

**follow_known_camera.py** — camera-only follower (alternative)
- Sub: `~image_topic` (default `/xtion/rgb/image_raw`), `~scan_topic` (default `/scan_front_raw`)
- Pub: `~cmd_vel_topic` (default `/key_vel`), `~tts_topic`, `/tiago_hri/status`
- `SEARCHING → FOLLOWING` state machine with laser hard-stop. Matches faces by
  mean pixel difference against a faces folder (independent of the LBPH model).

**face_recognition_node.py** — dlib-based greeter (alternative perception)
- Sub: `/head_front_camera/rgb/image_raw`
- Pub: `/tiago_hri/status`; speaks via `/tts` action.
- Uses the `face_recognition` (dlib) library; faces dir via `~faces_directory` param.

**image_saver.py** — manual snapshot tool (press `s`/`q`).

### Navigation

**waypoint_navigate.py** — named-waypoint navigator
- Sub: `/tiago_delivery/waypoint_command` (`std_msgs/String`), `~scan_topic`
- Pub: `/tiago_delivery/status`, `/explain`; action client to `move_base`.
- Waypoints: `home`, `collection_point_1..5` (map-frame x, y, yaw).
- Laser pause/resume: cancels and resends the goal when an obstacle is close.

**collection_goal_runner.py** — single parametric goal runner
- Params: `~goal_x`, `~goal_y`, `~goal_yaw`, `~target_frame`, `~timeout_seconds`.
- Pub: `/collection_point/task_state`; same laser pause/resume logic.

**explainer_node.py** — arrival behaviour
- Sub: `/explain` (`std_msgs/String` = location name)
- Pub: `/tts_web_command`; action client to `play_motion`.
- Per-location script + arm motion (`wave`, `home`).

### Voice

**voice_to_text.py** — ROS-side mic → publishes recognised speech to
`/tiago_hri/voice_command` when a trigger word is heard (Google STT).

**laptop_voice_bridge.py** — no-ROS laptop mic → sends recognised text over a TCP
socket to the robot. Robot IP via `ROBOT_IP` env var; port `55555`.

## Key ROS topics

| Topic | Type | Producers → Consumers |
| --- | --- | --- |
| `/xtion/rgb/image_raw` | `sensor_msgs/Image` | camera → follow_me_brain, follow_known_camera, capture_faces |
| `/scan`, `/scan_front_raw` | `sensor_msgs/LaserScan` | laser → followers, navigators |
| `/mobile_base_controller/cmd_vel` | `geometry_msgs/Twist` | followers → base |
| `/tiago_delivery/waypoint_command` | `std_msgs/String` | brain → waypoint_navigate |
| `/explain` | `std_msgs/String` | waypoint_navigate → explainer_node |
| `/tts_web_command`, `/tts` | String / TtsAction | explainer, perception → speech |
| TCP `:55555` | raw text | laptop_voice_bridge → follow_me_brain |

## Known architectural debt
- **Two dataset paths**: `~/PAR_26/src/tiago_hri/...` (capture/train/brain) vs
  `~/Bandit_Assist/.../tiago_delivery/faces` (follow_known_camera). Unify.
- **Three recognition methods**: LBPH model, dlib `face_recognition`, and raw
  pixel `absdiff`. Pick one.
- Obstacle handling is laser-only and marked as a placeholder for camera/YOLO.
