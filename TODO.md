### THings to do

MAPPER
[] read from rgbd camera realsense
[] rtab SLAM working - rviz full map view
[] clean pointcloud, wireframe extract
[] wireframe pass it to quest


DRONE CONTROL
[] a preprogrammed or remote control of a small trajectgory
[] jetson on board with realsense attached, ssh from laptop to control, and powered on


VR VIZUALTION
[] wireframe layout with just a cube, communication over websocket
[] webserver launch through laptop or jetson


PERCEPTION
[] get yolo running for object/person detection
[] wireframe of choosen object should be displayed differently/highlighted
[] string input from high-level for object name

AGENT
[] some small stt model -> yolo input
[] probably drone commands with voice - agent to control - dimos stack


INTEGRATION
[] Quest frame wrt global frame transformtation to layover wireframe of room layout correctly
probably need aruco - just for tutorial

MISC
[] combine scripts for launching - lowP


DEMO
[] realtime - person moving behind the wall - should be visible in quest



### THINGS TO DO (Claude additions)

SPECS (lock these before code drifts — CLAUDE.md says specs are contracts)
[] freeze the "scene" JSON wire format in shared/protocol.md (Cube schema: center, size, quat, color; reserve a future delta-edges message type)
[] document SLAM-frame → WebXR-local-floor transform in shared/frames.md (RTAB-Map +Z-up robot frame vs three.js Y-up, metres, handedness) and put the helper in headset/scene.py so emitters can't get it wrong

LATENCY (measure from day one — don't guess)
[] per-stage timestamps end-to-end: depth capture → SLAM pose → wireframe extract → JSON encode → WS send → browser receive → first render. Log per-frame ms; surface a /stats endpoint
[] bandwidth budget: bytes/sec on /ws at expected map size (N edges × bytes/edge × Hz). Pick a LAN ceiling and a degraded-link mode before the live map goes in

MAPPER → HEADSET
[] wireframe diff protocol: send only added/removed edges per tick, not the full scene. Full snapshot on connect / on demand. Needed before edge count climbs past a few hundred or push rate past ~10 Hz
[] perception channel separate from geometry: highlighted objects (people, target) as a different message type with its own colour/blink, so the renderer doesn't have to re-derive what's special

INTEGRATION
[] aruco anchor flow: print marker, fix it at a known SLAM-frame pose, Quest detects it (passthrough camera API or controller-pointed click), origin snaps. Concrete mechanism for the "Quest frame wrt global frame" item already in the list
[] re-anchor / recenter button in the AR UI for when WebXR local-floor relocalises mid-session (battery swap, headset re-pair, walked between rooms)

OPS
[] simple shared-token auth on /ws (?token=… query param) so any random Quest on the LAN can't join and pull the map
[] LAN reachability check at server startup — bind, print all reachable IPs, warn if firewalled. Saves debugging on the field network
