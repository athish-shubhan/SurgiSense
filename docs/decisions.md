# Decisions

## Dataset

The SurgVU training videos are one 344 GB zip. This machine cannot store that, so training uses the public tool-localization release: seven 1 fps clips with COCO boxes (`cat1_test_set_public.zip`, about 164 MB). The label zip's `tasks.csv` and `tools.csv` describe other case ids, so they are not targets. Presence is computed from the boxes. Phase labels are not trained.

Videos 1, 2, 3, 5, and 6 are train. Video 4 is validation. Video 7 is test. Vessel sealer appears only in video 7. These numbers are about this split, not the hidden SurgVU challenge test set. Raw clips are downloaded locally and are not in git.

## Detector

A torchvision Faster R-CNN with MobileNet-V3-Large-320-FPN took 4 to 7 seconds a step here after the first compiled step. The model that was trained is MobileNetV3-Small through stride 16, then a 128-channel head. Objectness is focal loss, classes are cross-entropy on occupied cells, boxes are CIoU. ImageNet initialization, then both trunk and head are trained.

Full 640×512 training was still unfinished after 11 minutes of the first epoch (about 1.4 s/step). Inputs are resized to 320×256, which leaves a median box near 35 px on a stride-16 grid. Metrics stay in that resolution. JPEGs on disk stay 640×512.

## Temporal model

A few thousand frames is a poor fit for a transformer. The comparison is a linear classifier against a GRU on the same frozen MobileNet pool, windows 4 and 16. Augmentation on or off is the detector comparison.

## Simulation

Gazebo, Webots, and Isaac were not installed. The disk is small, and a second simulator would not test a different rule. The assistant node steps one MuJoCo 3.3.7 two-link arm. Hold sets the position target to the current joints and clears velocity, so the arm does not coast into a new pose. If MuJoCo does not import, the node uses `integrate` and writes `kinematic` in the log.

## ROS

A custom message package would mean building this workspace under ROS. The state is already JSON, so the topics are `std_msgs/String`, one process each.

Fast DDS, including a UDP-only profile, reported zero subscribers on this Mac (the ROS prefix path contains a space). CycloneDDS with shared memory off and interface `lo0` delivered the frames. The failed Fast DDS profile was removed; the working config is `ros2/cyclonedds.xml`.

## Considered and not done

Longer write-ups for the two larger changes are in `docs/adr/`.

- **Leave-one-video-out cross-validation.** The most important missing experiment. Validation and test share no tools, and both detectors score AP 0 on cadiere forceps in validation, so detector selection was decided by needle driver alone. Seven folds would turn one noisy comparison into seven paired ones. It needs 14 detector runs and was not run on this 8 GB machine. The faster validation pass (ADR 0002) removes most of the per-epoch overhead.
- **A memory-mapped cache of resized frames.** Decoding and resizing the 640x512 JPEGs is about half of a training step (110 of 227 ms measured). It was not built or verified with the disk space available.
- **Live inference inside ROS.** The ROS environment (`.rosenv`, Python 3.11, conda) has no torch, and the training environment (Python 3.10) has no rclpy. Perception replays the logged inference stream instead. Live inference is a packaging problem, not a design one.
- **Docker image for the ROS demo.** MPS is not available inside containers on macOS, and a `ros:jazzy` image with CPU torch could not be built and tested here. An unverified Dockerfile was left out.
- **A single shared backbone for detection and presence.** The temporal encoder is a second, frozen MobileNet. Sharing the detector trunk needs retraining and a new comparison.
- **A broker, Kubernetes or a model server.** One machine and three processes at 15 Hz; ROS 2 topics are already the transport.
- **QoS deadline/liveliness events instead of the watchdog.** See ADR 0001. They would complement it, but behaviour on this RMW and conda build was not tested.
