# Vision Pipelines for DEEP SPACE
This repo includes two methods for aligning to targets:
1. `SkewPairTargetProcessor`, which works together with a `FindTargetsPipeline` to do the
standard "sweet spot" style tracking.  It provides an extra parameter, "skew", which encodes info
about the difference in height of the two targets.
2. `Model3DPipeline`, which works independently and finds the 3D position and orientation of the 
target relative to the camera.

## Running
You can run this as a java application with `./gradlew run`.  Start up shuffleboard and connect to
localhost to use.  The filename of a test image to use is stored as a preference.  After changing 
it, you need to restart the program for it to take effect.