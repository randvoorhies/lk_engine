# LKEngine

This project is a simple C++ wrapper around OpenCV's C implementation of the
Lucas and Kanade pyramidal optical flow algorithm. The class takes care of
caching and keeping track of old pyramids, as well as filtering out bad points.
