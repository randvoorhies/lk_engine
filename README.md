# LKEngine

This project is a simple C++ wrapper around OpenCV's C implementation of the
Lucas and Kanade pyramidal optical flow algorithm. The class takes care of
caching and keeping track of old pyramids, as well as filtering out bad points.

## Project Status:
Currently works! Just include LKEngine.hpp in your project, and follow the example in test_lk_engine.cpp. 

## TODO:
1) Provide performance numbers of caching/no-caching, C++-version/C-version, etc.
2) Implement forward-backward filtering
