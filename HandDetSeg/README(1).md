## Hand detection algorithm based on Yolov5 model for object detection

Hand detection and segmentation algorithm based on:
-   Yolov5 model for object detection;
-   Grabcut algorithm for object segmentation;

Requirements:
- OpenCV >= 4.5.2
- GCC compiler

## Execution

### CMake C++ Linux
```C++ Linux
mkdir build
cd build
cmake ..
cmake --build .
cd ..
./build/main path-to-image path-to-detection-ground-truth.txt path-to-segmentation-ground-truth.png
```

