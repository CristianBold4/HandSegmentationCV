## Hand detection algorithm based on Yolov5 model for object detection

Hand detection algorithm based on Yolov5 model for object detection
Requirements
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
./build/main path-to-image 
```
Outputs a file ```./output/out.txt``` containing bounding boxes coordinates 

