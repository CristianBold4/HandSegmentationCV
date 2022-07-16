## Hand detection algorithm based on Yolov5 model for object detection

Hand detection algorithm based on Yolov5 model for object detection

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
./build/main path-to-image path-to-ground-truth.txt
```
Outputs a file ```./output/out.txt``` containing bounding boxes coordinates + class of the hand

## Example of displayed image

<img src="https://github.com/CristianBold4/CVproject/blob/main/HandDetection/input/07.jpg">

```./build/main input/07.jpg```
Results:
<img src="https://github.com/CristianBold4/CVproject/blob/main/HandDetection/output/img.jpg">
