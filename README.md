# CVproject

Computer Vision project 2022

<ul>
  <li> Boldrin Cristian </li> <br>
  <li> Bellan Riccardo </li> <br>
</ul>

<h1> Models for Hand Detection </h1>

* First model: spec of EfficientDet-3, trained on whole EgoHandsDataset with 4800 images (3829 training, 476 validation, 495 test), 15 epochs and batch size 16
* Second model: transfer learning from yolov5s trained on whole EgoHandsDataset for 100 epochs, batch size 64
* Fine-tuned-model
* Augmented-model 

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

