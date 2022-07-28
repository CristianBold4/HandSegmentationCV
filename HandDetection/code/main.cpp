//
// Created by Cristian on 16/07/2022.
//

// Include Libraries.
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include "../header/Detection.h"

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;



int main(int argc, char** argv)
{

	string class_list_path = "hands_labels.names";
	string net_path = "aug_model.onnx";


	// Load image.
	Mat frame;
	string img_path = argv[1];
	frame = imread(img_path);

	if (frame.empty()) {
	cerr << "Error reading input image!\n";
	return -1;
	}

	Detection det = Detection(class_list_path, net_path);
	det.make_detection(frame, argv[2]);
	
	det.make_detection_testset(20);
	//det.compute_avg_IoU_testset(20);

	// -- show output
	imshow("Output", frame);
	waitKey(0);

	imwrite("img.jpg", frame);

	return 0;
}
