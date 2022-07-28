//
// Created by Cristian on 16/07/2022.
//

// Include Libraries
#include <iostream>
#include "../header/Detection.h"
#include "../header/Segmentation.h"

// Namespaces
using namespace cv;
using namespace std;
using namespace cv::dnn;


int main(int argc, char** argv)
{

    // -- path of the dictionary containing the map id - class
    string class_list_path = "hands_labels.names";
    // -- path of the exported trained model
    string net_path = "aug_model.onnx";
    string bb_path = "output/out.txt";
    
	// -- load input image
	Mat frame, frame_copy;
	string img_path = argv[1];
	frame = imread(img_path);
	frame_copy = frame.clone();

	if (frame.empty()) {
	    cerr << "Error reading input image!\n";
	    return -1;
	}

	Detection det = Detection(class_list_path, net_path);
	Segmentation seg;

    vector<array<int, 4>> bb_coordinates;
    vector<int> classes;

    // -- if requested performs detection and segmentation and compute performance metrics;
    // -- otherwise only performs detection and segmentation
    if (argc == 4) {
         
        // -- detection part
        cout << "DETECTION RESULTS:\n";
        det.make_detection(frame, argv[2], bb_coordinates, classes);
        
        // -- segmentation part
        cout << "\nSEGMENTATION RESULTS:\n";
        seg.make_segmentation(frame_copy, bb_coordinates, classes,argv[3]);
    
    } else if (argc == 2) {

        // -- detection part
        det.make_detection(frame, bb_coordinates, classes);
               
        // -- segmentation part
        seg.make_segmentation(frame_copy, bb_coordinates, classes);
    
    } else {
        cerr << "Error! Wrong arguments\n";
    }


	// -- show and save output
	imshow("Detection", frame);
	waitKey(0);
	imwrite("detection.jpg", frame);
	
	imshow("Segmentation", frame_copy);
	waitKey(0);
	imwrite("segmentation.jpg", frame_copy);

	return 0;
}
