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
	// Load class list.
	vector<string> class_list;
	ifstream ifs("hands_labels.names");
	string line;


	while (getline(ifs, line))
	{
	class_list.push_back(line);
	}
	
	// Load model: read net
	Net net;
	net = readNet("aug_model.onnx");
	


    // Load image.
    Mat frame;
    string img_path = argv[1];
    frame = imread(img_path);

    if (frame.empty()) {
        cerr << "Error reading input image!\n";
        return -1;
    }

    Detection det = Detection();

    vector<Mat> detections;
    detections = det.pre_process(frame, net);


    // read ground truth
    vector<array<int, 4>> gr_boxes_vec;
    det.read_bb_file(argv[2], gr_boxes_vec);

    string IoU;
    Mat img = det.post_process(frame, detections, class_list , gr_boxes_vec, IoU);

    // Put efficiency information.
    // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the 	 layers(in layersTimes)

    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string inference_time = format("Inference time : %.2f ms", t);
    //putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
    cout << inference_time << endl;

    imshow("Output", img);
    waitKey(0);

    imwrite("img.jpg", img);

	return 0;
}
