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


	// -- load input image
	Mat frame, frame_copy;
	string img_path = argv[1];
	frame = imread(img_path);
	frame_copy = frame.clone();

	if (frame.empty()) {
	    cerr << "Error reading input image!\n";
	    return -1;
	}

    // -- detection part
	Detection det = Detection(class_list_path, net_path);

    if (argc == 3) {
        det.make_detection(frame, argv[2]);
    } else {
        det.make_detection(frame);
    }



    // -- segmentation part
	string gt_mask_path = argv[3];
	Mat gt_mask = imread(gt_mask_path);
	if (gt_mask.empty()) {
	cerr << "Error reading ground truth image for segmentation!\n";
	return -1;
	}
	
	
	//det.make_detection_testset(20);
	//det.compute_avg_IoU_testset(20);

	// -- show output
	imshow("Output", frame);
	waitKey(0);

	imwrite("detection.jpg", frame);
	
	
	Segmentation seg;
	vector<array<int, 4>> boxes_vec;
	vector<int> class_labels;
	
	string bb_path = "output/out.txt";
	
	Mat src_bb,tres_diff, segmented;
	seg.read_bb_file_label(frame_copy.rows, frame_copy.cols, bb_path, boxes_vec, class_labels);
	//seg.draw_box_image_label(frame_copy, src_bb, boxes_vec, class_labels, true);
	//seg.show_image(src_bb, "src_bb");
	
	vector<Mat> difference_bb_vec;
	seg.difference_from_center_hand_label(frame_copy, difference_bb_vec, boxes_vec, class_labels);
	
	vector<Mat> treshold_bb_vec;
	seg.treshold_difference(difference_bb_vec, treshold_bb_vec);
	
	seg.segmentation_GB_mask(frame_copy, segmented, tres_diff, treshold_bb_vec, boxes_vec, class_labels);
	
	seg.apply_mask(frame_copy, segmented, segmented, false);
	seg.show_image(segmented, "GB-mask segmentation"); 
	imwrite("segmentation.jpg", segmented);
	
    cvtColor(gt_mask, gt_mask, COLOR_BGR2GRAY);
	float pixel_accuracy = seg.compute_pixel_accuracy(tres_diff, gt_mask);
	float IOU = seg.compute_IOU(tres_diff, gt_mask);
	
	cout<< "Segmentation:   PA:"<< pixel_accuracy << "      IOU:" << IOU << "\n";

	return 0;
}
