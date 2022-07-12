#include <iostream>
#include <fstream>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "Segmentation.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	Segmentation seg = Segmentation();
	string bb_path, rgb_path, mask_path;
	vector<int> test_indeces = {   1,2,3,4,5,6,7,8,9,10, 11,12,13,14,15,16,17,18,19,20 };
	//vector<int> test_indeces = { 17 };	//use this vector to choose images from test dataset
	int test_image_num = 1;
	int test_num = 1;
	string method = "GrabCut-mask";
	string out_name = "segmentation_results_v" + to_string(test_num) + ".txt";
	ofstream out_file(out_name);
	out_file << "SEGMENTATION RESULTS\nMethod used: " << method << ";\ntest num. " << test_num<< "\n\n";
	float pixel_accuracy_sum = 0;
	float IOU_sum = 0;
	for (int k = 0; k < test_indeces.size(); k++) {
		test_image_num = test_indeces[k];
		//remember to change the det folder to det_label if using labels
		if (test_image_num < 10) {
			bb_path = "../../../det_label/0" + to_string(test_image_num) + ".txt";
			rgb_path = "../../../rgb/0" + to_string(test_image_num) + ".jpg";
			mask_path = "../../../mask/0" + to_string(test_image_num) + ".png";
		}
		else {
			bb_path = "../../../det_label/" + to_string(test_image_num) + ".txt";
			rgb_path = "../../../rgb/" + to_string(test_image_num) + ".jpg";
			mask_path = "../../../mask/" + to_string(test_image_num) + ".png";
		}

		Mat src = imread(rgb_path);
		Mat gt_mask = imread(mask_path);
		if (src.empty()) {
			cerr << "TEST IMG " << to_string(test_image_num) << ": " << "Error! Input image is empty\n";
		}
		if (gt_mask.empty()) {
			cerr << "TEST IMG " << to_string(test_image_num) << ": " << "Error! ground truth mask image is empty\n";
		}
		vector<array<int, 4>> boxes_vec;
		vector<int> class_labels;
		//seg.read_bb_file(bb_path, boxes_vec);
		seg.read_bb_file_label(bb_path, boxes_vec, class_labels);
		//for (int i = 0; i < class_labels.size(); i++) cout << class_labels[i] << "\n";

		Mat src_bb;
		seg.draw_box_image_label(src, src_bb, boxes_vec, class_labels, true);
		//seg.draw_box_image(src, src_bb, boxes_vec);
		seg.show_image(src_bb, to_string(test_image_num) + ")src_bb");

		//NORMAL
		Mat gt_segmentation;
		seg.apply_mask(src, gt_segmentation, gt_mask, true);
		//seg.show_image(gt_segmentation, to_string(test_indeces[k]) + ") GT segmentation");

		//Mat pyr_filtered;
		//pyrMeanShiftFiltering(src, pyr_filtered, 2, 15, 2);
		//seg.show_image(pyr_filtered, to_string(test_indeces[k]) + ") pyrMeanShiftFiltered");

		//compute difference image
		Mat difference, tres_diff, gb_th;
		seg.difference_from_center_hand_label(src, difference, boxes_vec, class_labels);
		//seg.difference_from_center_hand(src, difference, boxes_vec);
		//seg.show_image(difference, to_string(test_indeces[k]) + ") difference from skin");

		seg.treshold_difference(difference, tres_diff, boxes_vec);
		//seg.show_image(tres_diff, to_string(test_indeces[k]) + ") difference tresholded");
		
		seg.draw_segmentation_GB_mask(src, gb_th, tres_diff, boxes_vec);
		//seg.show_image(gb_th,to_string(test_indeces[k]) +") GB mask");
		//seg.show_image(tres_diff, to_string(test_indeces[k]) + ") bin mask");

		seg.apply_mask(src, gb_th, gb_th, false);
		//seg.show_image(gb_th, to_string(test_indeces[k]) +") GB-mask segmentation");

		cvtColor(gt_mask, gt_mask, COLOR_BGR2GRAY);
		
		float pixel_accuracy = seg.compute_pixel_accuracy(tres_diff, gt_mask);
		float IOU = seg.compute_IOU(tres_diff, gt_mask);
		pixel_accuracy_sum += pixel_accuracy;
		IOU_sum += IOU;
		String arrow = "";
		if(pixel_accuracy < 0.975) arrow = "*";
		cout << "Img " << to_string(test_image_num) << ")	" << "PA: " << pixel_accuracy << arrow << ";	IOU: " << IOU <<  " \n";
		out_file << "Img " << to_string(test_image_num) << ")	" << "PA: " << pixel_accuracy << arrow << ";	IOU: " << IOU << " \n";

		seg.draw_box_image(gb_th, gb_th, boxes_vec);
		seg.show_image(gb_th, to_string(test_indeces[k]) + ") GB-mask segmentation+bb");
			
	}
	
	cout << "\naverage PA: " << pixel_accuracy_sum/test_indeces.size() << " \n";
	out_file <<  "\naverage PA: " << pixel_accuracy_sum / test_indeces.size() << " \n";

	cout << "\naverage IOU: " << IOU_sum / test_indeces.size() << " \n";
	out_file << "\naverage IOU: " << IOU_sum / test_indeces.size() << " \n";

	out_file.close();
	return 0;
}

