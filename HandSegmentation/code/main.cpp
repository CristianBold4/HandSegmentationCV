#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "Segmentation.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	Segmentation seg = Segmentation();
	string bb_path, rgb_path, mask_path;
	vector<int> test_indeces = { 1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 };
	//vector<int> test_indeces = { 2,18,16 };	//use this vector to choose images from test dataset
	int test_image_num = 1;
	for (int k = 0; k < test_indeces.size(); k++) {
		test_image_num = test_indeces[k];

		if (test_image_num < 10) {
			bb_path = "../../../det/0" + to_string(test_image_num) + ".txt";
			rgb_path = "../../../rgb/0" + to_string(test_image_num) + ".jpg";
			mask_path = "../../../mask/0" + to_string(test_image_num) + ".png";
		}
		else {
			bb_path = "../../../det/" + to_string(test_image_num) + ".txt";
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
		vector<array<int, 4>> boxes_vec = seg.read_bb_file(bb_path);

		Mat src_bb;
		seg.draw_box_image(src, src_bb, boxes_vec);
		//seg.show_image(src_bb, to_string(test_image_num) + ")src_bb");


		//NORMAL
		
		Mat gt_segmentation;
		seg.apply_mask(src, gt_segmentation, gt_mask, true);
		//seg.show_image(gt_segmentation, to_string(test_indeces[k]) + ") GT segmentation");

		//compute difference image
		Mat difference, tres_diff, gb_th;
		seg.difference_from_center_hand(src, difference, boxes_vec);
		//seg.show_image(difference, to_string(test_indeces[k]) + ") difference from skin");

		seg.treshold_difference(difference, tres_diff, boxes_vec);
		//seg.show_image(tres_diff, to_string(test_indeces[k]) + ") difference tresholded");

		seg.draw_segmentation_GB_mask(src, gb_th, tres_diff, boxes_vec);
		//seg.show_image(gb_th,to_string(test_indeces[k]) +") GB mask");


		seg.apply_mask(src, gb_th, gb_th, false);
		//seg.show_image(gb_th, to_string(test_indeces[k]) +") GB-mask segmentation");

		cvtColor(gt_mask, gt_mask, COLOR_BGR2GRAY);
		float pixel_accuracy = seg.compute_pixel_accuracy(tres_diff, gt_mask);
		cout << "TEST IMG " << to_string(test_image_num) << ": " << "pixel accuracy : " << pixel_accuracy << " \n";

		seg.draw_box_image(gb_th, gb_th, boxes_vec);
		seg.show_image(gb_th, to_string(test_indeces[k]) + ") GB-mask segmentation+bb");
		

		//SUPERPIXEL
		/*
		Mat superpixel;
		seg.get_superpixel_image(src, superpixel);
		seg.show_image(superpixel, to_string(test_image_num) + ") superpixel image");

		Mat difference, tres_diff, gb_th;
		seg.difference_from_center_hand(superpixel, difference, boxes_vec);
		//seg.show_image(difference, to_string(test_image_num) + ") difference from skin");

		seg.treshold_difference(difference, tres_diff, boxes_vec);
		//seg.show_image(tres_diff, to_string(test_image_num) + ") difference tresholded");

		seg.draw_segmentation_GB_mask(superpixel, gb_th, tres_diff, boxes_vec);
		//seg.show_image(gb_th,to_string(test_image_num) + ")GB mask");

		seg.apply_mask(src, gb_th, gb_th, false);
		//seg.show_image(gb_th, to_string(test_image_num) + ") GB-mask segmentation");

		seg.draw_box_image(gb_th, gb_th, boxes_vec);
		seg.show_image(gb_th, to_string(test_indeces[k]) + ") GB-mask segmentation+bb");
		*/

	}

	return 0;
}





/*
Mat src_ycc,merged;
cvtColor(src, src_ycc, COLOR_BGR2YCrCb);
Mat Y(src.rows, src.cols, CV_8U);
Mat channels[3];
split(src_ycc, channels);
//namedWindow("segmentation", WINDOW_AUTOSIZE);
//imshow("segmentation", channels[0]);
//waitKey(0);

for (int i = 0; i < src_ycc.rows; i++) {
	for (int j = 0; j < src_ycc.cols; j++) {
		Y.at<unsigned char>(i, j) = src_ycc.at<Vec3b>(i, j)[0];

	}
}

//merge(channels, 3,merged );
//cvtColor(merged, merged, COLOR_YCrCb2BGR);
namedWindow("r", WINDOW_AUTOSIZE);
imshow("r",Y);
waitKey(0);

*/