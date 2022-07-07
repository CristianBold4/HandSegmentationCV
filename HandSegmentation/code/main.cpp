#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "Segmentation.h"

using namespace std;
using namespace cv;

int main(int argc, char ** argv) {
	Segmentation seg = Segmentation();
    string bb_path = "../../../det/01.txt";
    Mat src = imread("../../../rgb/01.jpg");
	Mat gt_mask = imread("../../../mask/01.png");
	cvtColor(gt_mask, gt_mask, COLOR_BGR2GRAY);


	Mat gt_segmentation;
	seg.apply_mask(src, gt_segmentation, gt_mask);
	namedWindow("GT segmentation", WINDOW_AUTOSIZE);
	imshow("GT segmentation", gt_segmentation);
	waitKey(0);
	
    if (src.empty()) {
        cerr << "Error! Input image is empty\n";
    }
   
    Mat segmented, boxed, clustered;

    vector<array<int, 4>> boxes_vec;
    boxes_vec = seg.read_bb_file(bb_path);

    //show source image with bounding boxes
    seg.draw_box_image(src, boxed, boxes_vec);
    //namedWindow("src", WINDOW_AUTOSIZE);
    //imshow("src", boxed);
    //waitKey(0);

	// segmentation with k-means
	seg.draw_segmentation_Km(src, segmented, boxes_vec);
	//namedWindow("segmentation", WINDOW_AUTOSIZE);
	//imshow("segmentation", segmented);
	//waitKey(0);
	
	//compute difference image
	Mat difference, tres_diff, tres_diff_bb, gb_th;
	seg.difference_from_center(src, difference, boxes_vec);
	//namedWindow("diff", WINDOW_AUTOSIZE);
	//imshow("diff", difference);
	//waitKey(0);

	seg.treshold_difference(difference, tres_diff, boxes_vec);
	//namedWindow("tf", WINDOW_AUTOSIZE);
	//imshow("tf", tres_diff);
	//waitKey(0);

	seg.draw_segmentation_GB_mask(src, gb_th, tres_diff, boxes_vec);
	//namedWindow("gbmask", WINDOW_AUTOSIZE);
	//imshow("gbmask", gb_th);
	//waitKey(0);

	seg.draw_box_image(gb_th, gb_th, boxes_vec);
	namedWindow("my segmentation", WINDOW_AUTOSIZE);
	imshow("my segmentation", gb_th);
	waitKey(0);

	float pixel_accuracy = seg.compute_pixel_accuracy(tres_diff, gt_mask);
	cout << "Segmentation test results\n";
	cout << "pixel accuracy: " << pixel_accuracy << " \n";

	//namedWindow("mask", WINDOW_AUTOSIZE);
	//imshow("mask", tres_diff);
	//waitKey(0);

	//namedWindow("GT mask", WINDOW_AUTOSIZE);
	//imshow("GT mask", gt_mask);
	//waitKey(0);

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