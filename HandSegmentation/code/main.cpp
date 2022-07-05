#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "Segmentation.h"

using namespace std;
using namespace cv;

int main(int argc, char ** argv) {

    string bb_path = "../../../det/07.txt";
    Mat src = imread("../../../rgb/07.jpg");

    if (src.empty()) {
        cerr << "Error! Input image is empty\n";
    }
    Segmentation seg = Segmentation();
    Mat segmented, boxed, clustered;

    vector<array<int, 4>> boxes_vec;
    boxes_vec = seg.read_bb_file(bb_path);

    //show source image with bounding boxes
    seg.draw_box_image(src, boxed, boxes_vec);
    namedWindow("src", WINDOW_AUTOSIZE);
    imshow("src", boxed);
    waitKey(0);


/*
	// segmentation with k-means
	seg.draw_segmentation_Km(src, segmented, boxes_vec);
	namedWindow("segmentation", WINDOW_AUTOSIZE);
	imshow("segmentation", segmented);
	waitKey(0);
	*/


	// compute difference image
	Mat difference, tres_diff, tres_diff_bb, gb_th;
	seg.difference_from_center(src, difference, boxes_vec);
	//namedWindow("diff", WINDOW_AUTOSIZE);
	//imshow("diff", difference);
	//waitKey(0);

	seg.treshold_difference(difference, tres_diff, boxes_vec);
	namedWindow("tf", WINDOW_AUTOSIZE);
	imshow("tf", tres_diff);
	waitKey(0);


	seg.draw_segmentation_GB_mask(src, gb_th, tres_diff, boxes_vec);
	//namedWindow("gbmask", WINDOW_AUTOSIZE);
	//imshow("gbmask", gb_th);
	//waitKey(0);
	seg.draw_box_image(gb_th, gb_th, boxes_vec);
	namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", gb_th);
	waitKey(0);


    return 0;

}