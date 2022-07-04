#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "Segmentation.h"

using namespace std;
using namespace cv;

int main(int argc, char ** argv) {

    string bb_path = "../det/12.txt";
    Mat src = imread("../rgb/12.jpg");

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


    /*
    // compute difference image
    Mat difference;
    seg.difference_from_center(src, difference, boxes_vec);
    namedWindow("diff", WINDOW_AUTOSIZE);
    imshow("diff", difference);
    waitKey(0);

    //si potrebbe fare treshold della difference ma devo trovare un buon valore.
    threshold(difference, difference,30, 250,THRESH_BINARY_INV);
    namedWindow("tresh_diff", WINDOW_AUTOSIZE);
    imshow("tresh_diff", difference);
    waitKey(0);
    */


    // segmentation with GrabCut
    seg.draw_segmentation_GB(src, segmented, boxes_vec);
    seg.draw_box_image(segmented, boxed, boxes_vec);
    namedWindow("image", WINDOW_AUTOSIZE);
    imshow("image", boxed);
    waitKey(0);

    return 0;

}