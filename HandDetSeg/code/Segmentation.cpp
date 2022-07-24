#include "../header/Segmentation.h"
#include <iostream>
#include <fstream>


using namespace cv;
using namespace std;

/* @brief Checks if the given bounding boxes size and coordinates are valid and consistent with the given image size.
* This method checks if each array stored into bb_vector correctly describes a bounding box contained inside a given image. 
* Each bounding box should be identified by an array of 4 int, according with the following notation:[ (top-lef corner x coordinate), (top-lef corner y coordinate), (width), (height) ]
* The data inside bb_vector are considered valid if each integer value is nonnegative and if each bounding box is completely contained into an image with  src_r rows and  src_c columns
* 
@param src_r the number of rows of the image that should contain the bounding boxes
@param src_c the number of columns of the image that should contain the bounding boxes
@param bb_vector a vector of array, each array containing the coordinates and size of a single bounding box
@return true if the data inside bb_vector are valid; false otherwise
**/
bool Segmentation::valid_bb_cordinates(int src_r, int src_c, const std::vector<std::array<int, 4>>& bb_vector)
{
	for (int k = 0; k < bb_vector.size(); k++) {
		int x = bb_vector[k][0];
		int y = bb_vector[k][1];
		int w = bb_vector[k][2];
		int h = bb_vector[k][3];

		if (x < 0 || y < 0 || w < 0 || h < 0) { return false; }
		if (x > src_c || y > src_r || (x + w) > src_c || (y + h) > src_r) {  return false; }
	}
	return true;
}



/* @brief Returns the coordinates of the given bounding boxes enlarged of a specific quantity of pixels.
* This method computes and returns the coordinates of the bounding boxes obtained by enlarging each bounding box contained into bb_vector.
* Each bounding box is enlarged by adding a given quantity of rows or columns on each side of the box.
* If the enlarged box is not completely contained into the source image the original cordinates are returned.
* A single bounding box is identified by an array of 4 int: top-lef corner x cordinate, top-lef corner y cordinate, width and height.
* 
@param src_r the number of rows of the image that should contain the bounding boxes
@param src_c the number of columns of the image that should contain the bounding boxes
@param bb_vector a vector of array, each array containing the cordinates and size of a single original bounding box
@return a vector of array, each array containing the coordinates and size of a single enlarged bounding box.

**/
std::vector<std::array<int, 4>> Segmentation::get_wide_cordinates(int src_r, int src_c, const std::vector<std::array<int, 4>>& bb_vector)
{
	int delta = 5;
	vector<array<int, 4>> wide_bb_vector;

	for (int k = 0; k < bb_vector.size(); k++) {
		int x = bb_vector[k][0];
		int y = bb_vector[k][1];
		int w = bb_vector[k][2];
		int h = bb_vector[k][3];
		
		//compute wide cordinates
		array<int, 4> wide_bb = { x - delta, y - delta, w + delta*2, h + delta*2 };
		vector<std::array<int, 4>> check_vec = { wide_bb };
		//if wide cordinates are valid use them; otherwise use original cordinates
		if (valid_bb_cordinates(src_r, src_c, check_vec)) {	wide_bb_vector.push_back(wide_bb); }
		else { wide_bb_vector.push_back(bb_vector[k]); }
	}
	return wide_bb_vector;	
}


/*@brief Draws the specified bounding boxes into the provided image, with colors based on labels.
* This method draws the bounding boxes specified in bb_vector into the image provided by src and save the result into dst. 
* A single bounding box is identified by an array of 4 int: top-lef corner x coordinate, top-lef corner y coordinate, width and height.
* The color of each bounding box depends on the class label associated with such box: 
* - purple for my_left (0);
* - yellow for my_right (1);
* - red for your_left (2);
* - green for your_right (3);

@param src input image where to draw the bounding boxes
@param dst output image 
@param bound_boxes vector of arrays containing the coordinates of the bounding boxes
@param labels a vector of int, each element containing the label associated with the corresponding bounding box in bb_vector.
@param  show_color_point if set to true allows to visualize the points selected by the method difference_from_center_hand_label to approximate the skin color
**/
void Segmentation::draw_box_image_label(const cv::Mat& src, cv::Mat& dst, const std::vector<std::array<int, 4>>& bb_vector, const std::vector<int>& labels, bool show_color_point)
{
	dst = src.clone();
	for (int k = 0; k < bb_vector.size(); k++) {
		int x = bb_vector[k][0];
		int y = bb_vector[k][1];
		int w = bb_vector[k][2];
		int h = bb_vector[k][3];
		rectangle(dst, Point(x, y), Point(x + w, y + h), colors[labels[k]] * 0.7, 1, 1);
		if (show_color_point) {
			int label = labels[k];
			circle(dst, Point(x + w / 2, y + h / 2), 3, colors[label], 2);
			if (label  == 0) circle(dst, Point(x + w / 3, y + h * 2 / 3), 3, Scalar(0,0,0), 2);
			if (label == 1) circle(dst, Point(x + w * 2 / 3, y + h * 2 / 3), 3, Scalar(0, 0, 0), 2);
			if (label == 2) circle(dst, Point(x + w * 2 / 3, y + h / 3), 3, Scalar(0, 0, 0), 2);
			if (label == 3) circle(dst, Point(x + w / 3, y + h / 3), 3, Scalar(0, 0, 0), 2);
		}
	}
}



/* @brief Method that performs hand segmentation on the provided image, exploiting K-means clustering.
* This method consider separately the portion of the src image insight each of the specified bounding boxes. Then it performs binary segmentation on such portion, using K-means.
* The K-means algorithm is performed in a vector space where each vector is associated to a single pixel and it contains: 
* - the Cb channel value of the pixel;
* - the Cr channel value of the pixel;
* - the euclidean distance between the pixel and the center of the bounding box it belongs to;
* The results of the segmentation are stored into a colored mask (a specific color for each hand) and a into binary mask (white for hand, black for not hand).
* The color for each hand in the colored mask is chosen according to the class label associated with the considered bounding box:
* - purple for my_left (0);
* - yellow for my_right (1);
* - red for your_left (2);
* - green for your_right (3);

@param src input image where to perform segmentation
@param col_mask output colored image (with black background) that contains the mask computed for each hand (with different colors).
@param bin_mask output grayscale image that store the final binary mask of the segmentation with classes hand / not hand.
@param bb_vector vector of arrays containing the coordinates of the bounding boxes.
@param class_labels a vector of int, each element containing the label associated with the corresponding bounding box in bb_vector.

**/
void Segmentation::segmentation_Km(const cv::Mat& src, cv::Mat& col_mask, cv::Mat& bin_mask, const vector<array<int, 4>>& bb_vector, const std::vector<int>& class_labels)
{
	Mat gaus_blurred, src_ycc, labels;
	col_mask = Mat::zeros(src.rows, src.cols, CV_8UC3);
	Mat out_mask = Mat::zeros(src.rows, src.cols, CV_8UC1);
	GaussianBlur(src, gaus_blurred, Size(7, 7), 0);
	cvtColor(gaus_blurred, src_ycc, COLOR_BGR2YCrCb, 0);
	
	//each iteration work on a single bounding box
	for (int l = 0; l < bb_vector.size(); l++) {

		int x = bb_vector[l][0];
		int y = bb_vector[l][1];
		int w = bb_vector[l][2];
		int h = bb_vector[l][3];
		
		//create a Mat object that contains the datapoints used by K-means
		Mat roi(src_ycc(Rect(x, y, w, h)));
		float center_x = float(roi.rows) / 2;
		float center_y = float(roi.cols) / 2;
		Mat points = Mat::zeros(roi.cols * roi.rows, 3, CV_32F);
		for (int i = 0; i < roi.rows; i++) {
			for (int j = 0; j < roi.cols; j++) {
				// takes into account the euclidean distance between a pixel and the centre (divided to adjust relative weight)
				points.at<float>((i * roi.cols + j), 0) = sqrt((center_x - i) * (center_x - i) + (center_y - j) * (center_y - j)) / 5;
				points.at<float>((i * roi.cols + j), 1) = roi.at<Vec3b>(i, j)[1];
				points.at<float>((i * roi.cols + j), 2) = roi.at<Vec3b>(i, j)[2];
			}
		}

		//run the K-means algorithm
		int K = 2;
		kmeans(points, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 20, 0.5), 10, KMEANS_RANDOM_CENTERS);
		
		//use the labels computed by K-means to draw the output masks
		roi = col_mask(Rect(x, y, w, h));
		Mat roi_mask = out_mask(Rect(x, y, w, h));
		for (int i = 0; i < roi.rows; i++) {
			for (int j = 0; j < roi.cols; j++) {
				int label = labels.at<int>(i * roi.cols + j);
				if (label == labels.at<int>((roi.rows / 2 * roi.cols) + (roi.cols / 2))) {
					colors[class_labels[l]] * 0.3;
					roi_mask.at<unsigned char>(i, j) = 255;
				}
			}
		}
	}
	bin_mask = out_mask;
}




/* @brief Method that performs hand segmentation on the provided image, using GrabCut algorithm.

* The GrabCut algorithm is performed on the source image in the YCrCb color space, considering a single bounding box at a time:
* all the pixel outside the considered bounding box are labeled as certain background.
* The results of the segmentation are stored into a colored mask (a specific color for each hand) and a into binary mask (white for hand, black for not hand).
* The color for each hand is chosen according to the class label associated with the considered bounding box:
* - purple for my_left (0);
* - yellow for my_right (1);
* - red for your_left (2);
* - green for your_right (3);

@param src input image where to perform segmentation
@param col_mask output colored image with black background that contains the mask computed for each hand (with different colors).
@param bin_mask output grayscale image that store the final binary mask of the segmentation with classes hand / not hand.
@param bound_boxes vector of arrays containing the coordinates of the bounding boxes.
@param class_labels a vector of int, each element containing the label associated with the corresponding bounding box in bb_vector.

**/
void Segmentation::segmentation_GB(const cv::Mat& src, cv::Mat& col_mask, cv::Mat& bin_mask,const  vector<array<int, 4>>& bb_vector, const std::vector<int>& class_labels)
{
	Mat src_ycc;
	cvtColor(src, src_ycc, COLOR_BGR2YCrCb, 0);
	Mat out_mask = Mat::zeros(src.rows, src.cols, CV_8UC1);
	col_mask = Mat::zeros(src.rows, src.cols, CV_8UC3);
	
	//each iteration work on a single bounding box
	for (int k = 0; k < bb_vector.size(); k++) {
		int x = bb_vector[k][0];
		int y = bb_vector[k][1];
		int w = bb_vector[k][2];
		int h = bb_vector[k][3];

		Mat  bg1, fg1;
		Mat mask = Mat();

		// run GrabCut algorithm on the source image considering the k-th bounding box; pixels outside the bounding box are considered background
		grabCut(src_ycc, mask, Rect(x, y, w, h), bg1, fg1, 1, GC_INIT_WITH_RECT);

		//use the labels computed by GrabCut to draw the output masks of the k-th hand.
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (mask.at<unsigned char>(i, j) == GC_PR_FGD || mask.at<unsigned char>(i, j) == GC_FGD) {
					col_mask.at<Vec3b>(i, j) = colors[class_labels[k]] * 0.3;
					out_mask.at<unsigned char>(i, j) = 255;
				}
			}
		}
	}
	bin_mask = out_mask;
}


/* @brief Method that performs hand segmentation on the provided image, using GrabCut algorithm with a given initial mask for each hand.

* The GrabCut algorithm is performed on the source image in the YCrCb color space, considering a single bounding box at a time:
* all the pixel outside the considered bounding box are labeled as certain background.
* The algorithm starts from a given initial binary mask of each hand: white pixels are considered as probable foreground (hand), black pixels as probable background (not hand). 


@param src input image where to perform segmentation
@param col_mask output colored image with black background that contains the mask computed for each hand (with different colors).
@param mask output colored image with black background that contains the mask computed for each hand (with different colors).
@param mask_vec vector of bynary images used by GrabCut as initial mask for each hand segmentation.
@param bound_boxes vector of arrays containing the coordinates of the bounding boxes.
@param class_labels a vector of int, each element containing the label associated with the corresponding bounding box in bb_vector.
**/
void Segmentation::segmentation_GB_mask(const cv::Mat& src, cv::Mat& col_mask, cv::Mat& mask, const std::vector<cv::Mat>& mask_vec,const  std::vector<std::array<int, 4>>& bound_boxes, const std::vector<int>& class_labels)
{
	Mat src_ycc;
	Mat out_mask = Mat::zeros(src.rows, src.cols, CV_8UC1);
	col_mask = Mat::zeros(src.rows, src.cols, CV_8UC3);
	cvtColor(src, src_ycc, COLOR_BGR2YCrCb, 0);
	
	
	//each iteration work on a single bounding box
	for (int k = 0; k < bound_boxes.size(); k++) {

		int x = bound_boxes[k][0];
		int y = bound_boxes[k][1];
		int w = bound_boxes[k][2];
		int h = bound_boxes[k][3];

		Mat  bg, fg;
		Mat label_mask = Mat(src.rows, src.cols, CV_8UC1, GC_BGD);
		Mat mask_roi = mask_vec[k].clone();
		Mat label_roi(label_mask(Rect(x, y, w, h)));

		//Convert the provided initial binary mask of the k-th hand on a mask with labels used by grabCut function:
		//white becomes probable foreground; black becomes probable background
		for (int i = 0; i < mask_roi.rows; i++) {
			for (int j = 0; j < mask_roi.cols; j++) {
				if (mask_roi.at<unsigned char>(i, j) == 255) {
					label_roi.at<unsigned char>(i, j) = GC_PR_FGD;
				}
				else label_roi.at<unsigned char>(i, j) = GC_PR_BGD;
			}
		}
		// run the grabCut algorithm
		grabCut(src_ycc, label_mask, Rect(x, y, w, h), bg, fg, 2, GC_INIT_WITH_MASK);

		//use the labels computed by GrabCut to draw the output masks of the k-th hand.
		bool segmentation_present = false;
		for (int i = 0; i < col_mask.rows; i++) {
			for (int j = 0; j < col_mask.cols; j++) {
				if (label_mask.at<unsigned char>(i, j) == GC_PR_FGD || label_mask.at<unsigned char>(i, j) == GC_FGD) {
					segmentation_present = true;
					out_mask.at<unsigned char>(i, j) = 255;
					col_mask.at<Vec3b>(i, j) = colors[class_labels[k]] * 0.3;				
				}
			}
		}
		
		//if Grabcut did not label any pixel of a small BB as foreground use the provided initial mask as output mask
		if (!segmentation_present && ( (h*w)<(src.rows*src.cols/5) )  ) {
			Mat dst_roi(col_mask(Rect(x, y, w, h)));
			Mat out_mask_roi(out_mask(Rect(x, y, w, h)));

			for (int i = 0; i < dst_roi.rows; i++) {
				for (int j = 0; j < dst_roi.cols; j++) {
					if (mask_roi.at<unsigned char>(i, j) == 255) {
						dst_roi.at<Vec3b>(i, j) = colors[class_labels[k]] * 0.3;
						out_mask_roi.at<unsigned char>(i, j) = 255;
					}
				}
			}
		}
	}
	mask = out_mask;


}


/* @brief Compute difference image from estimated skin color for each bounding box.
For each specified bounding box this method computes the color difference image between the pixels of the src image that are inside that box 
and the value of the approximated skin color of the hand contained in the box.
The skin color is approximated by considering the color value of the central pixel of each bounding box. 

@param src input image
@param dst output image
@param bound_boxes vector of arrays containing the coordinates of the bounding boxes

**/
void Segmentation::difference_from_center_hand(const cv::Mat& src, std::vector<Mat>& difference_bb_vec, const std::vector<std::array<int, 4>>& bound_boxes)
{
	Mat difference, src_ycc;
	cvtColor(src, src_ycc, COLOR_BGR2YCrCb, 0);
	
	//each iteration work on a single bounding box
	for (int k = 0; k < bound_boxes.size(); k++) {

		int x = bound_boxes[k][0];
		int y = bound_boxes[k][1];
		int w = bound_boxes[k][2];
		int h = bound_boxes[k][3];
		Mat roi(src_ycc(Rect(x, y, w, h)));

		//consider the value of the central pixel
		Vec3b center_val = roi.at<Vec3b>(h / 2, w / 2);
		
		//compute difference from center value for each pixel inside the bounding box
		Mat difference_bb(h, w, CV_8U);
		for (int i = 0; i < roi.rows; i++) {
			for (int j = 0; j < roi.cols; j++) {
				float cvd[2];
				cvd[0] = abs(center_val[1] - roi.at<Vec3b>(i, j)[1]);
				cvd[1] = abs(center_val[2] - roi.at<Vec3b>(i, j)[2]);
				difference_bb.at<unsigned char>(i, j) = (cvd[0] + cvd[1] ) / 2;
			}
		}
		difference_bb_vec.push_back(difference_bb);
		//show_image(difference_bb*5, to_string(k));
	}
}



/* @brief Compute difference image from estimated skin color for each bounding box, exploiting classification data.

* For each specified bounding box this method computes the color difference image between the pixels of the src image that are inside that box
* and the value of the skin color of the hand contained in the box.
* The skin color is computed by considering the average color value between the central pixel of each bounding box and a second pixel 
* collocated in a specific position that depends on the label associated with the bounding box. 

@param src input image
@param difference_bb_vec vector that, after the call, stores the difference images computed for each bounding box.
@param bound_boxes vector of arrays containing the coordinates of the bounding boxes
@param class_labels a vector of int, with the same size of bound_boxes, containing the class label associated with each bounding box.

**/
void Segmentation::difference_from_center_hand_label(const cv::Mat& src, std::vector<cv::Mat>& difference_bb_vec, const std::vector<std::array<int, 4>>& bound_boxes, const std::vector<int>& class_labels)
{
	Mat difference, src_ycc;
	cvtColor(src, src_ycc, COLOR_BGR2YCrCb, 0);
	
	//each iteration work on a single bounding box
	for (int k = 0; k < bound_boxes.size(); k++) {

		int x = bound_boxes[k][0];
		int y = bound_boxes[k][1];
		int w = bound_boxes[k][2];
		int h = bound_boxes[k][3];
		int label = class_labels[k];
		Mat roi(src_ycc(Rect(x, y, w, h)));

		//consider the value of the central pixel
		Vec3b center_val = roi.at<Vec3b>(h / 2, w / 2);

		//consider the value of the second pixel according to label
		Vec3b decenter_val = roi.at<Vec3b>(h / 2, w / 2);
		if (label == 0) decenter_val = roi.at<Vec3b>(h * 2 / 3, w / 3);
		if (label == 1) decenter_val = roi.at<Vec3b>(h * 2 / 3, w * 2 / 3);
		if (label == 2) decenter_val = roi.at<Vec3b>(h / 3, w * 2 / 3);
		if (label == 3) decenter_val = roi.at<Vec3b>(h / 3, w / 3);

		//compute average value
		Vec3b skin_color;
		for (int i = 0; i < 3; i++) skin_color[i] = (center_val[i] + decenter_val[i]) / 2;
		
		//compute difference from average value for each pixel inside the bounding box
		Mat difference_bb(h, w, CV_8U);
		for (int i = 0; i < roi.rows; i++) {
			for (int j = 0; j < roi.cols; j++) {
				float cvd[2];
				cvd[0] = abs(skin_color[1] - roi.at<Vec3b>(i, j)[1]);
				cvd[1] = abs(skin_color[2] - roi.at<Vec3b>(i, j)[2]);
				difference_bb.at<unsigned char>(i, j) = (cvd[0] + cvd[1] ) / 2;
			}
		}
		difference_bb_vec.push_back(difference_bb);
	}

}



/* @brief Apply a threshold to a set of difference image.
* This method apply a threshold on each grayscale image contained into difference_bb_vec and save the binary image obtained into the corresponding position of threshold_bb_vec.
* The threshold value is computed separately for each difference image, exploiting the OTSU method.

@param difference_bb_vec vector that stores the difference images to be thresholded
@param threshold_bb_vec vector that, after the call, stores the images obtained after the threshold of each difference image
@param bound_boxes vector of arrays containing the cordinates of the bounding boxes

**/
void Segmentation::treshold_difference(const std::vector<cv::Mat>& difference_bb_vec, std::vector<cv::Mat>& threshold_bb_vec)
{
	//each iteration work on a single difference image
	for (int k = 0; k < difference_bb_vec.size(); k++) {

		Mat roi = difference_bb_vec[k].clone();
		Mat thresholded_roi;
		threshold(roi, thresholded_roi, 1, 255, THRESH_BINARY_INV + THRESH_OTSU);
		threshold_bb_vec.push_back(thresholded_roi);
	}
}


/*  @brief evaluates a segmentation using the pixel accuracy metric. 
This method evaluates the specified segmentation mask using the pixel accuracy metric. 

@param mask segmentation mask that needs to be evaulated
@param ground_th ground truth mask for the segmentation

**/
float Segmentation::compute_pixel_accuracy(const cv::Mat& mask, const cv::Mat& ground_th)
{
	float correctly_classified = 0;
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask.at<unsigned char>(i, j) == ground_th.at<unsigned char>(i, j)) { correctly_classified++; }
		}
	}
	float accuracy = correctly_classified / (mask.rows * mask.cols);
	return accuracy;
}


/*  @brief evaluates a segmentation using the Intersection Over Union metric.
This method evaluates the specified segmentation mask using the Intersection Over Union metric..

@param mask segmentation mask that needs to be evaulated
@param ground_th ground truth mask for the segmentation

**/
float Segmentation::compute_IOU(const cv::Mat& mask, const cv::Mat& ground_th)
{
	float mask_hand = 0;
	float mask_not_hand = 0;
	float gt_hand = 0;
	float gt_not_hand = 0;
	float intersection_hand = 0;
	float intersection_not_hand = 0;
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {

			if (mask.at<unsigned char>(i, j) == 255) { mask_hand++; }
			if (mask.at<unsigned char>(i, j) == 0) { mask_not_hand++; }
			if (ground_th.at<unsigned char>(i, j) == 255) { gt_hand++; }
			if (ground_th.at<unsigned char>(i, j) == 0) { gt_not_hand++; }
			if (mask.at<unsigned char>(i, j) == 255 && mask.at<unsigned char>(i, j) == ground_th.at<unsigned char>(i, j)) { intersection_hand++; }
			if (mask.at<unsigned char>(i, j) == 0 && mask.at<unsigned char>(i, j) == ground_th.at<unsigned char>(i, j)) { intersection_not_hand++; }
		}
	}
	float union_hand = (mask_hand + gt_hand) - intersection_hand;
	float union_not_hand = (mask_not_hand + gt_not_hand) - intersection_not_hand;
	float IOU = ((intersection_hand / union_hand) + (intersection_not_hand / union_not_hand)) / 2;
	
	return IOU;
}



/*  @brief Performs segmentation of the given image.
This method calls, in a specific sequence, all the methods of the Segmentation class needed to perform hand segmentation on the given source image.
In particular it exploits the GrabCut algorithm with an initial mask obtained from difference-from-skin computation using classification labels.
For the segmentation process only the portions of the source image inside the given bounding boxes will be considered.
This method also save the obtained binary mask image into a specific path.

@param src input image where to perform segmentation; after the call it stores the segmented image
@param bound_boxes vector of arrays containing the coordinates of the bounding boxes
@param class_labels a vector of int, with the same size of bound_boxes, containing the class label associated with each bounding box.

**/
void Segmentation::make_segmentation(cv::Mat& src, const std::vector<std::array<int, 4>>& bound_boxes, const std::vector<int>& class_labels)
{
	vector<Mat> difference_bb_vec;
	vector<Mat> treshold_bb_vec;
	Mat  bin_mask, col_mask;

	if (!valid_bb_cordinates(src.rows, src.cols, bound_boxes)) {
		cerr << "Error! the bounding box cordinates provided are not consistent with source image size";
		exit(1);
	}

	difference_from_center_hand_label(src, difference_bb_vec, bound_boxes, class_labels);
	treshold_difference(difference_bb_vec, treshold_bb_vec);
	segmentation_GB_mask(src, col_mask, bin_mask, treshold_bb_vec, bound_boxes, class_labels);
	addWeighted(src, 1, col_mask, 0.8, 0.0, src);
	imwrite("./output/bin_mask.png", bin_mask);
}

/*  @brief Performs segmentation of the given image, and evaluate its performance.
This method calls in sequence all the necessary methods to perform the segmentation of the give image.
In particular it exploits the GrabCut algorithm with an initial mask obtained from difference-from-skin computation using classification labels.
For the segmentation process only the portions of the source image inside the given bounding boxes will be considered.
This method also save the obtained binary mask image into a specific path.
It also evaluates the segmentation, by computing the pixel accuracy and IoU metric w.r.t. the given ground truth mask.

@param src input image where to perform segmentation; after the call it stores the segmented image
@param bound_boxes vector of arrays containing the cordinates of the bounding boxes
@param class_labels a vector of int, with the same size of bound_boxes, containing the class label associated with each bounding box.
@param gt_mask_path ground truth mask for the segmentation

**/
void Segmentation::make_segmentation(cv::Mat& src, const std::vector<std::array<int, 4>>& bound_boxes, const std::vector<int>& class_labels, std::string gt_mask_path)
{
	vector<Mat> difference_bb_vec;
	vector<Mat> treshold_bb_vec;
	Mat  bin_mask, col_mask;
	Mat gt_mask = imread(gt_mask_path);

	if (gt_mask.empty()) {
		cerr << "Error! ground truth mask image is empty\n";
	}

	if (!valid_bb_cordinates(src.rows, src.cols, bound_boxes)) {
		cerr << "Error! the bounding box cordinates provided are not consistent with source image size";
		exit(1);
	}

	cvtColor(gt_mask, gt_mask, COLOR_BGR2GRAY);

	difference_from_center_hand_label(src, difference_bb_vec, bound_boxes, class_labels);
	treshold_difference(difference_bb_vec, treshold_bb_vec);
	segmentation_GB_mask(src, col_mask, bin_mask, treshold_bb_vec, bound_boxes, class_labels);
	addWeighted(src, 1, col_mask, 0.8, 0.0, src);
	imwrite("./output/bin_mask.png", bin_mask);

	float pixel_accuracy = compute_pixel_accuracy(bin_mask, gt_mask);
	float IOU = compute_IOU(bin_mask, gt_mask);
	cout << "PA= " << pixel_accuracy << ";	IOU= " << IOU << "\n";

}
