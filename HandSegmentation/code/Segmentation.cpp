#include "Segmentation.h"
#include <iostream>
#include <fstream>
#include <opencv2/intensity_transform.hpp>

using namespace cv;
using namespace std;


/*
method that parse the txt file that contains the bounding boxes cordinates and return an array containing such cordinates
The cordinates are expressed as : [ (top-lef corner x cordinate), (top-lef corner y cordinate), (width), (heigth) ]

@param path path of the txt file that contains the bounding boxes cordinates.
@return a vector of array, each array containing the cordinates of a single bounding boxe.
**/
vector<array<int, 4>> Segmentation::read_bb_file(string path)
{
	String line;
	vector<String> line_vec;
	vector<array<int, 4>> boxes;
	ifstream bb_file(path);

	while (getline(bb_file, line)) {

		//cout << line << "\n";
		line_vec.push_back(line);
	}
	for (int i = 0; i < line_vec.size(); i++) {
		String parsed, iline;
		iline = line_vec[i];
		stringstream stringstream(iline);
		int cordinate_counter = 0;
		array<int, 4> cordinates;
		while (getline(stringstream, parsed, '	')) {

			if (cordinate_counter == 4) {
				cout << "unable to read bounding box file: more that 4 cordinates";
				exit(1);
			}

			int c = stoi(parsed);
			//cout << i << " : " << c << "\n";
			cordinates[cordinate_counter] = c;
			cordinate_counter++;
		}
		if (cordinate_counter < 4) {
			cout << "unable to read bounding box file: less that 4 cordinates " << cordinate_counter;
			exit(1);
		}
		boxes.push_back(cordinates);
	}
	return boxes;
}


/*
method that draw the specified bounding boxes into the image provided by src and save the result into dst

@param src input image
@param dst output image
@param bound_boxes vector of arrays containing the cordinates of the bounding boxes

**/
void Segmentation::draw_box_image(cv::Mat src, cv::Mat& dst, vector<array<int, 4>> bound_boxes)
{
	dst = src.clone();

	for (int k = 0; k < bound_boxes.size(); k++) {
		int x = bound_boxes[k][0];
		int y = bound_boxes[k][1];
		int w = bound_boxes[k][2];
		int h = bound_boxes[k][3];
		rectangle(dst, Point(x, y), Point(x + w, y + h), colors[k] * 0.6, 1, 1);
	}
}

/*
method that draw the specified mask (in red) into the image provided by src and save the result into dst

@param src input image
@param dst output image
@param mask the mask to be drawn

**/
void Segmentation::apply_mask(cv::Mat src, cv::Mat& dst, cv::Mat& mask)
{
	dst = src.clone();

	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {

			if (mask.at<unsigned char>(i, j) == 255) {
				multiply(dst.at<Vec3b>(i, j), trasparency_colors[2], dst.at<Vec3b>(i, j));
			}
		}
	}
}

/*
method that draw into dst the result of the segmentation perfomed on src, using K-means clustering.
The K-means algorithm is performed in a vector space where each vector contains the Cb and Cr channel value of each pixel (+ altro da decidere).

@param src input image
@param dst output image
@param bound_boxes vector of arrays containing the cordinates of the bounding boxes

**/
void Segmentation::draw_segmentation_Km(cv::Mat src, cv::Mat& dst, vector<array<int, 4>> bound_boxes)
{
	Mat gaus_blurred, src_ycc, out, labels, bil;
	dst = src.clone();
	GaussianBlur(src, gaus_blurred, Size(7, 7), 0);
	cvtColor(gaus_blurred, src_ycc, COLOR_BGR2YCrCb, 0);
	/*
	bilateralFilter(src, bil, -1, 50, 10);
	namedWindow("bilateral", WINDOW_AUTOSIZE);
	imshow("bilateral", bil);
	waitKey(0);
	*/

	for (int l = 0; l < bound_boxes.size(); l++) {

		int x = bound_boxes[l][0];
		int y = bound_boxes[l][1];
		int w = bound_boxes[l][2];
		int h = bound_boxes[l][3];

		Mat roi(src_ycc(Rect(x, y, w, h)));
		Mat points = Mat::zeros(roi.cols * roi.rows, 3, CV_32F);
		float center_x = float(roi.rows) / 2;
		float center_y = float(roi.cols) / 2;
		for (int i = 0; i < roi.rows; i++) {
			for (int j = 0; j < roi.cols; j++) {
				//the next line takes into account the euclidean distance between a pixel and the centre (divided to adjust weight)
				points.at<float>((i * roi.cols + j), 0) = sqrt((center_x - i) * (center_x - i) + (center_y - j) * (center_y - j)) / 3;
				points.at<float>((i * roi.cols + j), 1) = roi.at<Vec3b>(i, j)[1];
				points.at<float>((i * roi.cols + j), 2) = roi.at<Vec3b>(i, j)[2];
			}
		}

		int K = 2;
		kmeans(points, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 20, 0.5), 10, KMEANS_RANDOM_CENTERS);

		roi = dst(Rect(x, y, w, h));
		for (int i = 0; i < roi.rows; i++) {
			for (int j = 0; j < roi.cols; j++) {
				int label = labels.at<int>(i * roi.cols + j);
				if (label == labels.at<int>((roi.rows / 2 * roi.cols) + (roi.cols / 2))) {
					roi.at<Vec3b>(i, j) = colors[l];
				}

			}
		}
	}


}




/*
method that draw into dst the result of the segmentation perfomed on src, using GrabCut algorithm.
The GrabCut algorithm is performed considering the source image in the YCrCb color space.

@param src input image
@param dst output image
@param bound_boxes vector of arrays containing the cordinates of the bounding boxes

**/
void Segmentation::draw_segmentation_GB(cv::Mat src, cv::Mat& dst, vector<array<int, 4>> bound_boxes)
{
	Mat src_ycc, gaus_blurred;

	//GaussianBlur(src, gaus_blurred, Size(7, 7), 0);

	dst = src.clone();
	cvtColor(src, src_ycc, COLOR_BGR2YCrCb, 0);
	for (int k = 0; k < bound_boxes.size(); k++) {
		int x = bound_boxes[k][0];
		int y = bound_boxes[k][1];
		int w = bound_boxes[k][2];
		int h = bound_boxes[k][3];

		Mat  bg1, fg1;
		Mat mask = Mat(src.rows, src.cols, CV_8UC1, GC_BGD);
		Mat roi(mask(Rect(x, y, w, h)));
		roi.setTo(Scalar(GC_PR_FGD));

		//i due if servono per flaggare alcune parti della BB come background probabile/certo ma non so se è corretto.
		/*
		if (k == 1) {
			int dimBG = 10;
			Mat corr(mask(Rect(x + w - dimBG, y, dimBG, dimBG)));
			corr.setTo(Scalar(GC_BGD));
			corr= mask(Rect(x , y + h - dimBG, dimBG, dimBG));
			corr.setTo(Scalar(GC_BGD));

			//namedWindow("cl", WINDOW_AUTOSIZE);
			//imshow("cl", mask * 50);
			//waitKey(0);
		}
		if (k == 2) {

			int dimBG = 10;
			Mat corr(mask(Rect(x , y, dimBG, dimBG)));
			corr.setTo(Scalar(GC_BGD));
			corr = mask(Rect(x + w - dimBG, y + h - dimBG, dimBG, dimBG));
			corr.setTo(Scalar(GC_BGD));
		}*/

		// nota: leggermente più preciso aumentando iterazioni ma molto più lento
		grabCut(src_ycc, mask, Rect(x, y, w, h), bg1, fg1, 1, GC_INIT_WITH_MASK);

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (mask.at<unsigned char>(i, j) == GC_PR_FGD || mask.at<unsigned char>(i, j) == GC_FGD) {
					dst.at<Vec3b>(i, j) = colors[k];
				}

			}
		}
	}

}

/*
method that draw into dst the result of the segmentation perfomed on src, using GrabCut algorithm starting from a given initial mask.
The pixel of the mask with value 255 are considered to be probable foreground pixels; other pixels are considered probable background.
The GrabCut algorithm is performed considering the source image in the YCrCb color space.

@param src input image
@param dst output image
@mask a grayscale image to be used as initial mask for the Grabcut algorithm. After the segmentation it also store the final binary mask of the segmentation with classes hand / not hand
@param bound_boxes vector of arrays containing the cordinates of the bounding boxes

**/
void Segmentation::draw_segmentation_GB_mask(cv::Mat src, cv::Mat& dst, cv::Mat& mask, std::vector<std::array<int, 4>> bound_boxes)
{
	
	Mat src_ycc, gaus_blurred;
	dst = src.clone();
	cvtColor(src, src_ycc, COLOR_BGR2YCrCb, 0);
	Mat out_mask = Mat::zeros(mask.rows, mask.cols, CV_8UC1);

	for (int k = 0; k < bound_boxes.size(); k++) {

		int x = bound_boxes[k][0];
		int y = bound_boxes[k][1];
		int w = bound_boxes[k][2];
		int h = bound_boxes[k][3];

		Mat  bg1, fg1;
		Mat label_mask = Mat(mask.rows, mask.cols, CV_8UC1, GC_BGD);
		Mat roi(mask(Rect(x, y, w, h)));
		Mat label_roi(label_mask(Rect(x, y, w, h)));

		for (int i = 0; i < roi.rows; i++) {
			for (int j = 0; j < roi.cols; j++) {

				if (roi.at<unsigned char>(i, j) == 255) {
					label_roi.at<unsigned char>(i, j) = GC_PR_FGD;
				}
				else label_roi.at<unsigned char>(i, j) = GC_PR_BGD;

			}
		}

		// nota: leggermente più preciso aumentando iterazioni ma molto più lento
		grabCut(src_ycc, label_mask, Rect(x, y, w, h), bg1, fg1, 1, GC_INIT_WITH_MASK);

		
		for (int i = 0; i < dst.rows; i++) {
			for (int j = 0; j < dst.cols; j++) {
				if (label_mask.at<unsigned char>(i, j) == GC_PR_FGD || label_mask.at<unsigned char>(i, j) == GC_FGD) {
					 multiply( dst.at<Vec3b>(i, j), trasparency_colors[k], dst.at<Vec3b>(i, j));
					out_mask.at<unsigned char>(i, j) = 255;
				}
			}
		}
	}
	mask = out_mask;
}



/*
method that compute the difference between each pixel of the src image and the approssimated value of the skin color and saves it into the dst image.
The skin color is computed by considering the average color of the central pixel value of each bounding box.

@param src input image
@param dst output image
@param bound_boxes vector of arrays containing the cordinates of the bounding boxes

**/
void Segmentation::difference_from_center(cv::Mat src, cv::Mat& dst, vector<array<int, 4>> bound_boxes)
{
	Mat averaged, difference, src_ycc;
	difference = src.clone();
	blur(src, averaged, Size(21,21));
	cvtColor(src, src_ycc, COLOR_BGR2YCrCb, 0);
	cvtColor(averaged, averaged, COLOR_BGR2YCrCb, 0);

	vector <Vec3b> center_value;
	array <float, 3> channel_sum = { 0.0,0.0,0.0 };
	Vec3b center_value_mean;

	for (int k = 0; k < bound_boxes.size(); k++) {

		int x = bound_boxes[k][0];
		int y = bound_boxes[k][1];
		int w = bound_boxes[k][2];
		int h = bound_boxes[k][3];

		/*
		namedWindow("cl", WINDOW_AUTOSIZE);
		imshow("cl", averaged);
		waitKey(0);
		*/

		Mat roi(averaged(Rect(x, y, w, h)));
		center_value.push_back(roi.at<Vec3b>(h / 2, w / 2));
	}

	for (int k = 0; k < bound_boxes.size(); k++) {
		channel_sum[0] += center_value[k][0];
		channel_sum[1] += center_value[k][1];
		channel_sum[2] += center_value[k][2];
	}
	center_value_mean[0] = uchar(channel_sum[0] / bound_boxes.size());
	center_value_mean[1] = uchar(channel_sum[1] / bound_boxes.size());
	center_value_mean[2] = uchar(channel_sum[2] / bound_boxes.size());

	for (int i = 0; i < src_ycc.rows; i++) {
		for (int j = 0; j < src_ycc.cols; j++) {
			difference.at<Vec3b>(i, j)[0] = 0; //abs(center_value_mean[0] - src.at<Vec3b>(i, j)[0]);
			difference.at<Vec3b>(i, j)[1] = abs(center_value_mean[1] - src_ycc.at<Vec3b>(i, j)[1]);
			difference.at<Vec3b>(i, j)[2] =  abs(center_value_mean[2] - src_ycc.at<Vec3b>(i, j)[2]);
		}
	}
	
	cvtColor(difference, difference, COLOR_YCrCb2BGR, 0);
	cvtColor(difference, difference, COLOR_BGR2GRAY, 0);
	//intensity_transform::logTransform(difference, difference);
	GaussianBlur(difference, difference, Size(21,21), 0);
	dst = difference;
}


/*
method that perform a treshold of a given image considering only the specified boxes. 
The treshold is performed separately inside each boxes, exploiting the OTSU method.


@param difference input image
@param dst output image
@param bound_boxes vector of arrays containing the cordinates of the bounding boxes

**/
void Segmentation::treshold_difference(cv::Mat difference, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes)
{
	dst = difference.clone();
	vector<Mat> thresholded_boxes;
	for (int k = 0; k < bound_boxes.size(); k++) {

		int x = bound_boxes[k][0];
		int y = bound_boxes[k][1];
		int w = bound_boxes[k][2];
		int h = bound_boxes[k][3];
		/*
		namedWindow("cl", WINDOW_AUTOSIZE);
		imshow("cl", averaged);
		waitKey(0);
		*/
		Mat roi(difference(Rect(x, y, w, h)));
		Mat thresholded_roi;
		double minVal, maxVal;
		minMaxIdx(roi, &minVal, &maxVal);
		
		//two alternatives
		//threshold(roi, thresholded_roi, minVal*1.4, 255, THRESH_BINARY_INV);
		threshold(roi, thresholded_roi, 1, 255, THRESH_BINARY + THRESH_OTSU);
		thresholded_boxes.push_back(thresholded_roi);
	}

	for (int k = 0; k < bound_boxes.size(); k++) {
		//dst(Rect(bound_boxes[k][0], bound_boxes[k][1], bound_boxes[k][2], bound_boxes[k][3])) = thresholded_boxes[k];
		thresholded_boxes[k].copyTo(dst(Rect(bound_boxes[k][0], bound_boxes[k][1], bound_boxes[k][2], bound_boxes[k][3])));
	}
}

/*
method that evaluate a segmentation using the pixel accuracy metric. 

@param mask segmentation mask that needs to be evaulated
@param ground_th ground truth of the segmentation
@param bound_boxes vector of arrays containing the cordinates of the bounding boxes

**/
float Segmentation::compute_pixel_accuracy(cv::Mat mask, cv::Mat ground_th)
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



//codice per calcolo laplacian
/*
Mat kernel = (Mat_<float>(3, 3) <<
	0, 1, 0,
	1, -4, 1,
	0, 1, 0);
Mat imgLaplacian;
filter2D(src, imgLaplacian, CV_32F, kernel);
Mat sharp;
src.convertTo(sharp, CV_32F);
Mat imgResult = sharp - imgLaplacian;
// convert back to 8bits gray scale
imgResult.convertTo(imgResult, CV_8UC3);
namedWindow("sharp", WINDOW_AUTOSIZE);
imshow("sharp", imgResult);
waitKey(0);
*/






