#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


class Segmentation {
private:
	
	std::array<cv::Vec3b,4> colors = { cv::Vec3b(127, 0, 0),cv::Vec3b(0, 127, 0),cv::Vec3b(0, 0, 127),cv::Vec3b(124,53, 10) };
	cv::Vec<float,3> trasparency_colors[4] = { cv::Vec<float,3>(2,0.8, 0.8),cv::Vec<float,3>(0.8, 2, 0.8),cv::Vec<float,3>(0.8, 0.8, 2),cv::Vec<float,3>(2, 0.8, 2) };

public:
	
	void show_image(cv::Mat to_show, std::string window_name);
	std::vector<std::array<int, 4>>  read_bb_file(std::string path);
	void draw_box_image(cv::Mat src, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	void apply_mask(cv::Mat src, cv::Mat& dst, cv::Mat mask, bool same_color);
	
	void draw_segmentation_Km(cv::Mat src, cv::Mat& dst, cv::Mat& bin_mask, std::vector<std::array<int, 4>> bound_boxes);
	void draw_segmentation_GB(cv::Mat src, cv::Mat& dst, cv::Mat& bin_mask, std::vector<std::array<int, 4>> bound_boxes);
	void draw_segmentation_GB_mask(cv::Mat src, cv::Mat& dst, cv::Mat& mask, std::vector<std::array<int, 4>> bound_boxes);
	
	void difference_from_center_hand(cv::Mat src, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	void difference_from_center(cv::Mat src, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	void treshold_difference(cv::Mat difference, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	float compute_pixel_accuracy(cv::Mat mask, cv::Mat ground_th);
	
	void get_superpixel_image(cv::Mat src, cv::Mat& dst);
};
