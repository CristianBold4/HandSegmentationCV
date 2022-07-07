#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


class Segmentation {
private:
	
	cv::Vec3b colors[4] = { cv::Vec3b(255, 0, 0),cv::Vec3b(0, 255, 0),cv::Vec3b(0, 0, 255),cv::Vec3b(255, 117, 20) };
	cv::Vec3b trasparency_colors[4] = { cv::Vec3b(1, 0, 0),cv::Vec3b(0, 1, 0),cv::Vec3b(0, 0, 1),cv::Vec3b(1, 0, 1) };

public:
	
	std::vector<std::array<int, 4>>  read_bb_file(std::string path);
	void draw_box_image(cv::Mat src, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	void apply_mask(cv::Mat src, cv::Mat& dst, cv::Mat& mask);
	
	void draw_segmentation_Km(cv::Mat src, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	void draw_segmentation_GB(cv::Mat src, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	void draw_segmentation_GB_mask(cv::Mat src, cv::Mat& dst, cv::Mat& mask, std::vector<std::array<int, 4>> bound_boxes);
	
	
	void difference_from_center(cv::Mat src, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	void treshold_difference(cv::Mat difference, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	float compute_pixel_accuracy(cv::Mat mask, cv::Mat ground_th);
	
	
};
