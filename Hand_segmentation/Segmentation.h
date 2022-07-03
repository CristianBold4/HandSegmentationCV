#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


class Segmentation {
private:
	
	cv::Vec3b colors[4] = { cv::Vec3b(255, 0, 0),cv::Vec3b(0, 255, 0),cv::Vec3b(0, 0, 255),cv::Vec3b(255, 117, 20) };

public:
	
	std::vector<std::array<int, 4>>  read_bb_file(std::string path);
	void crop(cv::Mat src, cv::Mat& dst, std::array<int, 4>);
	void  draw_segmentation_Km(cv::Mat src, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	
	void draw_segmentation_GB(cv::Mat src, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	void draw_box_image(cv::Mat src, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	
	void difference_from_center(cv::Mat src, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	
};
