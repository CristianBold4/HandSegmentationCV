//
// Created by Riccardo.
//

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


class Segmentation {
private:
	
	std::array<cv::Vec3b,4> colors = { cv::Vec3b(255, 0, 127),cv::Vec3b(0,255, 255),cv::Vec3b(0, 0, 255),cv::Vec3b(0,204, 0) };
	cv::Vec<float,3> trasparency_colors[4] = { cv::Vec<float,3>(2,0.8, 0.8),cv::Vec<float,3>(0.8, 2, 0.8),cv::Vec<float,3>(0.8, 0.8, 2),cv::Vec<float,3>(2, 0.8, 2) };
	bool valid_bb_cordinates(int src_r, int src_c, const std::vector<std::array<int, 4>>& bb_vector);

public:
	
	void read_bb_file_label(int src_r, int src_c, std::string path, std::vector<std::array<int, 4>>& bb_vector, std::vector<int>& class_labels);
	void get_bb_labels(int src_r, int src_c,std::array<int, 4> ordered_bb[4],std::vector<std::array<int, 4>>& bb_vector, std::vector<int>& class_labels);
	std::vector<std::array<int, 4>> get_wide_cordinates(int src_r, int src_c,const std::vector<std::array<int, 4>>& bb_vector);
	void draw_box_image_label(const cv::Mat& src, cv::Mat& dst, const std::vector<std::array<int, 4>>& bb_vector, const std::vector<int>& labels, bool show_color_point);
	//void apply_mask(cv::Mat src, cv::Mat& dst, cv::Mat mask, bool same_color);
	void segmentation_Km(const cv::Mat& src, cv::Mat& dst, cv::Mat& bin_mask,const std::vector<std::array<int, 4>>& bound_boxes, const std::vector<int>& class_labels);
	void segmentation_GB(const cv::Mat& src, cv::Mat& dst, cv::Mat& bin_mask, const std::vector<std::array<int, 4>>& bound_boxes, const std::vector<int>& class_labels);
	void segmentation_GB_mask(const cv::Mat& src, cv::Mat& dst, cv::Mat& mask, const std::vector<cv::Mat>& mask_vec, const std::vector<std::array<int, 4>>& bound_boxes, const std::vector<int>& class_labels);
	void difference_from_center_hand(const cv::Mat& src, std::vector<cv::Mat>& difference_bb_vec, const std::vector<std::array<int, 4>>& bound_boxes);
	void difference_from_center_hand_label(const cv::Mat& src, std::vector<cv::Mat>& difference_bb_vec, const std::vector<std::array<int, 4>>& bound_boxes, const std::vector<int>& class_labels);
	void treshold_difference(const std::vector<cv::Mat>& difference_bb_vec, std::vector<cv::Mat>& treshold_bb_vec);
	float compute_pixel_accuracy(const cv::Mat& mask,const cv::Mat& ground_th);
	float compute_IOU(const cv::Mat& mask, const cv::Mat& ground_th);
	void show_image(const cv::Mat& to_show, const std::string& window_name);
	
	void make_segmentation( cv::Mat& src,const std::vector<std::array<int, 4>>& bound_boxes, const std::vector<int>& class_labels);
	void make_segmentation( cv::Mat& src, const std::vector<std::array<int, 4>>& bound_boxes, const std::vector<int>& class_labels, std::string gt_mask_path);


	//void draw_box_image(cv::Mat src, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	//void  read_bb_file(int src_r, int src_c, std::string path, std::vector<std::array<int, 4>>& bb_vector);
	//void difference_from_center(cv::Mat src, cv::Mat& dst, std::vector<std::array<int, 4>> bound_boxes);
	//void get_superpixel_image(cv::Mat src, cv::Mat& dst);
};
