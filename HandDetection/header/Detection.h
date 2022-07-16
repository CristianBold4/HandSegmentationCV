//
// Created by Cristian on 16/07/2022.
//

#ifndef YOLOV5_DETECTION_H
#define YOLOV5_DETECTION_H

// Include Libraries.
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

class Detection {

    private:

        float counter = 0;
        int n_det = 0;

    public:

        // Text parameters.
        const float FONT_SCALE = 0.7;
        const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
        const int THICKNESS = 1;

        // Constants.
        const float INPUT_WIDTH = 640.0;
        const float INPUT_HEIGHT = 640.0;
        const float SCORE_THRESHOLD = 0.5;
        const float NMS_THRESHOLD = 0.5;
        const float CONFIDENCE_THRESHOLD = 0.55;

        // Colors.
        cv::Scalar BLACK = cv::Scalar(0,0,0);
        cv::Scalar BLUE = cv::Scalar(255, 178, 50);
        cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
        cv::Scalar RED = cv::Scalar(0,0,255);

        const cv::Scalar MY_LEFT = cv::Scalar(255,0,127); //purple
        const cv::Scalar MY_RIGHT = cv::Scalar(0,255,255); //yellow
        const cv::Scalar YOUR_LEFT = cv::Scalar(0,0,255); //red
        const cv::Scalar YOUR_RIGHT = cv::Scalar(0,204,0); //green

        // labels colors.
        const  cv::Scalar LABELS_COLORS [4] = {MY_LEFT, MY_RIGHT, YOUR_LEFT, YOUR_RIGHT};

        void read_bb_file(std::string path, std::vector<std::array<int, 4>> &bb_vector);

        void draw_label(cv::Mat& input_image, std::string label, int left, int top);

        std::vector<cv::Mat> pre_process(cv::Mat &input_image, cv::dnn::Net &net);

        std::string compute_IoU(std::array<int,4> pred_boxes_vec[4], std::vector<std::array<int,4>> gr_boxes_vec);

        cv::Mat post_process(cv::Mat &input_image, std::vector<cv::Mat> &outputs, const std::vector<std::string> &class_name,
                             std::vector<std::array<int, 4>> gr_boxes_vec, std::string &IoU);

        void write_output(std::array<int,4> ordered_bb [4]);

        void compute_avg_IoU_testset(cv::dnn::Net &net, std::vector<std::string> &class_list);

};


#endif //YOLOV5_DETECTION_H
