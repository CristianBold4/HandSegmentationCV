//
// Created by Cristian on 16/07/2022.
//

// Include Libraries
#include <iostream>
#include "../header/Detection.h"
#include "../header/Segmentation.h"

// Namespaces
using namespace cv;
using namespace std;
using namespace cv::dnn;

/**
 * TEST METHOD, to compute average IoU and pixel accuracy on the testset
 * @param N_IMAGES
 * @param net_path
 * @author Riccardo
 */

void compute_testset_performance(int N_IMAGES, string net_path){

    float pixel_accuracy_sum=0;
    float IOU_sum=0;
     ofstream out_file("segmentation_test_4.txt");
    out_file<< "TEST 4 - K-means \n\n";
    
    for(int j=1; j<=N_IMAGES; j++){
        cout<< "img "<< j<< "   ";
        out_file<< "img "<< j<< "   ";
        //if(j==16) j++; //salto immagine 16
        Mat frame, frame_copy;
       
        string img_path, det_path,gt_mask_path;

        if (j < 10) {
            img_path = "./input/0" + to_string(j) + ".jpg";
            det_path = "./det/0" + to_string(j) + ".txt";
            gt_mask_path = "./gt_mask/0" + to_string(j) + ".png";
        } else {
            img_path = "./input/" + to_string(j) + ".jpg";
            det_path = "./det/" + to_string(j) + ".txt";
             gt_mask_path = "./gt_mask/" + to_string(j) + ".png";
        }
        
        frame = imread(img_path);
        frame_copy = frame.clone();

        if (frame.empty()) {
            cerr << "Error reading input image!\n";
            return;
        }

        // -- path of the dictionary containing the map id - class
	    string class_list_path = "hands_labels.names";

        string bb_path = "output/out.txt";

        Detection det = Detection(class_list_path, net_path);
	    Segmentation seg;
	    
	     vector<array<int, 4>> bb_coordinates;
         vector<int> classes;
	    
	    // -- DETECTION
         det.make_detection(frame, bb_coordinates, classes);
        
        // -- SEGMENTATION
       
	    vector<Mat> difference_bb_vec;
	    vector<Mat> treshold_bb_vec;
	    Mat  bin_mask, col_mask;
	    Mat gt_mask = imread(gt_mask_path);
	    if (gt_mask.empty()) {
		    cerr << "Error! ground truth mask image is empty\n";
	    }
	    cvtColor(gt_mask, gt_mask, COLOR_BGR2GRAY);

	   
       // if(j==16) { bb_coordinates[1][2] = bb_coordinates[1][2]-1;  } //fix img 16
        // seg.segmentation_Km(frame_copy,col_mask, bin_mask, boxes_vec, class_labels);
        seg.difference_from_center_hand_label(frame_copy, difference_bb_vec, bb_coordinates, classes);
        //seg.difference_from_center_hand(frame_copy, difference_bb_vec, boxes_vec);
        seg.treshold_difference(difference_bb_vec, treshold_bb_vec);
        seg.segmentation_GB_mask(frame_copy, col_mask, bin_mask, treshold_bb_vec, bb_coordinates, classes);
        //seg.segmentation_GB(frame_copy, col_mask, bin_mask, boxes_vec, class_labels);

	    //seg.apply_mask(frame_copy, frame_copy, col_mask, false);
	    
	    float PA = seg.compute_pixel_accuracy(bin_mask, gt_mask);
	    float IOU = seg.compute_IOU(bin_mask, gt_mask);
	    pixel_accuracy_sum += PA;
	    IOU_sum += IOU;
	    cout << "PA= " << PA << ";	IOU= " << IOU << "\n\n";
	    out_file << "PA= " << PA << ";	IOU= " << IOU << "\n\n";
    }
     float pixel_accuracy_avg = pixel_accuracy_sum/(N_IMAGES); 
     float IOU_avg = IOU_sum/(N_IMAGES);
   
    cout << "\nSegmentation:	avg_PA= " << pixel_accuracy_avg << ";	avg_IOU= " << IOU_avg << "\n";
    out_file << "\nSegmentation:	avg_PA= " << pixel_accuracy_avg << ";	avg_IOU= " << IOU_avg << "\n";
    
    out_file.close();
}



int main(int argc, char** argv)
{

    // -- path of the dictionary containing the map id - class
    string class_list_path = "hands_labels.names";
    // -- path of the exported trained model
    string net_path = "aug_model.onnx";
    string bb_path = "output/out.txt";

	// -- load input image
	Mat frame, frame_copy;
	string img_path = argv[1];
	frame = imread(img_path);
	frame_copy = frame.clone();

	if (frame.empty()) {
	    cerr << "Error reading input image!\n";
	    return -1;
	}

	Detection det = Detection(class_list_path, net_path);
	Segmentation seg;

    vector<array<int, 4>> bb_coordinates;
    vector<int> classes;

    // -- if requested performs detection and segmentation and compute performance metrics;
    // -- otherwise only performs detection and segmentation
    if (argc == 4) {

        // -- detection part
        det.make_detection(frame, argv[2], bb_coordinates, classes);
        
        // -- segmentation part
        seg.make_segmentation(frame_copy, bb_coordinates, classes,argv[3]);
    
    } else if (argc == 2) {

        // -- detection part
        det.make_detection(frame, bb_coordinates, classes);
               
        // -- segmentation part
        seg.make_segmentation(frame_copy, bb_coordinates, classes);
    
    } else {
        cerr << "Error! Wrong arguments\n";
    }


	// -- show and save output
	imshow("Detection", frame);
	waitKey(0);
	imwrite("detection.jpg", frame);
	
	imshow("Segmentation", frame_copy);
	waitKey(0);
	imwrite("segmentation.jpg", frame_copy);

	return 0;
}
