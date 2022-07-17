//
// Created by Cristian on 16/07/2022.
//

#include "../header/Detection.h"



// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

Detection::Detection(const std::string &class_list_path, const std::string &net_path) {
    // Load class list.
    ifstream ifs(class_list_path);
    string line;


    while (getline(ifs, line)) {
        this->class_list.push_back(line);
    }

    // Load model: read net
    this->net = readNet(net_path);

}

void Detection::read_bb_file(const string &path, vector<array<int, 4>> &bb_vector) {
    String line;
    vector<String> line_vec;
    vector<array<int, 4>> boxes;
    ifstream bb_file(path);

    while (getline(bb_file, line)) {

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

    bb_vector = boxes;
}

// Draw the predicted bounding box.
void Detection::draw_label(Mat &input_image, string label, int left, int top) {
	// Display the label at the top of the bounding box.
	int baseLine;
	Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
	top = max(top, label_size.height);
	// Top left corner.
	Point tlc = Point(left, top);
	// Bottom right corner.
	Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
	// Draw black rectangle.
	rectangle(input_image, tlc, brc, BLACK, FILLED);
	// Put the label on the black rectangle.
	putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

vector<Mat> Detection::pre_process(Mat &input_image) {
	// Convert to blob.
	Mat blob;
	blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

	net.setInput(blob);

	// Forward propagate.
	vector<Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	return outputs;
}

string Detection::compute_IoU(array<int, 4> pred_boxes_vec[4], vector<array<int, 4>> gr_boxes_vec) {

    int left, right, top, bottom, width, height;
    int gr_left, gr_right, gr_top, gr_bottom, gr_width, gr_height;

    // compute IoU

    int c = 0;


    string out;

    for (int i = 0; i < 4; i++) {

        if (pred_boxes_vec[i][0] != -1) { //existing bbox



            left = pred_boxes_vec[i][0];
            top = pred_boxes_vec[i][1];
            width = pred_boxes_vec[i][2];
            height = pred_boxes_vec[i][3];

            gr_left = gr_boxes_vec[c][0];
            gr_top = gr_boxes_vec[c][1];
            gr_width = gr_boxes_vec[c][2];
            gr_height = gr_boxes_vec[c][3];


            right = left + width;
            bottom = top + height;
            gr_right = gr_left + gr_width;
            gr_bottom = gr_top + gr_height;

            cout << "Pred: " << left << " " << top << " " << width << " " << height << endl;
            cout << "Truth: " << gr_left << " " << gr_top << " " << gr_width << " " << gr_height << endl;

            // determine the (x, y)-coordinates of the intersection rectangle
            int xA = max(left, gr_left);
            int yA = max(top, gr_top);
            int xB = min(right, gr_right);
            int yB = min(bottom, gr_bottom);

            // compute intersection
            int inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1);

            // compute area of both bboxes
            int pred_area = (right - left + 1) * (bottom - top + 1);
            int gr_area = (gr_right - gr_left + 1) * (gr_bottom - gr_top + 1);

            float IoU = static_cast<float> (inter_area) / static_cast<float>(pred_area + gr_area - inter_area);

            cout << "IoU: " << IoU << endl;
            out += "IoU: " + to_string(IoU) + "\n";

            c++;

            n_det++;
            counter += IoU;
        }


    }

    return out;
}

void Detection::write_output(array<int, 4> ordered_bb[4]) {

    ofstream outfile("./output/out.txt");

    // write orderd output
    for (int i = 0; i < 4; i++) {
        //write the output with class id
        if (ordered_bb[i][0] != -1) {
            outfile << ordered_bb[i][0] << " " << ordered_bb[i][1] << " " << ordered_bb[i][2] << " " << ordered_bb[i][3]
                    << " " << i << " " << "\n";
        }

    }

    outfile.close();
}

void Detection::post_process(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name,
                             vector<array<int, 4>> gr_boxes_vec, string &IoU, array<int, 4> ordered_bb[4]) {

    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *) outputs[0].data;

    // Network output dimensions
    const int dimensions = 9;
    const int rows = 25200;

    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float *classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD) {
                // Store class ID and confidence in the pre-defined respective vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += 9;
    }

    // Perform Non-Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);


    //array<int, 4> ordered_bb[4];
    // init array
    for (int i = 0; i < 4; i++) {
        ordered_bb[i][0] = -1;
    }


    // -- compute bounding box coordinates
    for (int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;

        //cout << class_ids[idx] << " " << class_name[class_ids[idx]] << endl;

        int id = class_ids[idx];
        ordered_bb[id][0] = left;
        ordered_bb[id][1] = top;
        ordered_bb[id][2] = width;
        ordered_bb[id][3] = height;


        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), LABELS_COLORS[class_ids[idx]],
                  THICKNESS);

        // Draw class labels.
        draw_label(input_image, label, left, top);

    }

    IoU = compute_IoU(ordered_bb, gr_boxes_vec);

}

void Detection::make_detection(cv::Mat &frame, const std::string& ground_truth_path) {
    vector<Mat> detections;
    detections = pre_process(frame);

    // read ground truth
    vector<array<int, 4>> gr_boxes_vec;
    read_bb_file(ground_truth_path, gr_boxes_vec);

    string IoU;
    array<int, 4> ordered_bb [4];
    post_process(frame, detections, class_list, gr_boxes_vec, IoU, ordered_bb);

    // -- Put efficiency information.
    // -- The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)

    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string inference_time = format("Inference time : %.2f ms", t);
    //putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
    cout << inference_time << endl;

    // -- write the output
    write_output(ordered_bb);

}

// -- TEST METHODS

// -- test method to write all testset predicted bounding boxes
void Detection::make_detection_testset(int N_IMAGES) {

    ofstream outfile_IoU("./test_results/pred_boxes_1.txt");

    for (int j = 1; j <= N_IMAGES; j++) {

        // Load image.
        Mat frame;

        string img_path, det_path;

        if (j < 10) {
            img_path = "./input/0" + to_string(j) + ".jpg";
            det_path = "./det/0" + to_string(j) + ".txt";
        } else {
            img_path = "./input/" + to_string(j) + ".jpg";
            det_path = "./det/" + to_string(j) + ".txt";

        }


        frame = imread(img_path);

        if (frame.empty()) {
            cerr << "Error reading input image!\n";
            return;
        }


        vector<Mat> detections;
        detections = pre_process(frame);


        // read ground truth
        vector<array<int, 4>> gr_boxes_vec;
        array<int, 4> ordered_bb [4];
        read_bb_file(det_path, gr_boxes_vec);

        string IoU;

        post_process(frame, detections, class_list, gr_boxes_vec, IoU, ordered_bb);

        string bboxes;

        for (int i = 0; i <4; i++) {
            if (ordered_bb[i][0] != -1) {
                //write the output with class id
                bboxes += to_string(ordered_bb[i][0]) += string(" ") += to_string(ordered_bb[i][1]) += string(" ") +=
                to_string(ordered_bb[i][2]) += string(" ") += to_string(ordered_bb[i][3])
                        += string(" ") += to_string(i) += "\n";
            }
        }

        // write results
        outfile_IoU << img_path << "\n\n" << bboxes << "\n" << "---------------------" << "\n";


    }

    outfile_IoU.close();


}


// test method to compute average IoU on testset
void Detection::compute_avg_IoU_testset(int N_IMAGES) {

    ofstream outfile_IoU("./test_results/IoU_model2.txt");

    for (int j = 1; j <= N_IMAGES; j++) {

        // Load image.
        Mat frame;

        string img_path, det_path;

        if (j < 10) {
            img_path = "./input/0" + to_string(j) + ".jpg";
            det_path = "./det/0" + to_string(j) + ".txt";
        } else {
            img_path = "./input/" + to_string(j) + ".jpg";
            det_path = "./det/" + to_string(j) + ".txt";

        }


        frame = imread(img_path);

        if (frame.empty()) {
            cerr << "Error reading input image!\n";
            return;
        }


        vector<Mat> detections;
        detections = pre_process(frame);


        // read ground truth
        vector<array<int, 4>> gr_boxes_vec;
        array<int, 4> ordered_bb [4];
        read_bb_file(det_path, gr_boxes_vec);

        string IoU;

        post_process(frame, detections, class_list, gr_boxes_vec, IoU, ordered_bb);

        // write results
        outfile_IoU << img_path << "\n\n" << IoU << "\n" << "---------------------" << "\n";


    }

    cout << "Avg IoU: " << (counter / n_det) << endl;
    cout << "Total detections: " << n_det << endl;
    outfile_IoU << "\n\nAvg IoU: " << (counter / n_det) << endl;

    outfile_IoU.close();

}
