/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>    

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */
    string detectorType = "SHITOMASI";    // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    string descriptorType = "BRISK";      // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
    string arg;
    if(argc == 2){
        std::cout << "Usage: " << argv[0] << " <detectorType> <descriptorType> <matcherType> <selectorType>" << std::endl;
        std::cout << "      - detectorType: SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT" << std::endl;
        std::cout << "      - descriptorType: BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT" << std::endl;
        std::cout << "      - matcherType: MAT_BF, MAT_FLANN" << std::endl;
        std::cout << "      - matcherType: SEL_NN, SEL_KNN" << std::endl;
        return 0;
    }
    if(argc >= 3){
        arg = argv[1];
        if(arg == "SHITOMASI" || arg == "HARRIS" || arg == "FAST" || arg == "BRISK" ||
           arg == "ORB" || arg == "AKAZE" || arg == "SIFT"){ 
            detectorType = arg;
        }else{
            cerr << "INVALID argument: " << arg << endl;
            return -1;
        }
        arg = argv[2];
        if(arg == "BRISK" || arg == "BRIEF" || arg == "ORB" ||
           arg == "FREAK" || arg == "AKAZE" || arg == "SIFT"){ 
            descriptorType = arg;
        }else{
            cerr << "INVALID argument: " << arg << endl;
            return -1;
        }
    }
    if(argc >= 5){
        arg = argv[3];
        if(arg == "MAT_BF" || arg == "MAT_FLANN"){ 
            matcherType = arg;
        }else{
            cerr << "INVALID argument: " << arg << endl;
            return -1;
        }
        arg = argv[4];
        if(arg == "SEL_NN" || arg == "SEL_KNN"){ 
            selectorType = arg;
        }else{
            cerr << "INVALID argument: " << arg << endl;
            return -1;
        }
    }
    std::cout << "*************************************************************************************************" << std::endl;
    std::cout << "Using detector: " << detectorType << ", descriptor: " << descriptorType << ", matcher: " << matcherType << ", selection: " << selectorType << std::endl;
    std::cout << "*************************************************************************************************" << std::endl;
    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    vector<double> det_times;     // Detection time for each image
    vector<double> des_times;     // Desciptor extration time for each image
    vector<int> num_detected_keypoints;     // Number of detected key points for each image
    vector<int> keypoints_sizes;     // Mean keypoint size for each image
    vector<int> num_matched_keypoints;     // Number of matched key points for each image
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        if(dataBuffer.size() < dataBufferSize){  
            dataBuffer.push_back(frame);
        }else{
            dataBuffer.erase(dataBuffer.begin());
            dataBuffer.push_back(frame);
        }

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done, image number: " << imgIndex << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        
        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        double det_time;
        if (detectorType.compare("SHITOMASI") == 0){
            det_time = detKeypointsShiTomasi(keypoints, imgGray, false);
        }else if (detectorType.compare("HARRIS") == 0){
            det_time = detKeypointsHarris(keypoints, imgGray, false);
        }else{
            det_time = detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        det_times.push_back(det_time);
        
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        vector<cv::KeyPoint> keypoints_vehicle;
        if (bFocusOnVehicle){
            for(auto &keypoint : keypoints)
            {
                if (vehicleRect.contains(keypoint.pt)){
                    keypoints_vehicle.push_back(keypoint);
                }
            }
            keypoints =  keypoints_vehicle;
        }
        num_detected_keypoints.push_back(keypoints.size());
        double mean_size = 0;
        for(auto &kp: keypoints){
            mean_size += kp.size;
        }
        mean_size /= keypoints.size();
        keypoints_sizes.push_back(mean_size);

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;
            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        auto des_time = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        des_times.push_back(des_time);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string type = "DES_BINARY"; // BRIEF, BRISK, ORB, FREAK and KAZE
            if(descriptorType.compare("SIFT") == 0){
                type = "DES_HOG"; // SIFT
            }

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, type, matcherType, selectorType);
            num_matched_keypoints.push_back(matches.size());
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = false;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }
        std::cout << "*************************************************************************************************" << std::endl;
    } // eof loop over all images

    auto mean_det_time = std::accumulate(det_times.begin(), det_times.end(), 0.0) / det_times.size();
    auto mean_des_time = std::accumulate(des_times.begin(), des_times.end(), 0.0) / des_times.size();
    auto mean_detections = std::accumulate(num_detected_keypoints.begin(), num_detected_keypoints.end(), 0.0) / num_detected_keypoints.size();
    auto mean_matches = std::accumulate(num_matched_keypoints.begin(), num_matched_keypoints.end(), 0.0) / num_matched_keypoints.size();
    auto mean_sizes = std::accumulate(keypoints_sizes.begin(), keypoints_sizes.end(), 0.0) / keypoints_sizes.size();

    cout << "REPORT" << std::endl;
    cout << "- Mean detection time: " << mean_det_time << endl;
    cout << "- Mean detection + desciption extraction time: " << mean_det_time + mean_des_time << endl;
    cout << "- Mean detections: " << mean_detections << endl;
    cout << "- Mean matches: " << mean_matches << endl;
    cout << "- Mean keypoint size: " << mean_sizes << endl;

    return 0;
}
