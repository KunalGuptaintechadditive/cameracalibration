#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <iostream>

using namespace std;
using namespace cv;

//persepective transform used on buildplate
//Mat perspectiveTransform(const Mat& originalImage, const vector<Point2f>& source_points, int x, int y) {
//    // Define the destination points (corner points of the desired square)
//    vector<Point2f> destination_points = {
//        Point2f(0, 0),
//        Point2f(x, 0),
//        Point2f(x, y),
//        Point2f(0, y)
//    };
//
//    // Compute the perspective transformation matrix
//    Mat transformMatrix = getPerspectiveTransform(source_points, destination_points);
//
//    // Apply the perspective transformation
//    Mat correctedImage;
//    warpPerspective(originalImage, correctedImage, transformMatrix, Size(x, y));
//    return correctedImage;
//}

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{ 6,9 };

int main()
{
    // Creating vector to store vectors of 3D points for each checkerboard image (real world cordinates)
    std::vector<std::vector<cv::Point3f> > objpoints;

    // Creating vector to store vectors of 3D points for each checkerboard image (image world cordinates)
    std::vector<std::vector<cv::Point2f> > imgpoints;

    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < CHECKERBOARD[1]; i++)
    {
        for (int j = 0; j < CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j, i, 0));
    }


    // Extracting path of individual image stored in a given directory
    std::vector<cv::String> images;
    // Path of the folder containing checkerboard on baseplate images
    //created this from another opencv project
    std::string path = "../images/*.png";

    cv::glob(path, images);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

    cv::Mat frame, gray;
    // vector to store the pixel coordinates of detected checker board corners 
    std::vector<cv::Point2f> corner_pts;
    bool success;
    // cout << "diiffff" << CHECKERBOARD[0] * CHECKERBOARD[1] - CHECKERBOARD[0];
    vector<Point2f> source_points;


    // Looping over all the images in the directory
    for (int i{ 0 }; i < images.size(); i++)
    {
        frame = cv::imread(images[i]);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Finding checker board corners
        // If desired number of corners are found in the image then success = true  
        success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        //
        /*
         * If desired number of corner are detected,
         * we refine the pixel coordinates and display
         * them on the images of checker board
        */
        if (success)
        {
            cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

            // refining pixel coordinates for given 2d points.
            cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            // Displaying the detected corner points on the checker board
            cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }

        //cv::imshow("Image", frame);
        //cv::waitKey(0);
    }
    
    // cordinates for persepective transforming the buildplate images
    size_t foruse = static_cast<size_t>(CHECKERBOARD[0]) * static_cast<size_t>(CHECKERBOARD[1]);
    source_points.push_back(corner_pts[0]);
    source_points.push_back(corner_pts[foruse - CHECKERBOARD[0]]);
    source_points.push_back(corner_pts[foruse - 1]);
    source_points.push_back(corner_pts[CHECKERBOARD[0] - 1]);

    cv::destroyAllWindows();

    cv::Matx33f cameraMatrix(cv::Matx33f::eye()); // intrinsic camera matrix
    cv::Vec<float, 5> distCoeffs(0, 0, 0, 0, 0); // distortion coefficients

    std::vector<cv::Mat> R, T;

   /* int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 +
        cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT;*/

    int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_PRINCIPAL_POINT + cv::CALIB_FIX_K1;
    /*
     * Performing camera calibration by
     * passing the value of known 3D points (objpoints)
     * and corresponding pixel coordinates of the
     * detected corners (imgpoints)
    */

    float error = cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T, flags);
    std::cout << "Reprojection error = " << error << endl;
    std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
    std::cout << "distCoeffs : " << distCoeffs << std::endl;
    std::cout << "Rotation vector : " << R.size() << std::endl;
    std::cout << "Translation vector : " << T.size() << std::endl;


    // Trying to undistort the image using the camera parameters obtained from calibration

    cv::Mat image;
    image = cv::imread(images[0]);
    cv::Mat dst, map1, map2, new_camera_matrix;
    cv::Size imageSize(cv::Size(image.cols, image.rows));

    // Refining the camera matrix using parameters obtained by calibration
    new_camera_matrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0);

    // undistorting image and storing it in dst
    //cv::undistort(frame, dst, new_camera_matrix, distCoeffs, new_camera_matrix);

    imshow("orignal image", image); 

    //Applying the perspective transform for orthographic view
    int x = image.cols;
    int y = image.rows;

   // cv::imshow("undistorted image", dst);

    cv::Mat map11, map22;
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),
        cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
        imageSize, CV_16SC2, map11, map22);

    cv::Mat correctedImage;
    cv::remap(image, correctedImage, map11, map22, cv::INTER_LINEAR);

    cv::imshow("better undistorted image", correctedImage);
    cv::waitKey(0);



   // Mat finalimg = perspectiveTransform(correctedImage, source_points, x, y);
    //Displaying the undistorted image of buildplate 
    //cv::imshow("after perspective transform", finalimg);  //flat chessboard placed on the buildplate
    //cv::waitKey(0);

    //baseplate images



    //std::vector<cv::String> buildimages;
    //// Path of the folder containing checkerboard images
    //std::string path1 = "../buildlayerimages/*.png";

    //cv::glob(path1, buildimages);
    //cv::Mat buildimage;
    //buildimage = cv::imread(buildimages[0]);

    //// Define the source points (corner points of the distorted rectangle)
    //vector<Point2f> source_points1 = {
    //    Point2f(409, 232), // top-left corner
    //    Point2f(981, 196), // top-right corner
    //    Point2f(1015, 807), // bottom-right corner
    //    Point2f(412, 809)  // bottom-left corner
    //};

    ////undistorted image
    //Mat undst;
    //cv::undistort(buildimage, undst, new_camera_matrix, distCoeffs, new_camera_matrix);
    //int sidelength = 1000;

    //Mat buildimage1 = perspectiveTransform(buildimage, source_points1, sidelength, sidelength);
    //Mat undst1 = perspectiveTransform(undst, source_points1, sidelength, sidelength);

    //// imshow("build orignal", buildimage1);
    //// imshow("undistorted build", undst1);
    //cv::waitKey(0);
    return 0;
}
