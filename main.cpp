#include <iostream>
#include <cstdio>
#include <cstdlib>
// opencv includes
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;
using namespace std;

typedef struct
{
    double cx;
    double cy;

    double fx;
    double fy;

    double k1;
    double k2;
    double k3;

    double p1;
    double p2;
} camParams_t;

typedef struct
{
    int cx;
    int cy;

    int fx;
    int fy;

    int k1;
    int k2;
    int k3;

    int p1;
    int p2;

    int newValue;
} camParamsInt_t;
void TrackBarCxCallback(int position, void *data)
{
    camParamsInt_t *ptr = (camParamsInt_t*)data;
    ptr->cx = position;
    ptr->newValue = 1;
}
void TrackBarCyCallback(int position, void *data)
{
    camParamsInt_t *ptr = (camParamsInt_t*)data;
    ptr->cy = position;
    ptr->newValue = 1;
}
void TrackBarFxCallback(int position, void *data)
{
    camParamsInt_t *ptr = (camParamsInt_t*)data;
    ptr->fx = position;
    ptr->newValue = 1;
}
void TrackBarFyCallback(int position, void *data)
{
    camParamsInt_t *ptr = (camParamsInt_t*)data;
    ptr->fy = position;
    ptr->newValue = 1;
}
/**/
void TrackBarK1Callback(int position, void *data)
{
    camParamsInt_t *ptr = (camParamsInt_t*)data;
    ptr->k1 = position;
    ptr->newValue = 1;
}

void TrackBarK2Callback(int position, void *data)
{
    camParamsInt_t *ptr = (camParamsInt_t*)data;
    ptr->k2 = position;
    ptr->newValue = 1;
}

void TrackBarK3Callback(int position, void *data)
{
    camParamsInt_t *ptr = (camParamsInt_t*)data;
    ptr->k3 = position;
    ptr->newValue = 1;
}

void TrackBarP1Callback(int position, void *data)
{
    camParamsInt_t *ptr = (camParamsInt_t*)data;
    ptr->p1 = position;
    ptr->newValue = 1;
}

void TrackBarP2Callback(int position, void *data)
{
    camParamsInt_t *ptr = (camParamsInt_t*)data;
    ptr->p2 = position;
    ptr->newValue = 1;
}
void Undistorting(Mat *imageSrc, camParams_t camParams, Mat *imageDst);

int main(int argc, char* argv[])
{
    const int argNumber = 2;
    if(argc < argNumber)
    {
        printf("Error: argc = %i, must be %i.\n", argc, argNumber);
        return -1;
    }
    char fileName[256];
    strcpy(fileName, argv[1]);
    camParams_t camParams;
    if(argc == 11)
    {
        camParams.cx = atof(argv[2]);
        camParams.cy = atof(argv[3]);

        camParams.fx = atof(argv[4]);
        camParams.fy = atof(argv[5]);

        camParams.k1 = atof(argv[6]);
        camParams.k2 = atof(argv[7]);
        camParams.k3 = atof(argv[8]);

        camParams.p1 = atof(argv[9]);
        camParams.p2 = atof(argv[10]);
    }
    else
    {
//        camParams.cx = 1280.0/2.0;
//        camParams.cy = 960.0/2.0;

//        camParams.fx = 4000.0;
//        camParams.fy = 4000.0;

//        camParams.k1 = 0.11;
//        camParams.k2 = 80.58;
//        camParams.k3 = -2508;

//        camParams.p1 = 0.011;
//        camParams.p2 = 0.014;

        camParams.cx = 640.0;
        camParams.cy = 512.0;

        camParams.fx = 2561.4754;
        camParams.fy = 2561.4754;

        camParams.k1 = -6.86134018741640e-08;
        camParams.k2 = 8.19996908328261e-13;
        camParams.k3 = -2.54243655007380e-18;

        camParams.p1 = -2.90971306430718e-10;
        camParams.p2 = -7.13804936692982e-11;
    }




    Mat imageSrc = imread(fileName, IMREAD_GRAYSCALE);
    Mat imageDst = Mat::zeros(imageSrc.rows, imageSrc.cols, CV_8U);
    namedWindow(fileName, WINDOW_AUTOSIZE);

    camParamsInt_t camParamsInt;
    camParamsInt.cx = static_cast<int>(camParams.cx);
    camParamsInt.cy = static_cast<int>(camParams.cy);

    camParamsInt.fx = static_cast<int>(camParams.fx);
    camParamsInt.fy = static_cast<int>(camParams.fy);

    camParamsInt.k1 = 5000;
    camParamsInt.k2 = 5000;
    camParamsInt.k3 = 5000;

    camParamsInt.p1 = 5000;
    camParamsInt.p2 = 5000;
    camParamsInt.newValue = 0;

    createTrackbar("k1", fileName, &camParamsInt.k1, 10000, TrackBarK1Callback, &camParamsInt);
    createTrackbar("k2", fileName, &camParamsInt.k2, 10000, TrackBarK2Callback, &camParamsInt);
    createTrackbar("k3", fileName, &camParamsInt.k3, 10000, TrackBarK3Callback, &camParamsInt);

    createTrackbar("p1", fileName, &camParamsInt.p1, 10000, TrackBarP1Callback, &camParamsInt);
    createTrackbar("p2", fileName, &camParamsInt.p2, 10000, TrackBarP2Callback, &camParamsInt);

    createTrackbar("cx", fileName, &camParamsInt.cx, 1280, TrackBarCxCallback, &camParamsInt);
    createTrackbar("cy", fileName, &camParamsInt.cy, 960, TrackBarCyCallback, &camParamsInt);

    createTrackbar("fx", fileName, &camParamsInt.fx, 8000, TrackBarCxCallback, &camParamsInt);
    createTrackbar("fy", fileName, &camParamsInt.fy, 8000, TrackBarCyCallback, &camParamsInt);


    printf("k1 = %e, k2 = %e, k3 = %e, p1 = %e, p2 = %e\n",
           camParams.k1, camParams.k2, camParams.k3,
           camParams.p1, camParams.p2);
    printf("cx = %lf, cy = %lf, fx = %lf, fy = %lf\n",
           camParams.cx, camParams.cy, camParams.fx,
           camParams.fy);

    for(;;)
    {

//        camParams.k1 = double((camParamsInt.k1 - 5000)/100);
//        camParams.k2 = double((camParamsInt.k2 - 5000)/100);
//        camParams.k3 = double((camParamsInt.k3 - 5000)/100);

//        camParams.p1 = double((camParamsInt.p1 - 5000)/100);
//        camParams.p2 = double((camParamsInt.p2 - 5000)/100);

//        camParams.cx = double(camParamsInt.cx);
//        camParams.cy = double(camParamsInt.cy);

//        camParams.fx = double(camParamsInt.fx);
//        camParams.fy = double(camParamsInt.fy);

        Mat intrinsic_matrix = Mat::zeros(3,3,CV_64F);
        Mat distortion_coeffs = Mat::zeros(1,5,CV_64F);

        intrinsic_matrix.at<double>(0,0) = camParams.fx;   // fx
        intrinsic_matrix.at<double>(1,1) = camParams.fy;   // fy
        intrinsic_matrix.at<double>(0,2) = camParams.cx;   // cx
        intrinsic_matrix.at<double>(1,2) = camParams.cy;   // cy
        intrinsic_matrix.at<double>(2,2) = 1.0;

        distortion_coeffs.at<double>(0,0) = camParams.k1; // k1D
        distortion_coeffs.at<double>(0,1) = camParams.k2; // k2D
        distortion_coeffs.at<double>(0,4) = camParams.k3; // k3D

        distortion_coeffs.at<double>(0,2) = camParams.p1; // p1D
        distortion_coeffs.at<double>(0,3) = camParams.p2; // p2D

        Mat map1, map2;

        Size image_size = imageSrc.size();
        // Build the undistort map which we will use for all
        // subsequent frames.
        initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs,
                                Mat(), intrinsic_matrix, image_size,
                                CV_32FC1, map1, map2);

        remap(imageSrc, imageDst, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, Scalar());


        //Undistorting(&imageSrc, camParams, &imageDst);
        if(camParamsInt.newValue == 1)
        {
            camParams.k1 = double((camParamsInt.k1 - 5000)/100);
            camParams.k2 = double((camParamsInt.k2 - 5000)/100);
            camParams.k3 = double((camParamsInt.k3 - 5000)/100);

            camParams.p1 = double((camParamsInt.p1 - 5000)/100);
            camParams.p2 = double((camParamsInt.p2 - 5000)/100);

            camParams.cx = double(camParamsInt.cx);
            camParams.cy = double(camParamsInt.cy);

            camParams.fx = double(camParamsInt.fx);
            camParams.fy = double(camParamsInt.fy);

            camParamsInt.newValue = 0;
            printf("k1 = %e, k2 = %e, k3 = %e, p1 = %e, p2 = %e\n",
                   camParams.k1, camParams.k2, camParams.k3,
                   camParams.p1, camParams.p2);
            printf("cx = %lf, cy = %lf, fx = %lf, fy = %lf\n",
                   camParams.cx, camParams.cy, camParams.fx,
                   camParams.fy);
        }


        char key = waitKey(1);
        if(key == 10)
        {
            char *message = (char*)malloc(128);
            sprintf(message, "k1 = %e, k2 = %e, k3 = %e, p1 = %e, p2 = %e",
                   camParams.k1, camParams.k2, camParams.k3,
                   camParams.p1, camParams.p2);
            putText(imageDst, message,Point(50,  imageDst.rows - 50), FONT_HERSHEY_PLAIN, 1.0, Scalar(0xFF, 0xFF, 0xFF), 1, 4);
            //imwrite("result.tiff", imageDst);

            sprintf(message, "cx = %lf, cy = %lf, fx = %lf, fy = %lf",
                   camParams.cx, camParams.cy, camParams.fx,
                   camParams.fy);
            putText(imageDst, message,Point(50,  imageDst.rows - 25), FONT_HERSHEY_PLAIN, 1.0, Scalar(0xFF, 0xFF, 0xFF), 1, 4);
            imwrite("result.tiff", imageDst);



            free(message);
            break;
        }

        if(!imageDst.empty())
            imshow(fileName, imageDst);

    }
    cout << "Hello World!" << endl;
    return 0;
}

void Undistorting(Mat *imageSrc, camParams_t camParams, Mat *imageDst)
{


    for(int u = 0; u < imageDst->rows; u++)
    {
        for(int v = 0; v < imageDst->cols; v++)
        {
            double x = (v - camParams.cx)/camParams.fx;
            double y = (u - camParams.cy)/camParams.fy;

            double radius = sqrt(pow(x, 2.0) + pow(y, 2.0));
            double radDist = 1.0 + camParams.k1 * pow(radius, 2.0) +
                    camParams.k2 * pow(radius, 4.0) +
                    camParams.k3 * pow(radius, 6.0);

            double tanDistx = 2.0 * camParams.p1*x*y + camParams.p2*(radius*radius + 2.0*x*x);
            double tanDisty = camParams.p1*(radius*radius + 2.0*y*y) +2.0*camParams.p2*x*y;

            double x_ = x * radDist + tanDistx;
            double y_ = y * radDist + tanDisty;

            double v_ = camParams.cx + x_ * camParams.fx;
            double u_ = camParams.cy + y_ * camParams.fy;

            if((u_ >= 0) && (u_ < imageSrc->rows) && (v_ >= 0) && (v_ < imageSrc->cols))
                imageDst->at<uchar>(u,v) = imageSrc->at<uchar>(cvRound(u_),cvRound(v_));

            else
               imageDst->at<uchar>(u,v) = 0;
        }
    }
}

