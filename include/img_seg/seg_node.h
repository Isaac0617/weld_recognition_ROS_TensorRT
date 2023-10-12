#ifndef IMG_SEG_NODE
#define IMG_SEG_NODE
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "img_seg/util.h"
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"
#include <image_transport/image_transport.h>
#include <chrono>
#include <math.h>
#include "AdsLib.h"
#include "AdsNotification.h"
#include "AdsVariable.h"

using sample::gLogError;
using sample::gLogInfo;



cv::Mat rosIMG2mat(sensor_msgs::ImageConstPtr img_msg);

class SampleSegmentation
{

public:
    explicit SampleSegmentation(const std::string& engineFilename);
    cv::Mat infer(const cv::Mat input_img, int32_t width, int32_t height);

private:
    std::string mEngineFilename;                    //!< Filename of the serialized engine.

    nvinfer1::Dims mInputDims;                      //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims;                     //!< The dimensions of the output to the network.
    int count = 0;
    util::UniquePtr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
};

template <typename T>
T read_value(const AdsDevice &route, std::string VAR_NAME);
double read_value2(const AdsDevice &route, std::string VAR_NAME);

void write_value(const AdsDevice &route, std::string VAR_NAME, double val);


#endif