#include "img_seg/seg_node.h"

constexpr long long operator"" _MiB(long long unsigned val)
{
    return val * (1 << 20);
}

using sample::gLogError;
using sample::gLogInfo;

//!
//! \class SampleSegmentation
//!
//! \brief Implements semantic segmentation using FCN-ResNet101 ONNX model.
//!

SampleSegmentation::SampleSegmentation(const std::string& engineFilename)
        : mEngineFilename(engineFilename)
        , mEngine(nullptr)
{
    // De-serialize engine from file
    std::ifstream engineFile(engineFilename, std::ios::binary);
    if (engineFile.fail())
    {
        return;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    util::UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};
    mEngine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    assert(mEngine.get() != nullptr);
}

//!
//! \brief Runs the TensorRT inference.
//!
//! \details Allocate input and output memory, and executes the engine.
//!
cv::Mat  SampleSegmentation::infer(const cv::Mat input_img, int32_t width, int32_t height)
{
  
    auto context = util::UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
 
    const int inputNum = 1 * 3 * 256 * 320;
    const int outputNum = 1 * 2 * 256 * 320;
    float* buffer[2];
    auto input_idx = mEngine->getBindingIndex("images");
    auto output_idx = mEngine->getBindingIndex("output");
    cudaMalloc((void**)&buffer[input_idx], inputNum * sizeof(float));
    cudaMalloc((void**)&buffer[output_idx], inputNum * sizeof(float));
    auto input_size = inputNum * sizeof(float);
    auto output_size = outputNum * sizeof(float);
    float* input_mem = buffer[0];
    float* output_mem = buffer[1];

    auto output_dims = context->getBindingDimensions(output_idx);

    // Read image data from file and mean-normalize it

    // image to float
    cv::Mat img{cv::Size(320,256), CV_8UC3};
    input_img.copyTo(img);
    // auto img = cv::imread("/home/nvidia/data_1t/sds_ws/ros_ws/src/img_seg/test_img/test1.jpg", cv::IMREAD_COLOR);

    if (height != img.rows || width != img.cols) {
        cv::resize(img, img, cv::Size(width, height));
    }

    // src: BCHW RGB [0,1] fp32
    auto src_dims = mEngine->getBindingDimensions(0);
    auto src_h = src_dims.d[2], src_w = src_dims.d[3];
    auto src_n = src_h * src_w;
    auto* pic_input = new float[input_size];
    cv::Mat src(src_h, src_w, CV_32FC3);
    {
        auto src_data = (float*)(src.data);
      int count = 0;
        for (int y = 0; y < height ; ++y) {

            for (int x = 0; x < width; ++x) {
                auto &&bgr = img.at<cv::Vec3b>(y, x);
                /*r*/ *(pic_input + y*width + x) = bgr[2] / 255.;
                /*g*/ *(pic_input + src_n + y*width + x) = bgr[1] / 255.;
                /*b*/ *(pic_input + src_n*2 + y*width + x) = bgr[0] / 255.;
            }
        }
    }

   
    cudaStream_t stream;
    bool res2 = cudaStreamCreate(&stream);
    // Copy image data to input binding memory
    cudaMemcpyAsync(input_mem,pic_input, input_size, cudaMemcpyHostToDevice, stream);
    // Run TensorRT inference
    void* bindings[] = {input_mem, output_mem};
    bool status = context->enqueueV2(bindings, stream, nullptr);

    // Copy predictions from output binding memory
    auto output_buffer = std::unique_ptr<float>{new float[output_size]};
    cudaMemcpyAsync(output_buffer.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream) ;
    cudaStreamSynchronize(stream);
    // Plot the semantic segmentation predictions of 2 classes in a colormap image and write to file
    const int num_classes{2};
    const std::vector<int> palette{(0x1 << 25) - 1, (0x1 << 15) - 1, (0x1 << 21) - 1};

    auto output_image{util::ArgmaxImageWriter(output_dims, palette, num_classes)};

    output_image.process(output_buffer.get());
    

  cv::Mat tmp = cv::Mat(output_image.mPPM.buffer);
  cv::Mat output = tmp.reshape(1, 256).clone();

    // THIS METHOD MAY RETURN WRONG IMG
    // cv::Mat output(cv::Size(320,256), CV_8UC1, output_image.mPPM.buffer);
    
    // count++;
    // cv::putText(output, std::to_string(count), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255) );
    // cv::resize (output, output, cv::Size(1280, 1024));
    // cv::imshow("res", output);
    // cv::waitKey(1);
    
    // Free CUDA resources
    cudaFree(input_mem);
    cudaFree(output_mem);
    return output;
}