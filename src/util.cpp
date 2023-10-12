/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>

#include "NvInfer.h"
#include "img_seg/util.h"

namespace util
{

size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
}

ImageBase::ImageBase(const nvinfer1::Dims& dims)
    : mDims(dims)
{
    assert(4 == mDims.nbDims);
    assert(1 == mDims.d[0]);
}

size_t ImageBase::volume() const
{
    return mDims.d[3] /* w */ * mDims.d[2] /* h */ * 3;
}

void ImageBase::read()
{
    std::ifstream infile(mPPM.filename, std::ifstream::binary);
    if (!infile.is_open())
    {
        std::cerr << "ERROR: cannot open PPM image file: " << mPPM.filename << std::endl;
    }
    infile >> mPPM.magic >> mPPM.w >> mPPM.h >> mPPM.max;
    infile.seekg(1, infile.cur);
    mPPM.buffer.resize(volume());
    infile.read(reinterpret_cast<char*>(mPPM.buffer.data()), volume());
    infile.close();
}

void ImageBase::write()
{
    std::ofstream outfile(mPPM.filename, std::ofstream::binary);
    if (!outfile.is_open())
    {
        std::cerr << "ERROR: cannot open PPM image file: " << mPPM.filename << std::endl;
    }
    outfile << mPPM.magic << " " << mPPM.h << " " << mPPM.w << " " << mPPM.max << std::endl;
    

    outfile.write(reinterpret_cast<char*>(mPPM.buffer.data()), volume()/3);
    outfile.close();
}


ArgmaxImageWriter::ArgmaxImageWriter(const nvinfer1::Dims& dims, const std::vector<int>& palette, const int num_classes)
    : ImageBase( dims)
    , mNumClasses(num_classes)
    , mPalette(palette)
{
  
}

void ArgmaxImageWriter::process(const float* buffer)
{
    mPPM.magic = "P6";
    mPPM.w = mDims.d[2];
    mPPM.h = mDims.d[3];
    mPPM.max = 255;
    mPPM.buffer.resize(volume() / 3);

    
    for (int j = 0, HW = mPPM.h * mPPM.w; j < HW; ++j)
    {
        int clsid = 0;
        if(buffer[j] > buffer[j + HW]){
            clsid = 0;
        }
        else{
            clsid = 1;
        }
        mPPM.buffer[j ] = _colors[clsid];
    //     mPPM.buffer[j*3+1] = colors[clsid][1];
    //     mPPM.buffer[j*3+2] = colors[clsid][2];
     }
   
    

}

}; // namespace util
