#pragma once

#include <cstddef>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <vector>

class CudaArray
{
private:
    size_t _size;
    thrust::device_vector<float> _data;

public:
    CudaArray(size_t size):_size(size), _data(size) {};

    float* get_raw_ptr()
    {
        return thrust::raw_pointer_cast(_data.data());
    }

    void copy_from_host(const std::vector<float>& host_vec)
    {
        thrust::copy(host_vec.begin(), host_vec.end(), _data.begin());
    }

    auto copy_to_host() const
    {
        std::vector<float> host_vec(_size);
        thrust::copy(_data.begin(), _data.end(), host_vec.begin());
        return host_vec;
    }

    const auto size() const {return _size;}

    auto sum() const
    {
        return thrust::reduce(_data.begin(), _data.end(), 0.0f, thrust::plus<float>());
    }
};

// Implemented in `72-py-cuarray.cu`
void launch_add(CudaArray& a, CudaArray& b, CudaArray& out);


