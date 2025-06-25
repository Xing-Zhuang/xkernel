#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include "utils.h"

void launch_reduce_kernel(const float* a,float* b, int n,REDUCE_KERNEL_VERSION version);

 
void reduce(torch::Tensor a, torch::Tensor b) {
 
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a tensor must be of type float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b tensor must be of type float32");
    
    TORCH_CHECK(a.dim() == 1, "The dimension of tensor a must be 1");
    TORCH_CHECK(b.dim() == 0, "The dimension of tensor a must be 1");

    TORCH_CHECK(a.is_contiguous(), "Tensor a must be contiguous");
        
    TORCH_CHECK(a.is_cuda(), "Tensor a must be on the GPU");
    TORCH_CHECK(b.is_cuda(), "Tensor b must be on the GPU");
   
    const int n = a.numel();
    launch_reduce_kernel(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        n,
        REDUCE_KERNEL_VERSION::V2
    );
    
    cudaDeviceSynchronize();
}

 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce", &reduce, "reduce two tensors (CUDA)");
}