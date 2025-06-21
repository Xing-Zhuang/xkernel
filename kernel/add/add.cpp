#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

void launch_add_kernel(const float* a, const float* b,float* c, int n);

 
torch::Tensor add(torch::Tensor a, torch::Tensor b) {
 
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Both tensors must be of type float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Both tensors must be of type float32");
 
    
    TORCH_CHECK(a.dim() == 1, "The dimension of tensor a must be 1");
    TORCH_CHECK(b.dim() == 1, "The dimension of tensor b must be 1");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor shapes must match");

    
    TORCH_CHECK(a.is_contiguous(), "Tensor a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Tensor b must be contiguous");
    
    
    TORCH_CHECK(a.is_cuda(), "Tensor a must be on the GPU");
    TORCH_CHECK(b.is_cuda(), "Tensor b must be on the GPU");


    auto c = torch::zeros_like(a,torch::kFloat32);
    const int n = a.numel();
    
    launch_add_kernel(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );
    
     
    cudaDeviceSynchronize(); 
    return c;
}

 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Add two tensors (CUDA)");
}