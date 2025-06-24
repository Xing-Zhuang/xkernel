#include <cuda_runtime.h>
#include <iostream>
#include "utils.h"

 
__global__ void add_kernel(
    const float*  __restrict__ a, 
    const float*  __restrict__ b, 
    float*  __restrict__ c, 
    int n
) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = 4*idx; 

    if (offset < n) {
        int remain = n-offset;
        if (remain>=4){
            float4 float4_a = FLOAT4(a+offset);
            float4 float4_b = FLOAT4(b+offset);
            float4 float4_c;
            float4_c.x = float4_a.x + float4_b.x;
            float4_c.y = float4_a.y + float4_b.y;
            float4_c.z = float4_a.z + float4_b.z;
            float4_c.w = float4_a.w + float4_b.w;
            FLOAT4(c+offset) = float4_c;
        }
        else{
            for(int i=0;i<remain;i++){
                c[offset+i] = a[offset+i] + b[offset+i];
            }
        }
  
    }
}

 
void launch_add_kernel(
    const float* a, 
    const float* b, 
    float* c, 
    int n
){
 
    int block_dim_x = 256;
     
    dim3 block(block_dim_x,1,1);
    dim3 grid(CEIL(n,(block_dim_x*4)),1,1);
    add_kernel<<<grid, block>>>(a, b, c, n);
}