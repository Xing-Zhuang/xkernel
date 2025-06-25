#include <cuda_runtime.h>
#include <iostream>
#include "utils.h"
#include <cassert>

 

//by share mem
#define BLOCK_SIZE_REDUCE_KERNEL_V1 256
__global__ void reduce_kernel_v1(
    const float*  __restrict__ a, 
    float*  __restrict__ b,
    int n
) {
      
    extern __shared__ float share_mem[BLOCK_SIZE_REDUCE_KERNEL_V1];//这里让share mem大小与block size，方便计算
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
     
    share_mem[tid] = 0;
    if(idx<n){
        share_mem[tid] = a[idx];
    } 
    __syncthreads();//synchronize all warps in a block

    //offset from blockDim.x/2 to 1 (for 合并访存）
    for(int offset = blockDim.x/2;offset>0;offset>>=1){
        if(tid<offset){
            share_mem[tid] = share_mem[tid] + share_mem[tid+offset];
        }
        __syncthreads();
    }
     
    if(tid == 0){
        atomicAdd(b,share_mem[tid]);
    }
}

//by warp shfl
#define WARP_SHFL_MASK 0xFFFFFFFF
__global__ void reduce_kernel_v2(
    const float*  __restrict__ a, 
    float*  __restrict__ b,
    int n
) {
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
     
    __shared__ float share_mem[32];//用于聚合所有warp reduce的结果（每个block最多有32个warp）
    

    float data = (idx<n)?a[idx]:0;
    
    for(int offset = WARP_SIZE/2; offset>0; offset>>=1){
        data = data + __shfl_down_sync(WARP_SHFL_MASK,data,offset);
    } 

    if(lane_id == 0){
        share_mem[warp_id] = data;
    }
    __syncthreads();
    
    
    if(warp_id == 0){
        data = (lane_id<blockDim.x/WARP_SIZE)?share_mem[lane_id]:0;
        for(int offset =  WARP_SIZE/2; offset>0; offset>>=1){
            data = data + __shfl_down_sync(WARP_SHFL_MASK,data,offset);
        }

        if(lane_id == 0){
            atomicAdd(b,data);
        }
    }

}

//
#define WARP_SHFL_MASK 0xFFFFFFFF
__global__ void reduce_kernel_v3(
    const float*  __restrict__ a, 
    float*  __restrict__ b,
    int n
) {
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
     
    __shared__ float share_mem[32];//用于聚合所有warp reduce的结果（每个block最多有32个warp）
    

    float data = (idx<n)?a[idx]:0;
    
    for(int offset = WARP_SIZE/2; offset>0; offset>>=1){
        data = data + __shfl_down_sync(WARP_SHFL_MASK,data,offset);
    } 

    if(lane_id == 0){
        share_mem[warp_id] = data;
    }
    __syncthreads();
    
    
    if(warp_id == 0){
        data = (lane_id<blockDim.x/WARP_SIZE)?share_mem[lane_id]:0;
        for(int offset =  WARP_SIZE/2; offset>0; offset>>=1){
            data = data + __shfl_down_sync(WARP_SHFL_MASK,data,offset);
        }

        if(lane_id == 0){
            atomicAdd(b,data);
        }
    }

}   

 
void launch_reduce_kernel(
    const float* a,
    float* b,
    int n,
    REDUCE_KERNEL_VERSION version
){
    if(version == REDUCE_KERNEL_VERSION::V1){
        int block_dim_x = BLOCK_SIZE_REDUCE_KERNEL_V1;

        dim3 block(block_dim_x,1,1);
        dim3 grid(CEIL(n,block_dim_x),1,1);
        
        reduce_kernel_v1<<<grid, block>>>(a, b, n);
    }
    else if(version == REDUCE_KERNEL_VERSION::V2){
        int block_dim_x = 1024;
        assert(block_dim_x % WARP_SIZE == 0);
       
        dim3 block(block_dim_x,1,1);
        dim3 grid(CEIL(n,block_dim_x),1,1);
        
        reduce_kernel_v2<<<grid, block>>>(a, b, n);
    }
    
}