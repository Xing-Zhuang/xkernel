
#ifndef UTIL
#define UTIL

#define CEIL(a,b) (a+b-1)/b
#define FLOAT4(addr) *((float4*)(addr))
#define WARP_SIZE 32

enum class REDUCE_KERNEL_VERSION{
    V1,
    V2,
    V3
};

#endif
