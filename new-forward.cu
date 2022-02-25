#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"
#define TILE_WIDTH 20   
#define maskWidth 7


//////////////////////////////////
/// Unrolling + Constant Weight + Optimized TileWidth
/////////////////////////////////

__constant__ float deviceK[maskWidth*maskWidth*64];
__global__ void conv_forward_kernel(float *__restrict__ y, const float *__restrict__ x, const float *__restrict__ k, const int B, const int M, const int C,
                                    const int H, const int W, const int K) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out;  // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out;  // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) deviceK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

const int W_grid = ceil(1.0*W_out/TILE_WIDTH);


    // Insert your GPU convolution kernel code here
    // int H_out = H – K + 1;
    // int W_out = W – K + 1;

    int n = blockIdx.x;                                   // SAMPLES IN MINI BATCH
    int m = blockIdx.y;                                   // OUTPUT FEATURE MAP
    int h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y;  // HEIGHT OF THE GRID
    int w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x;  // WIDTH OF THE GRID
    float acc = 0;
    //int h_base = (blockIdx.z / W_grid) * blockDim.y;
    //int w_base = (blockIdx.z % W_grid) * blockDim.x;
    //int tx = threadIdx.x;
    //int ty = threadIdx.y;

    
    if( h < H_out && w < W_out) {
        // turn off threads outside the dimensions selected
    
        // parallel_for(n = 0; n < N; n++)
        // parallel_for (m = 0; m < M; m++)
        // parallel_for(h = 0; h < H_out; h++)))
        // parallel_for(w = 0; w < W_out; w++) ))
        for (int c = 0; c < C; c++) {// SUM ACROSS ALL INPUT FEATURES))
            //for (int p = 0; p < K; p++) {  // LOOP THROUGH THE K BY K CONVOLUTION KERNEL))
               //for (int q = 0; q < K; q++) {))
                    acc += x4d(n, c, h + 0, w + 0) * k4d(m, c, 0, 0); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
                    acc += x4d(n, c, h + 0, w + 1) * k4d(m, c, 0, 1);
                    acc += x4d(n, c, h + 0, w + 2) * k4d(m, c, 0, 2);
                    acc += x4d(n, c, h + 0, w + 3) * k4d(m, c, 0, 3);
                    acc += x4d(n, c, h + 0, w + 4) * k4d(m, c, 0, 4);
                    acc += x4d(n, c, h + 0, w + 5) * k4d(m, c, 0, 5);
                    acc += x4d(n, c, h + 0, w + 6) * k4d(m, c, 0, 6);
                    acc += x4d(n, c, h + 1, w + 0) * k4d(m, c, 1, 0); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
                    acc += x4d(n, c, h + 1, w + 1) * k4d(m, c, 1, 1);
                    acc += x4d(n, c, h + 1, w + 2) * k4d(m, c, 1, 2);
                    acc += x4d(n, c, h + 1, w + 3) * k4d(m, c, 1, 3);
                    acc += x4d(n, c, h + 1, w + 4) * k4d(m, c, 1, 4);
                    acc += x4d(n, c, h + 1, w + 5) * k4d(m, c, 1, 5);
                    acc += x4d(n, c, h + 1, w + 6) * k4d(m, c, 1, 6);
                    acc += x4d(n, c, h + 2, w + 0) * k4d(m, c, 2, 0); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
                    acc += x4d(n, c, h + 2, w + 1) * k4d(m, c, 2, 1);
                    acc += x4d(n, c, h + 2, w + 2) * k4d(m, c, 2, 2);
                    acc += x4d(n, c, h + 2, w + 3) * k4d(m, c, 2, 3);
                    acc += x4d(n, c, h + 2, w + 4) * k4d(m, c, 2, 4);
                    acc += x4d(n, c, h + 2, w + 5) * k4d(m, c, 2, 5);
                    acc += x4d(n, c, h + 2, w + 6) * k4d(m, c, 2, 6);
                    acc += x4d(n, c, h + 3, w + 0) * k4d(m, c, 3, 0); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
                    acc += x4d(n, c, h + 3, w + 1) * k4d(m, c, 3, 1);
                    acc += x4d(n, c, h + 3, w + 2) * k4d(m, c, 3, 2);
                    acc += x4d(n, c, h + 3, w + 3) * k4d(m, c, 3, 3);
                    acc += x4d(n, c, h + 3, w + 4) * k4d(m, c, 3, 4);
                    acc += x4d(n, c, h + 3, w + 5) * k4d(m, c, 3, 5);
                    acc += x4d(n, c, h + 3, w + 6) * k4d(m, c, 3, 6);
                    acc += x4d(n, c, h + 4, w + 0) * k4d(m, c, 4, 0); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
                    acc += x4d(n, c, h + 4, w + 1) * k4d(m, c, 4, 1);
                    acc += x4d(n, c, h + 4, w + 2) * k4d(m, c, 4, 2);
                    acc += x4d(n, c, h + 4, w + 3) * k4d(m, c, 4, 3);
                    acc += x4d(n, c, h + 4, w + 4) * k4d(m, c, 4, 4);
                    acc += x4d(n, c, h + 4, w + 5) * k4d(m, c, 4, 5);
                    acc += x4d(n, c, h + 4, w + 6) * k4d(m, c, 4, 6);
                    acc += x4d(n, c, h + 5, w + 0) * k4d(m, c, 5, 0); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
                    acc += x4d(n, c, h + 5, w + 1) * k4d(m, c, 5, 1);
                    acc += x4d(n, c, h + 5, w + 2) * k4d(m, c, 5, 2);
                    acc += x4d(n, c, h + 5, w + 3) * k4d(m, c, 5, 3);
                    acc += x4d(n, c, h + 5, w + 4) * k4d(m, c, 5, 4);
                    acc += x4d(n, c, h + 5, w + 5) * k4d(m, c, 5, 5);
                    acc += x4d(n, c, h + 5, w + 6) * k4d(m, c, 5, 6);
                    acc += x4d(n, c, h + 6, w + 0) * k4d(m, c, 6, 0); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
                    acc += x4d(n, c, h + 6, w + 1) * k4d(m, c, 6, 1);
                    acc += x4d(n, c, h + 6, w + 2) * k4d(m, c, 6, 2);
                    acc += x4d(n, c, h + 6, w + 3) * k4d(m, c, 6, 3);
                    acc += x4d(n, c, h + 6, w + 4) * k4d(m, c, 6, 4);
                    acc += x4d(n, c, h + 6, w + 5) * k4d(m, c, 6, 5);
                    acc += x4d(n, c, h + 6, w + 6) * k4d(m, c, 6, 6);
            
        }
        y4d(n, m, h, w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}


//////////////////////////////////
/// Unrolling + FP16  Conversion
/////////////////////////////////

// __global__ void float2halfK(__half *out, const float *in, int dim) {
//     int i = (blockDim.x * blockIdx.x) + threadIdx.x;

//     if (i < dim) {
//         out[i] = __float2half(in[i]);
//     }
// }

// __global__ void half2floatK(float *out, const __half *in, int dim) {
//     int i = (blockDim.x * blockIdx.x) + threadIdx.x;

//     if (i < dim) {
//         out[i] = __half2float(in[i]);
//     }
// }



// __constant__ float deviceK[maskWidth*maskWidth*64];
// __global__ void conv_forward_kernel(__half *__restrict__ y, const __half *__restrict__ x, const __half *__restrict__ k, const int B, const int M, const int C,
//                                     const int H, const int W, const int K) {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     y - output
//     x - input
//     k - kernel
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;
//     //(void)H_out;  // silence declared but never referenced warning. remove this line when you start working
//     //(void)W_out;  // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = y4d(0,0,0,0)
//     // y4d(0,0,0,0) = a

// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) deviceK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

// const int W_grid = ceil(1.0*W_out/TILE_WIDTH);


//     // Insert your GPU convolution kernel code here
//     // int H_out = H – K + 1;
//     // int W_out = W – K + 1;

//     int n = blockIdx.x;                                   // SAMPLES IN MINI BATCH
//     int m = blockIdx.y;                                   // OUTPUT FEATURE MAP
//     int h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y;  // HEIGHT OF THE GRID
//     int w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x;  // WIDTH OF THE GRID
//     __half acc = 0;
//     //int h_base = (blockIdx.z / W_grid) * blockDim.y;
//     //int w_base = (blockIdx.z % W_grid) * blockDim.x;
//     //int tx = threadIdx.x;
//     //int ty = threadIdx.y;

    
//     if( h < H_out && w < W_out) {
//         // turn off threads outside the dimensions selected
    
//         // parallel_for(n = 0; n < N; n++)
//         // parallel_for (m = 0; m < M; m++)
//         // parallel_for(h = 0; h < H_out; h++)))
//         // parallel_for(w = 0; w < W_out; w++) ))
//         for (int c = 0; c < C; c++) {// SUM ACROSS ALL INPUT FEATURES))
//             //for (int p = 0; p < K; p++) {  // LOOP THROUGH THE K BY K CONVOLUTION KERNEL))
//                //for (int q = 0; q < K; q++) {))
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 0, w + 0) , k4d(m, c, 0, 0))); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 0, w + 1) , k4d(m, c, 0, 1)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 0, w + 2) , k4d(m, c, 0, 2)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 0, w + 3) , k4d(m, c, 0, 3)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 0, w + 4) , k4d(m, c, 0, 4)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 0, w + 5) , k4d(m, c, 0, 5)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 0, w + 6) , k4d(m, c, 0, 6)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 1, w + 0) , k4d(m, c, 1, 0))); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 1, w + 1) , k4d(m, c, 1, 1)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 1, w + 2) , k4d(m, c, 1, 2)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 1, w + 3) , k4d(m, c, 1, 3)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 1, w + 4) , k4d(m, c, 1, 4)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 1, w + 5) , k4d(m, c, 1, 5)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 1, w + 6) , k4d(m, c, 1, 6)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 2, w + 0) , k4d(m, c, 2, 0))); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 2, w + 1) , k4d(m, c, 2, 1)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 2, w + 2) , k4d(m, c, 2, 2)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 2, w + 3) , k4d(m, c, 2, 3)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 2, w + 4) , k4d(m, c, 2, 4)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 2, w + 5) , k4d(m, c, 2, 5)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 2, w + 6) , k4d(m, c, 2, 6)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 3, w + 0) , k4d(m, c, 3, 0))); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 3, w + 1) , k4d(m, c, 3, 1)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 3, w + 2) , k4d(m, c, 3, 2)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 3, w + 3) , k4d(m, c, 3, 3)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 3, w + 4) , k4d(m, c, 3, 4)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 3, w + 5) , k4d(m, c, 3, 5)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 3, w + 6) , k4d(m, c, 3, 6)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 4, w + 0) , k4d(m, c, 4, 0))); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 4, w + 1) , k4d(m, c, 4, 1)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 4, w + 2) , k4d(m, c, 4, 2)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 4, w + 3) , k4d(m, c, 4, 3)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 4, w + 4) , k4d(m, c, 4, 4)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 4, w + 5) , k4d(m, c, 4, 5)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 4, w + 6) , k4d(m, c, 4, 6)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 5, w + 0) , k4d(m, c, 5, 0))); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 5, w + 1) , k4d(m, c, 5, 1)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 5, w + 2) , k4d(m, c, 5, 2)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 5, w + 3) , k4d(m, c, 5, 3)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 5, w + 4) , k4d(m, c, 5, 4)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 5, w + 5) , k4d(m, c, 5, 5)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 5, w + 6) , k4d(m, c, 5, 6)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 6, w + 0) , k4d(m, c, 6, 0))); //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 6, w + 1) , k4d(m, c, 6, 1)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 6, w + 2) , k4d(m, c, 6, 2)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 6, w + 3) , k4d(m, c, 6, 3)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 6, w + 4) , k4d(m, c, 6, 4)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 6, w + 5) , k4d(m, c, 6, 5)));
//                     acc = __hadd (acc , __hmul(x4d(n, c, h + 6, w + 6) , k4d(m, c, 6, 6)));
            
//         }
//         y4d(n, m, h, w) = acc;
//     }

// #undef y4d
// #undef x4d
// #undef k4d
// }

//////////////////////////////////
/// M2 CODE
/////////////////////////////////


// __global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     y - output
//     x - input
//     k - kernel
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;
//     // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
//     // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = y4d(0,0,0,0)
//     // y4d(0,0,0,0) = a

// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     const int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
    
//     const int n = blockIdx.x;
//     const int m = blockIdx.y;
//     const int h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y;
//     const int w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x;

//     float acc = 0;

//     if (h < H_out && w < W_out) {
//         // y4d(n, m, h, w) = 0.0f;
//         for (int c = 0; c < C; c++) { // sum over all input channels
//             for (int p = 0; p < K; p++) // loop over KxK filter
//                 for (int q = 0; q < K; q++)
//                     acc += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//         }
//         y4d(n, m, h, w) = acc;
//     }

// #undef y4d
// #undef x4d
// #undef k4d
// }

//////////////////////////////////
/// Weight Matrix in Constant + Shared Mem
/////////////////////////////////

// __constant__ float deviceK[maskWidth*maskWidth*64];
// __global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C,
//                                     const int H, const int W, const int K) {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     y - output
//     x - input
//     k - kernel
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;
//     //(void)H_out;  // silence declared but never referenced warning. remove this line when you start working
//     //(void)W_out;  // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = y4d(0,0,0,0)
//     // y4d(0,0,0,0) = a

// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// //#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
// #define k4d(i3, i2, i1, i0) deviceK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

// const int W_grid = ceil(1.0*W_out/TILE_WIDTH);
// const int inputTile = TILE_WIDTH + K - 1; 
// extern __shared__ float sharedMem[];
// float* inputShared = &sharedMem[0];
// float* kernelShared = &sharedMem[inputTile*inputTile];

//     // Insert your GPU convolution kernel code here
//     // int H_out = H – K + 1;
//     // int W_out = W – K + 1;

//     int n = blockIdx.x;                                   // SAMPLES IN MINI BATCH
//     int m = blockIdx.y; 
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;                                  // OUTPUT FEATURE MAP
//     int h = (blockIdx.z / W_grid) * blockDim.y + ty;  // HEIGHT OF THE GRID
//     int w = (blockIdx.z % W_grid) * blockDim.x + tx;  // WIDTH OF THE GRID
//     float acc = 0;
//     int h_base = (blockIdx.z / W_grid) * blockDim.y;
//     int w_base = (blockIdx.z % W_grid) * blockDim.x;



    
//     //  turn off threads outside the dimensions selected
    
//     //     parallel_for(n = 0; n < N; n++)
//     //     parallel_for (m = 0; m < M; m++)
//     //     parallel_for(h = 0; h < H_out; h++)
//     //     parallel_for(w = 0; w < W_out; w++) 
//         for (int c = 0; c < C; c++) {// SUM ACROSS ALL INPUT FEATURES
//             if( tx < K && ty < K){
//             kernelShared[ty*K+tx] = k4d(m, c, ty, tx );
//             }
//             __syncthreads();
//             for(int i = h; i< h_base+ inputTile;i += TILE_WIDTH){
//                 for(int j = w; j < w_base + inputTile; j += TILE_WIDTH){
//                     inputShared[(i-h_base)*inputTile + j-w_base] = x4d(n, c, i, j);
//                 }
//             }
//             __syncthreads();
//             for (int p = 0; p < K; p++) {  // LOOP THROUGH THE K BY K CONVOLUTION KERNEL
//                 for (int q = 0; q < K; q++) {
//                     acc +=  x4d(n, c, h + p, w + q) *k4d(m, c, p, q);//inputShared[(ty+p)*inputTile+tx+q] * kernelShared[p*K+q] ;//  // *
//                 }
//             }
//            __syncthreads();
            
//         }
//     if( h < H_out && w < W_out) {
//         y4d(n, m, h, w) = acc;
//     }

// #undef y4d
// #undef x4d
// #undef k4d
// }


//////////////////////////////////
///Shared Unrolling + Constant Memory
/////////////////////////////////

// __constant__ float deviceK[maskWidth*maskWidth*64];
// __global__ void conv_forward_kernel(float *__restrict__ y, const float *__restrict__ x, const float *__restrict__ k, const int B, const int M, const int C,
//                                     const int H, const int W, const int K) {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     y - output
//     x - input
//     k - kernel
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int H_out = H - 7 + 1;
//     const int W_out = W - 7 + 1;
//     //(void)H_out;  // silence declared but never referenced warning. remove this line when you start working
//     //(void)W_out;  // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = y4d(0,0,0,0)
//     // y4d(0,0,0,0) = a

// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// //#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
// #define k4d(i3, i2, i1, i0) deviceK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

// const int W_grid = ceil(1.0*W_out/TILE_WIDTH);
// int inputTile = TILE_WIDTH + 7 - 1; 
// extern __shared__ float sharedMem[];
// float* inputShared = &sharedMem[0]; 
// float* kernelShared = &sharedMem[inputTile*inputTile];

//     // Insert your GPU convolution kernel code here
//     // int H_out = H – K + 1;
//     // int W_out = W – K + 1;

//     int n = blockIdx.x;                                   // SAMPLES IN MINI BATCH
//     int m = blockIdx.y;                                   // OUTPUT FEATURE MAP
//     int h = (blockIdx.z / W_grid) * TILE_WIDTH+ threadIdx.y;  // HEIGHT OF THE GRID
//     int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;  // WIDTH OF THE GRID
//     float acc = 0;
//     int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
//     int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
//     int tx = threadIdx.y;
//     int ty = threadIdx.x;


    

//         for (int c = 0; c < C; c++) {// SUM ACROSS ALL INPUT FEATURES
//              if( tx < 7 && ty < 7){
//              kernelShared[tx*7+ty] = k4d(m, c, tx, ty );
//              }
//              __syncthreads();
//             for(int i = h; i< h_base+ inputTile;i += TILE_WIDTH){
//                 for(int j = w; j < w_base + inputTile; j += TILE_WIDTH){
//                     inputShared[(i-h_base)*inputTile + j-w_base] = x4d(n, c, i, j);
//                 }
//             }
//             __syncthreads();
//             //for (int p = 0; p < K; p++) {  // LOOP THROUGH THE K BY K CONVOLUTION KERNEL
//             //    for (int q = 0; q < K; q++) {
//                     acc +=  inputShared[(tx+0)*inputTile+ty+0] *kernelShared[0*7+0]; //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc +=  inputShared[(tx+0)*inputTile+ty+1] *kernelShared[0*7+1];
//                     acc +=  inputShared[(tx+0)*inputTile+ty+2] *kernelShared[0*7+2];
//                     acc +=  inputShared[(tx+0)*inputTile+ty+3] *kernelShared[0*7+3];
//                     acc +=  inputShared[(tx+0)*inputTile+ty+4] *kernelShared[0*7+4];
//                     acc +=  inputShared[(tx+0)*inputTile+ty+5] *kernelShared[0*7+5];
//                     acc +=  inputShared[(tx+0)*inputTile+ty+6] *kernelShared[0*7+6];

//                     acc +=  inputShared[(tx+1)*inputTile+ty+0] *kernelShared[1*7+0]; //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc +=  inputShared[(tx+1)*inputTile+ty+1] *kernelShared[1*7+1];
//                     acc +=  inputShared[(tx+1)*inputTile+ty+2] *kernelShared[1*7+2];
//                     acc +=  inputShared[(tx+1)*inputTile+ty+3] *kernelShared[1*7+3];
//                     acc +=  inputShared[(tx+1)*inputTile+ty+4] *kernelShared[1*7+4];
//                     acc +=  inputShared[(tx+1)*inputTile+ty+5] *kernelShared[1*7+5];
//                     acc +=  inputShared[(tx+1)*inputTile+ty+6] *kernelShared[1*7+6];

//                     acc +=  inputShared[(tx+2)*inputTile+ty+0] *kernelShared[2*7+0]; //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc +=  inputShared[(tx+2)*inputTile+ty+1] *kernelShared[2*7+1];
//                     acc +=  inputShared[(tx+2)*inputTile+ty+2] *kernelShared[2*7+2];
//                     acc +=  inputShared[(tx+2)*inputTile+ty+3] *kernelShared[2*7+3];
//                     acc +=  inputShared[(tx+2)*inputTile+ty+4] *kernelShared[2*7+4];
//                     acc +=  inputShared[(tx+2)*inputTile+ty+5] *kernelShared[2*7+5];
//                     acc +=  inputShared[(tx+2)*inputTile+ty+6] *kernelShared[2*7+6];

//                     acc +=  inputShared[(tx+3)*inputTile+ty+0] *kernelShared[3*7+0]; //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc +=  inputShared[(tx+3)*inputTile+ty+1] *kernelShared[3*7+1];
//                     acc +=  inputShared[(tx+3)*inputTile+ty+2] *kernelShared[3*7+2];
//                     acc +=  inputShared[(tx+3)*inputTile+ty+3] *kernelShared[3*7+3];
//                     acc +=  inputShared[(tx+3)*inputTile+ty+4] *kernelShared[3*7+4];
//                     acc +=  inputShared[(tx+3)*inputTile+ty+5] *kernelShared[3*7+5];
//                     acc +=  inputShared[(tx+3)*inputTile+ty+6] *kernelShared[3*7+6];

//                     acc +=  inputShared[(tx+4)*inputTile+ty+0] *kernelShared[4*7+0]; //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc +=  inputShared[(tx+4)*inputTile+ty+1] *kernelShared[4*7+1];
//                     acc +=  inputShared[(tx+4)*inputTile+ty+2] *kernelShared[4*7+2];
//                     acc +=  inputShared[(tx+4)*inputTile+ty+3] *kernelShared[4*7+3];
//                     acc +=  inputShared[(tx+4)*inputTile+ty+4] *kernelShared[4*7+4];
//                     acc +=  inputShared[(tx+4)*inputTile+ty+5] *kernelShared[4*7+5];
//                     acc +=  inputShared[(tx+4)*inputTile+ty+6] *kernelShared[4*7+6];

//                     acc +=  inputShared[(tx+5)*inputTile+ty+0] *kernelShared[5*7+0]; //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc +=  inputShared[(tx+5)*inputTile+ty+1] *kernelShared[5*7+1];
//                     acc +=  inputShared[(tx+5)*inputTile+ty+2] *kernelShared[5*7+2];
//                     acc +=  inputShared[(tx+5)*inputTile+ty+3] *kernelShared[5*7+3];
//                     acc +=  inputShared[(tx+5)*inputTile+ty+4] *kernelShared[5*7+4];
//                     acc +=  inputShared[(tx+5)*inputTile+ty+5] *kernelShared[5*7+5];
//                     acc +=  inputShared[(tx+5)*inputTile+ty+6] *kernelShared[5*7+6];

//                     acc +=  inputShared[(tx+6)*inputTile+ty+0] *kernelShared[6*7+0]; //x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
//                     acc +=  inputShared[(tx+6)*inputTile+ty+1] *kernelShared[6*7+1];
//                     acc +=  inputShared[(tx+6)*inputTile+ty+2] *kernelShared[6*7+2];
//                     acc +=  inputShared[(tx+6)*inputTile+ty+3] *kernelShared[6*7+3];
//                     acc +=  inputShared[(tx+6)*inputTile+ty+4] *kernelShared[6*7+4];
//                     acc +=  inputShared[(tx+6)*inputTile+ty+5] *kernelShared[6*7+5];
//                     acc +=  inputShared[(tx+6)*inputTile+ty+6] *kernelShared[6*7+6];
//                 //}
//             //}
//         //__syncthreads();   
//         }
//     if( h < H_out && w < W_out) {
//         y4d(n, m, h, w) = acc;
//     }

// #undef y4d
// #undef x4d
// #undef k4d
// }

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k,
                                                    float **device_y_ptr, float **device_x_ptr, float **device_k_ptr,
                                                    const int B, const int M, const int C, const int H, const int W,
                                                    const int K) {
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    /*
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    #define yElements B *M *W_out *H_out
    #define xElements B *C *H *W
    #define kElements K *K *M *C

    
    cudaMalloc((void **)device_y_ptr, yElements * sizeof(float));
    cudaMalloc((void **)device_x_ptr, xElements * sizeof(float));
    cudaMalloc((void **)device_k_ptr, kElements * sizeof(float));
    // cudaMallocManaged((void **)device_y_ptr, yElements * sizeof(float));
    // cudaMallocManaged((void **)device_x_ptr, xElements * sizeof(float));
    // cudaMallocManaged((void **)device_k_ptr, kElements * sizeof(float));
    // cudaDeviceSynchronize();




    //cudaMemPrefetchAsync((*device_x_ptr), xElements * sizeof(float), 0, 0);

    //cudaMemcpy((*device_y_ptr), host_y, yElements * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpyAsync((*device_x_ptr), host_x, xElements * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpyAsync((*device_k_ptr), host_k, kElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((*device_x_ptr), host_x, xElements * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy((*device_k_ptr), host_k, kElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceK,host_k,kElements * sizeof(float), 0 ,cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B,
                                             const int M, const int C, const int H, const int W, const int K) {

    // const int H_out = H - K + 1;
    // const int W_out = W - K + 1;
    // __half* y_half;
    // __half* x_half;
    // __half* k_half;
    // cudaMalloc((void**) &y_half, sizeof(__half)*yElements);
    // cudaMalloc((void**) &x_half, sizeof(__half)*xElements);
    // cudaMalloc((void**) &k_half, sizeof(__half)*kElements);

    // float2halfK<<<ceil(1.0*xElements/(TILE_WIDTH*TILE_WIDTH)), TILE_WIDTH*TILE_WIDTH>>>(x_half, device_x, xElements);
    // float2halfK<<<ceil(1.0*kElements/TILE_WIDTH), TILE_WIDTH>>>(k_half, device_k, kElements);


const int Z = ceil(1.0 * H / TILE_WIDTH) * ceil(1.0 * W / TILE_WIDTH);
    // Set the kernel dimensions and call the kernel
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);
    //size_t sharedMemSize = sizeof(float) * ((TILE_WIDTH + K - 1)* (TILE_WIDTH + K - 1) + K*K);
    //conv_forward_kernel<<<gridDim, blockDim,sharedMemSize>>>(y_half, x_half, k_half, B, M, C, H, W, K);
    // conv_forward_kernel<<<gridDim, blockDim>>>(y_half, x_half, k_half, B, M, C, H, W, K);


    conv_forward_kernel<<<gridDim, blockDim>>>(device_y, device_x, device_k, B, M, C, H, W, K);


    // #define numElements B *M *H_out *W_out
    // half2floatK<<<ceil(1.0*numElements/(TILE_WIDTH*TILE_WIDTH)), TILE_WIDTH*TILE_WIDTH>>>(device_y, y_half, numElements);
    
    
    

    // cudaFree(y_half);
    // cudaFree(x_half);
    // cudaFree(k_half);
   
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k,
                                                    const int B, const int M, const int C, const int H, const int W,
                                                    const int K) {
// Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    #define numElements B *M *H_out *W_out
    cudaMemcpy(host_y, device_y, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpyAsync(host_y, device_y, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);

}

__host__ void GPUInterface::get_device_properties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1]
                  << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1]
                  << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}
