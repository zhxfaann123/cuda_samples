#include <iostream>
#include <cuda.h>
#include <vector>

#define warpSize 32
#define len 256
#define descriptor(x,y) descriptor[3*x+y]
using namespace std;

__global__ void print(int* src, int* dst, int* blockCounter, int* descriptor) {
    __shared__ int bid_s;
    __shared__ int shared[32];
    __shared__ int runningSum;
    int numThreads = blockDim.x * blockDim.y;
    int numBlock = gridDim.x;
    for (int j = 0; j < len; j += numThreads * numBlock) {
        // dynamic blockId assign
        if (threadIdx.y == 0 && threadIdx.x == 0) {
            bid_s = atomicAdd(&blockCounter[0], 1);
        }
        __syncthreads();
        int bid = bid_s;
        // init descrpt
        int tx = bid * numThreads + threadIdx.y * warpSize + threadIdx.x;
        descriptor(bid, 0) = 0; // Invilid: 0, Aggre : 1, prefix :2
        // aggregate
        int idx = tx + j;
        int val = idx < len ? src[idx] : 0;
        for (int i = 1; i < warpSize; i *= 2) {
            int tmp = __shfl_up_sync(0xffffffff, val, i);
            if (threadIdx.x >= i) val += tmp;
        }

        if (threadIdx.x == warpSize-1) {
            shared[threadIdx.y] = val;
        }

        __syncthreads();

        if (threadIdx.y == 0) {
            int val2 = shared[threadIdx.x];
            for (int i = 1; i < warpSize; i *= 2) {
                int tmp = __shfl_up_sync(0xffffffff, val2, i);
                if (threadIdx.x >= i) val2 += tmp;
            }
            shared[threadIdx.x] = val2;
        }
        //if (threadIdx.y == 0) printf("%d %d\n", tx+j, shared[threadIdx.x]);

        runningSum = 0;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            // V -> A
            descriptor(bid,0) = 1;
            descriptor(bid,1) = shared[warpSize-1];
            // look back phase

            for (int i = bid-1; i >= 0; i--) {
                while(descriptor(i,0) == 0) {}
                // A -> P
                if (descriptor(i,0) == 2) {
                    runningSum += descriptor(i,2);
                    descriptor(bid,0) = 2;
                    descriptor(bid,2) = runningSum;
                    break;
                // A -> A
                } else {
                    runningSum += descriptor(i,1);
                }
            }
            //printf("%d \n", runningSum);
        }
        __syncthreads();


        // distribute sum
        if (threadIdx.y > 0) {
            dst[idx] = runningSum + val + shared[threadIdx.y-1];
        } else {
            dst[idx] = runningSum + val;
        }

        printf("%d \n", dst[idx]);
    }
}

int main() {
    vector<int> src_h(len);
    vector<int> dst_h(len);
    for (int i = 0; i < len; i++) {
        src_h[i] = 1;
    }
    int* src_d;
    int* dst_d;

    int* blockCounter;
    int initBlockId = 0;

    int numState = 3;
    int numBlock = 2;
    int* descriptor;

    cudaMalloc((void**)&descriptor, numState*numBlock);
    cudaMalloc((void**)&blockCounter, sizeof(int));
    cudaMalloc((void**)&dst_d, sizeof(int)*len);
    cudaMalloc((void**)&src_d, sizeof(int)*len);

    cudaMemcpy(blockCounter, &initBlockId, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(src_d, src_h.data(), sizeof(int)*len, cudaMemcpyHostToDevice);
    cudaMemcpy(dst_h.data(), dst_d, sizeof(int)*len, cudaMemcpyDeviceToHost);

    dim3 threads(32,2,1);
    dim3 blocks(2);

    print<<<blocks,threads>>>(src_d, dst_d, blockCounter, descriptor);
    cudaDeviceSynchronize();

    //for (int i = 0; i < len; i++) {
    //    cout << dst_h[i] << endl;
    //}
    return 0;
}
