#include <iostream>
#include <cuda.h>
#include <vector>

#define warpSize 32
#define len 2046
#define descriptor(x,y) descriptor[3*x+y]
using namespace std;

__global__ void print(int* src, int* blockCounter, int* descriptor) {
    __shared__ int bid_s;
    __shared__ int shared[32];
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
        int val = tx < len ? src[idx] : 0;
        for (int i = 1; i < warpSize; i *= 2) {
            int tmp = __shfl_up_sync(0xffffffff, val, i);
            if (threadIdx.x >= i) val += tmp;
        }

        if (threadIdx.x == warpSize-1) {
            shared[threadIdx.y] = val;
        }

        __syncthreads();

        if (threadIdx.y == 0) {
            val = shared[threadIdx.x];
            for (int i = 1; i < warpSize; i *= 2) {
                int tmp = __shfl_up_sync(0xffffffff, val, i);
                if (threadIdx.x >= i) val += tmp;
            }
            shared[threadIdx.x] = val;
        }
        //if (threadIdx.y == 0) printf("%d %d\n", tx+j, shared[threadIdx.x]);


        if (threadIdx.x == 0 && threadIdx.y == 0) {
            // V -> A
            descriptor(bid,0) = 1;
            descriptor(bid,1) = shared[warpSize-1];
            // look back phase
            int runningSum = shared[warpSize-1];
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
        }
    }
}

int main() {
    vector<int> src_h(len);
    for (int i = 0; i < len; i++) {
        src_h[i] = 1;
    }
    int* src_d;
    int* out_d;

    int* blockCounter;
    int initBlockId = 0;

    int numState = 3;
    int numBlock = 2;
    int* descriptor;

    cudaMalloc((void**)&descriptor, numState*numBlock);
    cudaMalloc((void**)&blockCounter, sizeof(int));
    cudaMalloc((void**)&out_d, sizeof(int));
    cudaMalloc((void**)&src_d, sizeof(int)*len);

    cudaMemcpy(blockCounter, &initBlockId, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(src_d, src_h.data(), sizeof(int)*len, cudaMemcpyHostToDevice);

    dim3 threads(32,32,1);
    dim3 blocks(2);

    print<<<blocks,threads>>>(src_d, blockCounter, descriptor);
    cudaDeviceSynchronize();
    return 0;
}
