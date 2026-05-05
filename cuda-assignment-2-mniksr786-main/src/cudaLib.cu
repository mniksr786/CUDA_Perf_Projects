#include "cudaLib.cuh"
#include "cpuLib.cpp"
#include "cpuLib.h"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	return 0;
}

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 3.14159f;

	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}



/*
int runGpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args) {
	
	std::cout << "Lazy, you are! ... ";
	std::cout << "Filter pixels, you must! ... ";

	return 0;
}

int medianFilter_gpu (uint8_t* inPixels, ImageDim imgDim, 
	uint8_t* outPixels, MedianFilterArgs args) {

	return 0;
} */

//	STUDENT: Add functions here


int runGpuMedianFilter(std::string imgPath, std::string outPath, MedianFilterArgs args) { 

    ImageDim imgDim; 
    uint8_t *imgData, *d_inPixels, *d_outPixels; 

	auto tStart = std::chrono::high_resolution_clock::now();

    // Load image into CPU memory 
    int bytesRead = loadBytesImage(imgPath, imgDim, &imgData); 

    // Calculate image size considering pixel size 
    int imgSize = imgDim.height * imgDim.width * imgDim.channels * imgDim.pixelSize; 
    // std::cout << "Image Size = " << imgSize << " bytes\n"; 

    // Allocate device memory 
    cudaMalloc((void**)&d_inPixels, imgSize);
    cudaMalloc((void**)&d_outPixels, imgSize);

    // Copy data from H2D
    cudaMemcpy(d_inPixels, imgData, imgSize, cudaMemcpyHostToDevice);

    // Define block, grid size 
    dim3 blockDim(32, 32); 
    dim3 gridDim((imgDim.width + blockDim.x - 1) / blockDim.x, 
                 (imgDim.height + blockDim.y - 1) / blockDim.y); 

	// Call kernels with allocated shared memory
	size_t sharedMemSize = (blockDim.x + args.filterW - 1) * (blockDim.y + args.filterH - 1) * imgDim.channels * sizeof(uint8_t);   
    medianFilter_gpu<<<gridDim, blockDim, sharedMemSize>>>(d_inPixels, imgDim, d_outPixels, args); 

    // Synchronization to ensure all kernels finished
	cudaDeviceSynchronize(); 

    // Copy data from D2H
    cudaMemcpy(imgData, d_outPixels, imgSize, cudaMemcpyDeviceToHost); 

    // Save the output image 
    writeBytesImage(outPath, imgDim, imgData); 
    
    // Free CPU and GPU memory 
    cudaFree(d_inPixels); 
    cudaFree(d_outPixels); 
    free(imgData); 
    
	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "CUDA takes " << time_span.count() << " seconds.";

    return 0; 
} 


__global__ void medianFilter_gpu(uint8_t* inPixels, ImageDim imgDim, uint8_t* outPixels, MedianFilterArgs args) {   

    // Compute global pixel indices   
    int x = blockIdx.x * blockDim.x + threadIdx.x;   
    int y = blockIdx.y * blockDim.y + threadIdx.y;   
    int tx = threadIdx.x, ty = threadIdx.y;   

    // Filter parameters   
    int halfW = args.filterW / 2;   
    int halfH = args.filterH / 2;   

    // Max filter size (supports filter sizes up to 12x12)
    // Statically allocated
    uint8_t window[1024];   

    // Compute shared memory layout
    extern __shared__ uint8_t sharedMem[];   
    int sharedWidth = blockDim.x + args.filterW - 1;   
    int sharedHeight = blockDim.y + args.filterH - 1;   

    // Shared memory index   
    int s_x = tx + halfW;   
    int s_y = ty + halfH;

    // Global index for input/output pixels   
    int globalIndex = (y * imgDim.width + x) * imgDim.channels;   

    // Load pixels into shared memory (including halo region)   
    for (int dy = -halfH; dy <= halfH; dy++) {   
        for (int dx = -halfW; dx <= halfW; dx++) {   
            int load_x = x + dx;   
            int load_y = y + dy;   

            // Avoid out-of-bounds access   
            load_x = max(0, min(load_x, imgDim.width - 1));   
            load_y = max(0, min(load_y, imgDim.height - 1));   

            int globalLoadIndex = (load_y * imgDim.width + load_x) * imgDim.channels;   
            int sharedLoadIndex = ((ty + dy + halfH) * sharedWidth + (tx + dx + halfW)) * imgDim.channels;   

            if ((tx + dx + halfW) < sharedWidth && (ty + dy + halfH) < sharedHeight) {   
                for (int c = 0; c < imgDim.channels; c++) {   
                    sharedMem[sharedLoadIndex + c] = inPixels[globalLoadIndex + c];   
                }   
            }   
        }   
    }   
 
    // Ensure all threads finish loading into shared memory 
    __syncthreads();   

    // Apply median filter only within valid pixel bounds   
    if (x >= halfW && y >= halfH && x < imgDim.width - halfW && y < imgDim.height - halfH) {   
        for (int c = 0; c < imgDim.channels; c++) {   
            int count = 0;   

            // Extract the filter window   
            for (int dy = -halfH; dy <= halfH; dy++) {   
                for (int dx = -halfW; dx <= halfW; dx++) {   
                    int sx = s_x + dx;   
                    int sy = s_y + dy;   
                    if (sx >= 0 && sx < sharedWidth && sy >= 0 && sy < sharedHeight) {   
                        window[count++] = sharedMem[(sy * sharedWidth + sx) * imgDim.channels + c];   
                    }   
                }   
            }   

            // Sort and assign the median   
            sortPixels_gpu(window, count);   
            outPixels[globalIndex + c] = window[count / 2];   
        }   
    }   
}

__device__ void sortPixels_gpu(uint8_t *array, dim3 arrayDim) {   

    int size = arrayDim.x;   
    // Odd-even sort
    for (int i = 0; i < size; ++i) {   
        for (int j = (i % 2); j < size - 1; j += 2) {   
            if (array[j] > array[j + 1]) {   
                uint8_t temp = array[j];   
                array[j] = array[j + 1];   
                array[j + 1] = temp;   
            }   
        }   
    }   
}


__global__ void poolLayer_gpu(float *input, TensorShape inShape,
                               float *output, TensorShape outShape, PoolLayerArgs args) { 

    uint32_t poolH = args.poolH; 
    uint32_t poolW = args.poolW; 
    uint32_t strideH = args.strideH; 
    uint32_t strideW = args.strideW; 
    uint32_t padH = args.padH; 
    uint32_t padW = args.padW; 

    // Output coordinates for the current thread 
    uint32_t outRow = blockIdx.y * blockDim.y + threadIdx.y; 
    uint32_t outCol = blockIdx.x * blockDim.x + threadIdx.x; 
    uint32_t channel = blockIdx.z; 

    // Ensure output is within bounds 
    if (outRow < outShape.height && outCol < outShape.width) { 

        // Initialize the poolPick value for max pooling 
        float poolPick = -1e10; // Min value for max pooling 

        // Loop through the pooling window 
        for (uint32_t poolRow = 0; poolRow < poolH; ++poolRow) { 
            for (uint32_t poolCol = 0; poolCol < poolW; ++poolCol) { 
                // Compute the actual row and column in the input tensor 
                int row = outRow * strideH + poolRow - padH; 
                int col = outCol * strideW + poolCol - padW; 

                // Handle padding by skipping out-of-bound indices 
                if (row >= 0 && row < static_cast<int>(inShape.height) && 
                    col >= 0 && col < static_cast<int>(inShape.width)) { 
                    uint32_t inputIdx = (channel * inShape.height * inShape.width) + (row * inShape.width + col); 
                    float val = input[inputIdx]; 

                    // Perform max pooling 
                    poolPick = fmaxf(poolPick, val); 
                } 
            } 
        } 

        // Store the result in the output tensor 
        uint32_t outputIdx = (channel * outShape.height * outShape.width) + (outRow * outShape.width + outCol); 
        output[outputIdx] = poolPick; 
    } 
} 

  

int runGpuPool(TensorShape inShape, PoolLayerArgs poolArgs) { 

    TensorShape outShape; 
    outShape.height = (inShape.height + 2 * poolArgs.padH - poolArgs.poolH) / poolArgs.strideH + 1; 
    outShape.width = (inShape.width + 2 * poolArgs.padW - poolArgs.poolW) / poolArgs.strideW + 1; 
    outShape.channels = inShape.channels; 

    auto tStart = std::chrono::high_resolution_clock::now();
    
    // Allocate memory for input and output tensors 
    float *h_input = new float[inShape.height * inShape.width * inShape.channels];
    float *h_output = new float[outShape.height * outShape.width * outShape.channels]; 
    
    // Initialize input tensor with random values 
    for (uint32_t i = 0; i < inShape.height * inShape.width * inShape.channels; i++) { 
        h_input[i] = static_cast<float>((rand() % 256) + 1);  // Random values between 0 and 255 
    } 

    // Print input tensor 
    /* std::cout << "Input Tensor:\n"; 
    for (uint32_t c = 0; c < inShape.channels; c++) { 
        std::cout << "Channel " << c << ":\n"; 
        for (uint32_t i = 0; i < inShape.height; i++) { 
            for (uint32_t j = 0; j < inShape.width; j++) { 
                float val = h_input[(c * inShape.height * inShape.width) + (i * inShape.width + j)]; 
                std::cout << val << " "; 
            } 
            std::cout << "\n"; 
        } 
    } */

    float *d_input, *d_output; 

    // Allocate device memory 
    cudaMalloc(&d_input, inShape.height * inShape.width * inShape.channels * sizeof(float)); 
    cudaMalloc(&d_output, outShape.height * outShape.width * outShape.channels * sizeof(float)); 

    // Copy input data to device 
    cudaMemcpy(d_input, h_input, inShape.height * inShape.width * inShape.channels * sizeof(float), cudaMemcpyHostToDevice); 

    // Define grid and block dimensions 
    dim3 blockSize(32, 32); 
    dim3 gridSize((outShape.width + blockSize.x - 1) / blockSize.x,  
                  (outShape.height + blockSize.y - 1) / blockSize.y, inShape.channels); 

    // Launch kernel 
    poolLayer_gpu<<<gridSize, blockSize>>>(d_input, inShape, d_output, outShape, poolArgs); 

    // Synchronize to ensure kernel execution is complete 
    cudaDeviceSynchronize(); 

    // Copy result back to host 
    cudaMemcpy(h_output, d_output, outShape.height * outShape.width * outShape.channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output tensor 
    /* std::cout << "\nOutput Tensor (Max Pooled):\n"; 
    for (uint32_t c = 0; c < outShape.channels; c++) { 
        std::cout << "Channel " << c << ":\n"; 
        for (uint32_t i = 0; i < outShape.height; i++) { 
            for (uint32_t j = 0; j < outShape.width; j++) { 
                float val = h_output[(c * outShape.height * outShape.width) + (i * outShape.width + j)]; 
                std::cout << val << " "; 
            } 
            std::cout << "\n"; 
        } 
    } */

    // Free memory 
    cudaFree(d_input); 
    cudaFree(d_output); 
    delete[] h_input; 
    delete[] h_output; 

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "GPU takes " << time_span.count() << " seconds."; 

    return 0; 
}
