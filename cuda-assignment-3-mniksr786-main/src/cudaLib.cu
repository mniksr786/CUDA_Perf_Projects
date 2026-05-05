
#include "cudaLib.cuh"
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




int runGpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args) {
	
	std::cout << "Lazy, you are! ... ";
	std::cout << "Filter pixels, you must! ... ";

	return 0;
}

int medianFilter_gpu (uint8_t inPixels, ImageDim imgDim, 
	uint8_t outPixels, MedianFilterArgs args) {

	return 0;
}


__device__ float gpu_Activation(float x) {   
    return max(0.0f, x); // ReLU activation   

}   

// CUDA CONV kernel   

__global__ void convKernel(float* input, TensorShape iShape, float* filter, TensorShape fShape,  
                           float* bias, float* output, TensorShape oShape, ConvLayerArgs args) {   

    int n = blockIdx.x / (oShape.channels * oShape.height * oShape.width);  //Batch index   
    int m = (blockIdx.x % (oShape.channels * oShape.height * oShape.width)) / (oShape.height * oShape.width);  // Output channel   
    int x = (blockIdx.x % (oShape.height * oShape.width)) / oShape.width;  // Output height index   
    int y = blockIdx.x % oShape.width;  // Output width index   

  
    extern __shared__ float sharedFilter[]; // Shared memory for filter   

    //Load filter into shared memory   
    int threadId = threadIdx.x;   
    while (threadId < fShape.height * fShape.width * fShape.channels) {   
        sharedFilter[threadId] = filter[m * fShape.channels * fShape.height * fShape.width + threadId];   
        threadId += blockDim.x;   

    }   
    
    __syncthreads();   

    //Initialize sum with bias   
    float sum = bias[m];   

    //Convolution operation   
    for (int i = 0; i < fShape.height; ++i) {   
        for (int j = 0; j < fShape.width; ++j) {   
            for (int k = 0; k < fShape.channels; ++k) {   
                int in_x = args.strideH * x + i - args.padH;   
                int in_y = args.strideW * y + j - args.padW;   

                if (in_x >= 0 && in_x < iShape.height && in_y >= 0 && in_y < iShape.width) {   
                    sum += input[n * iShape.channels * iShape.height * iShape.width +  //Account for batch index   
                                 k * iShape.height * iShape.width +   
                                 in_x * iShape.width + in_y] *   
                           sharedFilter[k * fShape.height * fShape.width + i * fShape.width + j];   
                }   
            }   
        }   
    }   

    //Apply activation if enabled   
    if (args.activation) {   
        sum = gpu_Activation(sum);   
    }   

    // Store output   
    output[n * oShape.channels * oShape.height * oShape.width +   
           m * oShape.height * oShape.width +   
           x * oShape.width + y] = sum;   

}   


extern int convLayer_gpu_kernel(float* input, TensorShape iShape, float* filter, TensorShape fShape,  
                                float* bias, float* output, TensorShape oShape, ConvLayerArgs& args) {   

    dim3 blockDim(256, 1, 1);   

    // gridDim covers batch size   
    dim3 gridDim(oShape.count * oShape.channels * oShape.height * oShape.width);   
    size_t sharedMemSize = fShape.height * fShape.width * fShape.channels * sizeof(float);   

    convKernel<<<gridDim, blockDim, sharedMemSize>>>(input, iShape, filter, fShape, bias, output, oShape, args);   

    cudaDeviceSynchronize();   

    return 0;   

}   

  

extern uint64_t evaluateGpuConv(TensorShape iShape, TensorShape fShape,
                          TensorShape& oShape, ConvLayerArgs args) {   

    uint64_t errorCount = 0;   

    oShape.height = (iShape.height + 2 * args.padH - fShape.height) / args.strideH + 1;   
    oShape.width = (iShape.width + 2 * args.padW - fShape.width) / args.strideW + 1;   
    oShape.channels = fShape.count;   
    oShape.count = iShape.count; // Set batch size   

    float* h_in = nullptr;   
    float* h_filter = nullptr;   
    float* h_bias = nullptr;   
    float* h_out_cpu = nullptr;   
    float* h_out_gpu = nullptr;   

    int retVal;   
    retVal = makeTensor(&h_in, iShape);   
    if (retVal != 0) return -1;   

    retVal = makeTensor(&h_filter, fShape);   
    if (retVal != 0) return -1;   

    retVal = makeVector(&h_bias, oShape.channels);   
    if (retVal != 0) return -1;   

    h_out_cpu = (float*)malloc(tensorSize(oShape) * sizeof(float));   
    h_out_gpu = (float*)malloc(tensorSize(oShape) * sizeof(float));   

    float* d_in;   
    float* d_filter;   
    float* d_bias;   
    float* d_out;   

    cudaMalloc(&d_in, tensorSize(iShape) * sizeof(float));   
    cudaMalloc(&d_filter, tensorSize(fShape) * sizeof(float));   
    cudaMalloc(&d_bias, oShape.channels * sizeof(float));   
    cudaMalloc(&d_out, tensorSize(oShape) * sizeof(float));
      
    cudaMemcpy(d_in, h_in, tensorSize(iShape) * sizeof(float), cudaMemcpyHostToDevice);   
    cudaMemcpy(d_filter, h_filter, tensorSize(fShape) * sizeof(float), cudaMemcpyHostToDevice);   
    cudaMemcpy(d_bias, h_bias, oShape.channels * sizeof(float), cudaMemcpyHostToDevice);
       
    // CONV Kernel launch
    convLayer_gpu_kernel(d_in, iShape, d_filter, fShape, d_bias, d_out, oShape, args);
       
    cudaMemcpy(h_out_gpu, d_out, tensorSize(oShape) * sizeof(float), cudaMemcpyDeviceToHost);   
 

#ifndef CONV_CHECK_DISABLE
    convLayer_cpu(h_in, iShape, h_filter, fShape, h_bias, h_out_cpu, oShape, args);   
    for (uint64_t i = 0; i < tensorSize(oShape); ++i) {   
        float delta = fabs(h_out_cpu[i] - h_out_gpu[i]);   
        printf("index %llu: CPU = %f, GPU = %f, delta = %f\n", i, h_out_cpu[i], h_out_gpu[i], delta);   

        if (fabs(h_out_cpu[i] - h_out_gpu[i]) > 1.0f) { // Fix threshold comparison   
            errorCount++;   
        }   
    }   

#endif   

    free(h_in);   
    free(h_filter);   
    free(h_bias);   
    free(h_out_cpu);   
    free(h_out_gpu);   

    cudaFree(d_in);   
    cudaFree(d_filter);   
    cudaFree(d_bias);   
    cudaFree(d_out);   

    return errorCount;   

}   

  

extern int runGpuConv(int argc, char** argv) {   
    TensorShape iShape = AlexL1_InShape;   
    TensorShape fShape = AlexL1_FilterShape;   
    ConvLayerArgs convArgs = AlexL1_ConvArgs;   

    std::cout << "Evaluate convolution:\n";   
    std::cout << "Input: " << iShape << " \n";   
    std::cout << "Filter: " << fShape << " \n";   

    TensorShape oShape;
    
    auto tStart = std::chrono::high_resolution_clock::now();

    uint64_t errorCount = evaluateGpuConv(iShape, fShape, oShape, convArgs);
    
	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "GPU CONV " << time_span.count() << " seconds.";
       
    std::cout << "Found " << errorCount << " / " << tensorSize(oShape) << " errors\n";   
    return 0;   

}  


// GEMM using shared memory 
__global__ void gemmKernel(float *a, float *b, float *c, int M, int N, int K) {   

    // Declare shared memory as 2D arrays 
    __shared__ float Asub[TILE_SIZE][TILE_SIZE+1];   
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE+1];   

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;   

    float sum = 0.0f;   

    // Loop over the tiles of A and B   
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) { 
		  
        // Load A submatrix into shared memory   
        if (row < M && t * TILE_SIZE + threadIdx.x < K)   
            Asub[threadIdx.y][threadIdx.x] = a[row * K + t * TILE_SIZE + threadIdx.x];   
        else   
            Asub[threadIdx.y][threadIdx.x] = 0.0f;   

        // Load B submatrix into shared memory   
        if (col < N && t * TILE_SIZE + threadIdx.y < K)   
            Bsub[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * N + col];   
        else   
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;   

        // Synchronize threads to ensure that the submatrices are loaded into shared memory   
        __syncthreads();   

        // Perform the matrix multiplication (sum += A * B)   
        for (int k = 0; k < TILE_SIZE; ++k) {   
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];   
        }   

        // Synchronize threads before loading the next tile   
        __syncthreads();   
    }   


    // Write the result to the global memory   
    if (row < M && col < N) {   
        c[row * N + col] = sum;   
    }   
} 

 
int gemmLayer_gpu(float *a, TensorShape aShape, float *b, TensorShape bShape, float *c, TensorShape cShape, GemmLayerArgs args) { 
    int M = aShape.height; 
    int K = aShape.width; 
    int N = bShape.width; 

    float *d_a, *d_b, *d_c; 

    size_t sizeA = M * K * sizeof(float); 
    size_t sizeB = K * N * sizeof(float); 
    size_t sizeC = M * N * sizeof(float);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if (USE_UNIFIED_MEMORY) { 
        cudaMallocManaged(&d_a, sizeA); 
        cudaMallocManaged(&d_b, sizeB); 
        cudaMallocManaged(&d_c, sizeC); 

        cudaMemcpyAsync(d_a, a, sizeA, cudaMemcpyHostToDevice, stream); 
        cudaMemcpyAsync(d_b, b, sizeB, cudaMemcpyHostToDevice, stream); 

        cudaMemPrefetchAsync(d_a, sizeA, cudaCpuDeviceId); 
        cudaMemPrefetchAsync(d_b, sizeB, cudaCpuDeviceId); 
        cudaMemPrefetchAsync(d_c, sizeC, cudaCpuDeviceId); 

    } else { 
        cudaMalloc(&d_a, sizeA); 
        cudaMalloc(&d_b, sizeB); 
        cudaMalloc(&d_c, sizeC); 
        cudaMemcpyAsync(d_a, a, sizeA, cudaMemcpyHostToDevice, stream); 
        cudaMemcpyAsync(d_b, b, sizeB, cudaMemcpyHostToDevice, stream); 
    } 


    dim3 blockSize(TILE_SIZE, TILE_SIZE); 
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE); 

    // Calculate the size of shared memory 
    size_t sharedMemSize = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);  // For both Asub and Bsub 

    // Launch the kernel 
    gemmKernel<<<gridSize, blockSize, sharedMemSize>>>(d_a, d_b, d_c, M, N, K); 

    // Synchronize
    cudaDeviceSynchronize();
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    if (CUDA_ERR_CHECK) {
		// Check for kernel launch errors 
		cudaError_t err = cudaGetLastError(); 
		if (err != cudaSuccess) { 
			std::cerr << "CUDA error during kernel launch: " << cudaGetErrorString(err) << std::endl; 
			return -1; 
		} 			
	}
    
    if (CUDA_ERR_CHECK) {
	    cudaError_t err = cudaGetLastError(); 
		if (err != cudaSuccess) { 
			std::cerr << "CUDA error after kernel execution: " << cudaGetErrorString(err) << std::endl; 
			return -1; 
		} 		
	}

    cudaMemcpy(c, d_c, sizeC, cudaMemcpyDeviceToHost); 
 
    if (VALIDATE_GEMM) {
		// Validate results for any matrix size by checking a random chunk of values 
		float *cpu_result = (float*) malloc(M * N * sizeof(float)); 
		gemmCpu(a, b, cpu_result, M, N, K); 

		// Compare random chunks of the results 
		if (compareMatricesChunk(cpu_result, c, M, N)) { 
			std::cout << "GPU result matches CPU result for the random chunk!" << std::endl; 
		} else { 
			std::cout << "Results mismatch in the random chunk!" << std::endl; 
		}
		free(cpu_result); 
	}

    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c); 

    return 0; 
} 

// Run GPU GEMM 
int runGpuGemm(int argc, char **argv) {
	
	uint32_t M = batchSize;
	uint32_t K = 4096;
	uint32_t N = 4096;
	 
    TensorShape aShape = {1, 1, M, K}; 
    TensorShape bShape = {1, 1, K, N}; 
    TensorShape cShape = {1, 1, M, N}; 
    GemmLayerArgs args = {TILE_SIZE, TILE_SIZE, 1};
    
    auto tStart = std::chrono::high_resolution_clock::now();
     
    float *a = nullptr;
    float *b = nullptr;
    float *c = (float*) malloc(M * N * sizeof(float));

	makeTensor(& a, aShape);
	makeTensor(& b, bShape);

    //for (int i = 0; i < M * K; ++i) a[i] = static_cast<float>(rand()) / RAND_MAX; 
    //for (int i = 0; i < K * N; ++i) b[i] = static_cast<float>(rand()) / RAND_MAX; 

    gemmLayer_gpu(a, aShape, b, bShape, c, cShape, args); 

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "GPU GEMM " << time_span.count() << " seconds.";
    
    free(a); 
    free(b); 
    free(c); 

    return 0; 
} 

  

// Evaluate GPU GEMM 

int evaluateGpuGemm() { 

    return runGpuGemm(0, nullptr); 

} 

// Function to compare two matrices for a small chunk 
bool compareMatricesChunk(float* A, float* B, int M, int N) {
	
	int chunk_size = 40; 
    srand(time(0));  // Seed random number generator for random chunk selection 

    for (int i = 0; i < chunk_size; ++i) { 
        int idx = rand() % (M * N);  // Random index in the matrix 
        int row = idx / N;           // Calculate row index 
        int col = idx % N;           // Calculate column index 

        if (std::fabs(A[idx] - B[idx]) > 1e-2) { 
            std::cout << "Mismatch at index (" << row << ", " << col << "): " 
                      << A[idx] << " vs " << B[idx] << std::endl; 
            return false; 
        } 
    } 
    return true; 
} 

// CPU implementation of GEMM for validation 
void gemmCpu(float* a, float* b, float* c, int M, int N, int K) { 
    for (int i = 0; i < M; ++i) { 
        for (int j = 0; j < N; ++j) { 
            float sum = 0.0f; 
            for (int k = 0; k < K; ++k) { 
                sum += a[i * K + k] * b[k * N + j]; 
            } 
            c[i * N + j] = sum; 
        } 
    } 
} 
