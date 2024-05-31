// 本文件定义每层的具体操作，使用cuda编程实现，包括使用shared memory和不使用的两个版本

#include "init.cu"

using  namespace std;

/*********************** conv&relu ***********************/
// conv和cuda_conv_relu为使用shared memory的cuda加速函数
__global__ static void conv(float *input, float *output, float *weights, int *input_channel, int *out_height){
    int tid = threadIdx.x; // input_channel
    int bid = blockIdx.x; // output_channel
    int ichannel = *input_channel;
    int iheight = *out_height+2;
    int oheight = *out_height;

    __shared__ float shared[512][3][3];
    for(int m=0; m<oheight; m++){
        for(int n=0; n<oheight; n++){
            __syncthreads();
            for(int f=0; f<3; f++){
                for(int w=0; w<3; w++){
                    shared[tid][f][w] = \
                    input[tid*iheight*iheight + (m+f)*iheight + (n+w)] * \
                    weights[bid*ichannel*3*3 + tid*3*3 + f*3 + w];
                }
            }
            __syncthreads();
            if(tid == 0){
                output[bid*(oheight*oheight) + m*oheight + n] = 0;
                for(int i=0; i<ichannel; i++){
                     for(int f=0; f<3; f++){
                        for(int w=0; w<3; w++){
                            output[bid*(oheight*oheight) + m*oheight + n] += shared[i][f][w];
                        }
                    }
                }
            }
        }
    }
}

void cuda_conv_relu(int layer_id, float *input, float *output, int input_channel, int output_channel, int out_height){
    int input_size = input_channel*(out_height+2)*(out_height+2);
    int output_size = output_channel*out_height*out_height;
    int weights_size = 512*512*3*3;

    float *gpu_input, *gpu_weights, *gpu_output;
    int *gpu_input_channel, *gpu_out_height;

    cudaMalloc((void**)&gpu_input, sizeof(float)*input_size);
    cudaMalloc((void**)&gpu_output, sizeof(float)*output_size);
    cudaMalloc((void**)&gpu_weights, sizeof(float)*weights_size);
    cudaMalloc((void**)&gpu_input_channel, sizeof(int));
    cudaMalloc((void**)&gpu_out_height, sizeof(int));

    cudaMemcpy(gpu_input, input, sizeof(float)*input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_weights, conv_weights[layer_id-1], sizeof(float)*weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_input_channel, &input_channel, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_out_height, &out_height, sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(output_channel);
    dim3 dimBlock(input_channel);

    conv<<<dimGrid, dimBlock>>>(gpu_input, gpu_output, gpu_weights, gpu_input_channel, gpu_out_height);

    cudaMemcpy(output, gpu_output, sizeof(float)*output_size, cudaMemcpyDeviceToHost);

    for(int i=0; i<output_channel; i++){
        for(int m=0; m<out_height; m++){
            for(int n=0; n<out_height; n++){
                output[i*out_height*out_height + m*out_height + n] += conv_bias[layer_id-1][i];
                if(output[i*out_height*out_height + m*out_height + n] < 0){
                    output[i*out_height*out_height + m*out_height + n] = 0;
                }
            }
        }
    }

    cudaFree(gpu_input);
    cudaFree(gpu_output);
    cudaFree(gpu_weights);
    cudaFree(gpu_input_channel);
    cudaFree(gpu_out_height);
}

// conv_back和cuda_conv_relu_back为不使用shared memory的旧版本cuda加速函数
__global__ static void conv_back(float *input, float *output, float *weights, int *input_channel, int *out_height){
    int tid = threadIdx.x; // input_channel
    int bid = blockIdx.x; // output_channel
    int ichannel = *input_channel;
    int iheight = *out_height+2;
    int oheight = *out_height;

    for(int m=0; m<oheight; m++){
        for(int n=0; n<oheight; n++){
            // output[bid][m][n][tid]
            output[bid*(oheight*oheight*ichannel) + m*(oheight*ichannel) + n*ichannel + tid] = 0;
            for(int f=0; f<3; f++){
                for(int w=0; w<3; w++){
                    // input[tid][m+f][n+w] * weights[bid][tid][f][w]
                    output[bid*(oheight*oheight*ichannel) + m*(oheight*ichannel) + n*ichannel + tid] += \
                    input[tid*iheight*iheight + (m+f)*iheight + (n+w)] * \
                    weights[bid*ichannel*3*3 + tid*3*3 + f*3 + w];
                }
            }
        }
    }
}

void cuda_conv_relu_back(int layer_id, float *input, float *output, int input_channel, int output_channel, int out_height){
    int input_size = input_channel*(out_height+2)*(out_height+2);
    int output_size = output_channel*out_height*out_height;
    int weights_size = 512*512*3*3;

    float *inter_output = (float *)malloc(output_size*input_channel*sizeof(float));

    float *gpu_input, *gpu_weights, *gpu_output;
    int *gpu_input_channel, *gpu_out_height;

    cudaMalloc((void**)&gpu_input, sizeof(float)*input_size);
    cudaMalloc((void**)&gpu_output, sizeof(float)*output_size*input_channel);
    cudaMalloc((void**)&gpu_weights, sizeof(float)*weights_size);
    cudaMalloc((void**)&gpu_input_channel, sizeof(int));
    cudaMalloc((void**)&gpu_out_height, sizeof(int));

    cudaMemcpy(gpu_input, input, sizeof(float)*input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_weights, conv_weights[layer_id-1], sizeof(float)*weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_input_channel, &input_channel, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_out_height, &out_height, sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(output_channel);
    dim3 dimBlock(input_channel);

    conv_back<<<dimGrid, dimBlock>>>(gpu_input, gpu_output, gpu_weights, gpu_input_channel, gpu_out_height);

    cudaMemcpy(inter_output, gpu_output, sizeof(float)*output_size*input_channel, cudaMemcpyDeviceToHost);

    for(int i=0; i<output_channel; i++){
        for(int m=0; m<out_height; m++){
            for(int n=0; n<out_height; n++){
                output[i*out_height*out_height + m*out_height + n] = conv_bias[layer_id-1][i];
                for(int j=0; j<input_channel; j++){
                    output[i*out_height*out_height + m*out_height + n] += \
                    inter_output[i*out_height*out_height*input_channel + m*out_height*input_channel + n*input_channel + j];
                }
            }
        }
    }

    for(int i=0; i<output_size; i++){
        if(output[i] < 0){
            output[i] = 0;
        }
    }

    free(inter_output);
    cudaFree(gpu_input);
    cudaFree(gpu_output);
    cudaFree(gpu_weights);
    cudaFree(gpu_input_channel);
    cudaFree(gpu_out_height);
}
/*********************************************************/

/************************ maxpool ************************/
__global__ static void maxpool(float *input, float *output, int *in_height, int *out_height){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int iheight = *in_height;
    int oheight = *out_height;

    for(int i=0; i<oheight; i++){
        float max_temp = 0;
        if(input[bid*iheight*iheight + (2*i)*iheight + 2*tid] > max_temp){
            max_temp = input[bid*iheight*iheight + (2*i)*iheight + 2*tid];
        }
        if(input[bid*iheight*iheight + (2*i+1)*iheight + 2*tid] > max_temp){
            max_temp = input[bid*iheight*iheight + (2*i+1)*iheight + 2*tid];
        }
        if(input[bid*iheight*iheight + (2*i)*iheight + 2*tid+1] > max_temp){
            max_temp = input[bid*iheight*iheight + (2*i)*iheight + 2*tid+1];
        }
        if(input[bid*iheight*iheight + (2*i+1)*iheight + 2*tid+1] > max_temp){
            max_temp = input[bid*iheight*iheight + (2*i+1)*iheight + 2*tid+1];
        }
        output[bid*oheight*oheight + i*oheight + tid] = max_temp;
    }
}

void cuda_maxpool(float *input, float *output, int channel, int in_height, int out_height){
    int input_size = channel*in_height*in_height;
    int output_size = channel*out_height*out_height;

    float *gpu_input, *gpu_output;
    int *gpu_in_height, *gpu_out_height;

    cudaMalloc((void**)&gpu_input, sizeof(float)*input_size);
    cudaMalloc((void**)&gpu_output, sizeof(float)*output_size);
    cudaMalloc((void**)&gpu_in_height, sizeof(int));
    cudaMalloc((void**)&gpu_out_height, sizeof(int));

    cudaMemcpy(gpu_input, input, sizeof(float)*input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_in_height, &in_height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_out_height, &out_height, sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(channel);
    dim3 dimBlock(out_height);

    maxpool<<<dimGrid, dimBlock>>>(gpu_input, gpu_output, gpu_in_height, gpu_out_height);

    cudaMemcpy(output, gpu_output, sizeof(float)*output_size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_input);
    cudaFree(gpu_output);
    cudaFree(gpu_in_height);
    cudaFree(gpu_out_height);
}
/*********************************************************/

/*********************** gemm&relu ***********************/
// gemm和cuda_gemm为使用shared memory的cuda加速函数
__global__ static void gemm(float *input, float *output, float *weights, int *input_size){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ float shared[512];

    if(tid == 0){
        for(int i=0; i<512; i++){
            shared[i] = 0;
        }
    }
    __syncthreads();
    for(int i=tid; i<(*input_size); i+=512){
        shared[tid] += input[i] * weights[bid*(*input_size) + i];
    }
    __syncthreads();
    if(tid == 0){
        output[bid] = 0;
        for(int i=0; i<512; i++){
            output[bid] += shared[i];
        }
    }
}

void cuda_gemm(int layer_id, float *input, float *output, int input_size, int output_size, bool relu){
    float *gpu_input, *gpu_output, *gpu_weights;
    int *gpu_input_size;

    cudaMalloc((void**)&gpu_input, sizeof(float)*input_size);
    cudaMalloc((void**)&gpu_output, sizeof(float)*output_size);
    cudaMalloc((void**)&gpu_weights, sizeof(float)*(input_size*output_size));
    cudaMalloc((void**)&gpu_input_size, sizeof(int));

    cudaMemcpy(gpu_input, input, sizeof(float)*input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_weights, gemm_weights[layer_id-1], sizeof(float)*(input_size*output_size), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_input_size, &input_size, sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(output_size);
    dim3 dimBlock(512);

    gemm<<<dimGrid, dimBlock>>>(gpu_input, gpu_output, gpu_weights, gpu_input_size);

    cudaMemcpy(output, gpu_output, sizeof(float)*output_size, cudaMemcpyDeviceToHost);

    for(int i=0; i<output_size; i++){
        output[i] += gemm_bias[layer_id-1][i];
    }
    if(relu){
        for(int i=0; i<output_size; i++){
            if(output[i] < 0){
                output[i] = 0;
            }
        }
    }

    cudaFree(gpu_input);
    cudaFree(gpu_output);
    cudaFree(gpu_weights);
    cudaFree(gpu_input_size);
}

// gemm_back和cuda_gemm_back为不使用shared memory的旧版本cuda加速函数
__global__ static void gemm_back(float *input, float *output, float *weights, int *input_size){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    output[bid*512 + tid] = 0;
    for(int i=tid; i<(*input_size); i+=512){
        output[bid*512 + tid] += input[i] * weights[bid*(*input_size) + i];
    }
}

void cuda_gemm_back(int layer_id, float *input, float *output, int input_size, int output_size, bool relu){
    float *gpu_input, *gpu_output, *gpu_weights;
    int *gpu_input_size;

    float *inter_output;
    inter_output = (float *)malloc(output_size*512*sizeof(float));

    cudaMalloc((void**)&gpu_input, sizeof(float)*input_size);
    cudaMalloc((void**)&gpu_output, sizeof(float)*output_size*512);
    cudaMalloc((void**)&gpu_weights, sizeof(float)*(input_size*output_size));
    cudaMalloc((void**)&gpu_input_size, sizeof(int));

    cudaMemcpy(gpu_input, input, sizeof(float)*input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_weights, gemm_weights[layer_id-1], sizeof(float)*(input_size*output_size), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_input_size, &input_size, sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(output_size);
    dim3 dimBlock(512);

    gemm_back<<<dimGrid, dimBlock>>>(gpu_input, gpu_output, gpu_weights, gpu_input_size);

    cudaMemcpy(inter_output, gpu_output, sizeof(float)*output_size*512, cudaMemcpyDeviceToHost);

    for(int i=0; i<output_size; i++){
        output[i] += gemm_bias[layer_id-1][i];
        for(int j=0; j<512; j++){
            output[i] += inter_output[i*512 + j];
        }
    }
    if(relu){
        for(int i=0; i<output_size; i++){
            if(output[i] < 0){
                output[i] = 0;
            }
        }
    }

    free(inter_output);
    cudaFree(gpu_input);
    cudaFree(gpu_output);
    cudaFree(gpu_weights);
    cudaFree(gpu_input_size);
}
/*********************************************************/