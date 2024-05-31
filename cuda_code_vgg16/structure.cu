// 本文件定义每层操作的封装函数，以及总的推理过程的封装函数

#include <math.h>
#include "functions.cu"

using  namespace std;

// 卷积层的补0函数
void padding(float *input, float *output, int channel, int height){
    for(int c=0; c<channel; c++){
        for(int i=0; i<height+2; i++){
            for(int j=0; j<height+2; j++){       
                output[c*(height+2)*(height+2) + i*(height+2) + j] = 0;
            }
        }
    }
    for(int c=0; c<channel; c++){
        for(int i=0; i<height; i++){
            for(int j=0; j<height; j++){       
                output[c*(height+2)*(height+2) + (i+1)*(height+2) + (j+1)] = \
                input[c*height*height + i*height + j];
            }
        }
    }
}

// 卷积层的封装函数
void layer_conv(int layer_id, float *input, float *output, int input_channel, int output_channel, int out_height){
    // printf("Convolution with ReLU from %i*%i*%i to %i*%i*%i\n", input_channel, out_height, out_height, output_channel, out_height, out_height);

    float *padding_input;
    padding_input = (float *)malloc((input_channel*(out_height+2)*(out_height+2))*sizeof(float));
    padding(input, padding_input, input_channel, out_height);

    /********************** gpu **************************/
    cuda_conv_relu(layer_id, padding_input, output, input_channel, output_channel, out_height);
    /*****************************************************/

    /********************** cpu **************************
    for(int i=0; i<output_channel; i++){
        for(int m=0; m<out_height; m++){
            for(int n=0; n<out_height; n++){
                output[i*out_height*out_height + m*out_height + n] = 0;
                for(int j=0; j<input_channel; j++){
                    for(int f=0; f<3; f++){
                        for(int w=0; w<3; w++){
                            output[i*out_height*out_height + m*out_height + n] += (\
                            padding_input[j*(out_height+2)*(out_height+2) + (m+f)*(out_height+2) + (n+w)] * \
                            conv_weights[layer_id-1][i*(input_channel*3*3) + j*(3*3) + f*3 + w]);
                        }
                    }
                }
            }
        }
    }
    for(int i=0; i<output_channel; i++){
        for(int mn=0; mn<out_height*out_height; mn++){
            output[i*out_height*out_height + mn] += conv_bias[layer_id-1][i];
            if(output[i*out_height*out_height + mn] < 0){
                output[i*out_height*out_height + mn] = 0;
            }
        }
    }
    /*****************************************************/
    free(padding_input);
    /***************** see the result *********************
    for(int i=0; i<1; i++){
        for(int m=0; m<out_height; m++){
            if(m<8){
                for(int n=0; n<out_height; n++){
                    if(n<8){
                        cout << output[i*out_height*out_height + m*out_height +n] << " ";
                    }
                }
                cout << endl;
            }
        }
        cout << endl;
    }
    ******************************************************/
}

// 最大池化层的封装函数
void layer_maxpool(float *input, float *output, int channel, int in_height, int out_height){
    // printf("Maxpool from %i*%i*%i to %i*%i*%i\n", channel, in_height, in_height, channel, out_height, out_height);

    /********************** gpu **************************/
    cuda_maxpool(input, output, channel, in_height, out_height);
    /*****************************************************/
    
    /********************** cpu **************************
    for(int i=0; i<channel; i++){
        for(int m=0; m<out_height; m++){
            for(int n=0; n<out_height; n++){
                output[i*out_height*out_height + m*out_height + n] = \
                fmax(input[i*in_height*in_height + 2*m*in_height + 2*n], \
                fmax(input[i*in_height*in_height + (2*m+1)*in_height + 2*n], \
                fmax(input[i*in_height*in_height + 2*m*in_height + 2*n+1], \
                input[i*in_height*in_height + (2*m+1)*in_height + 2*n+1])));
            }
        }
    }
    /*****************************************************/

    /***************** see the result *********************
    for(int i=0; i<1; i++){
        for(int m=0; m<out_height; m++){
            if(m<8){
                for(int n=0; n<out_height; n++){
                    if(n<8){
                        cout << output[i*out_height*out_height + m*out_height +n] << " ";
                    }
                }
                cout << endl;
            }
        }
        cout << endl;
    }
    ******************************************************/
}

// 全连接层的封装函数
void layer_gemm(int layer_id, float *input, float *output, int input_size, int output_size, bool relu){
    // printf("Gemm from %i to %i\n", input_size, output_size);

    /********************** gpu **************************/
    cuda_gemm(layer_id, input, output, input_size, output_size, relu);
    /*****************************************************/

    /********************** cpu **************************
    for(int i=0; i<output_size; i++){
        output[i] = gemm_bias[layer_id-1][i];
        for(int j=0; j<input_size; j++){
            output[i] += (input[j] * gemm_weights[layer_id-1][i*input_size + j]);
        }
    }
    if(relu==true){
        for(int i=0; i<output_size; i++){
            if(output[i] < 0){
                output[i] = 0;
            }
        }
    }
    /*****************************************************/
}

// 推理操作的封装函数
void inference(float *image, float *prob){
    float *layer1_output = (float *)malloc((64*244*244)*sizeof(float));
    layer_conv(1, image, layer1_output, 3, 64, 244);

    float *layer2_output = (float *)malloc((64*244*244)*sizeof(float));
    layer_conv(2, layer1_output, layer2_output, 64, 64, 244);
    free(layer1_output);

    float *maxpool1_output = (float *)malloc((64*122*122)*sizeof(float));
    layer_maxpool(layer2_output, maxpool1_output, 64, 244, 122);
    free(layer2_output);

    float *layer3_output = (float *)malloc((128*122*122)*sizeof(float));
    layer_conv(3, maxpool1_output, layer3_output, 64, 128, 122);
    free(maxpool1_output);

    float *layer4_output = (float *)malloc((128*122*122)*sizeof(float));
    layer_conv(4, layer3_output, layer4_output, 128, 128, 122);
    free(layer3_output);

    float *maxpool2_output = (float *)malloc((128*61*61)*sizeof(float));
    layer_maxpool(layer4_output, maxpool2_output, 128, 122, 61);
    free(layer4_output);

    float *layer5_output = (float *)malloc((256*61*61)*sizeof(float));
    layer_conv(5, maxpool2_output, layer5_output, 128, 256, 61);
    free(maxpool2_output);

    float *layer6_output = (float *)malloc((256*61*61)*sizeof(float));
    layer_conv(6, layer5_output, layer6_output, 256, 256, 61);
    free(layer5_output);

    float *layer7_output = (float *)malloc((256*61*61)*sizeof(float));
    layer_conv(7, layer6_output, layer7_output, 256, 256, 61);
    free(layer6_output);

    float *maxpool3_output = (float *)malloc((256*30*30)*sizeof(float));
    layer_maxpool(layer7_output, maxpool3_output, 256, 61, 30);
    free(layer7_output);

    float *layer8_output = (float *)malloc((512*30*30)*sizeof(float));
    layer_conv(8, maxpool3_output, layer8_output, 256, 512, 30);
    free(maxpool3_output);

    float *layer9_output = (float *)malloc((512*30*30)*sizeof(float));
    layer_conv(9, layer8_output, layer9_output, 512, 512, 30);
    free(layer8_output);

    float *layer10_output = (float *)malloc((512*30*30)*sizeof(float));
    layer_conv(10, layer9_output, layer10_output, 512, 512, 30);
    free(layer9_output);

    float *maxpool4_output = (float *)malloc((512*15*15)*sizeof(float));
    layer_maxpool(layer10_output, maxpool4_output, 512, 30, 15);
    free(layer10_output);

    float *layer11_output = (float *)malloc((512*15*15)*sizeof(float));
    layer_conv(11, maxpool4_output, layer11_output, 512, 512, 15);
    free(maxpool4_output);

    float *layer12_output = (float *)malloc((512*15*15)*sizeof(float));
    layer_conv(12, layer11_output, layer12_output, 512, 512, 15);
    free(layer11_output);

    float *layer13_output = (float *)malloc((512*15*15)*sizeof(float));
    layer_conv(13, layer12_output, layer13_output, 512, 512, 15);
    free(layer12_output);

    float *maxpool5_output = (float *)malloc((512*7*7)*sizeof(float));
    layer_maxpool(layer13_output, maxpool5_output, 512, 15, 7);
    free(layer13_output);

    float *gemm1_output = (float *)malloc(4096*sizeof(float));
    layer_gemm(1, maxpool5_output, gemm1_output, 25088, 4096, true);
    free(maxpool5_output);

    float *gemm2_output = (float *)malloc(4096*sizeof(float));
    layer_gemm(2, gemm1_output, gemm2_output, 4096, 4096, true);
    free(gemm1_output);

    layer_gemm(3, gemm2_output, prob, 4096, 1000, false);
    free(gemm2_output);
}