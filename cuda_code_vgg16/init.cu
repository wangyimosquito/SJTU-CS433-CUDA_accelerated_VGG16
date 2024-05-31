// 本文件读取weights和bias到全局变量中

#include <stdio.h>
#include <string.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using  namespace std;

float conv_bias[13][512];
float conv_weights[13][512*512*3*3];

float gemm_bias[3][4096];
float gemm_weights[3][4096*25088];

// npy文件已预先转换为txt文件，保留16位小数，此处读取txt文件
void read_para(string para_path, float *para){
    ifstream ifile(para_path.c_str());
    string line;
    int i = 0;
    while(getline(ifile, line)){
        if(line == "\n"){
            break;
        }
        para[i] = atof(line.c_str());
        i++;
    }
}

// 全局变量初始化函数
void initModel(){
    cout << "Begin reading ..." << endl;

    read_para("/home/group14/code/parameters/conv1_bias_64.txt", conv_bias[0]);
    read_para("/home/group14/code/parameters/conv1_weights_64_3_3_3.txt", conv_weights[0]);

    read_para("/home/group14/code/parameters/conv2_bias_64.txt", conv_bias[1]);
    read_para("/home/group14/code/parameters/conv2_weights_64_64_3_3.txt", conv_weights[1]);

    read_para("/home/group14/code/parameters/conv3_bias_128.txt", conv_bias[2]);
    read_para("/home/group14/code/parameters/conv3_weights_128_64_3_3.txt", conv_weights[2]);

    read_para("/home/group14/code/parameters/conv4_bias_128.txt", conv_bias[3]);
    read_para("/home/group14/code/parameters/conv4_weights_128_128_3_3.txt", conv_weights[3]);

    read_para("/home/group14/code/parameters/conv5_bias_256.txt", conv_bias[4]);
    read_para("/home/group14/code/parameters/conv5_weights_256_128_3_3.txt", conv_weights[4]);

    read_para("/home/group14/code/parameters/conv6_bias_256.txt", conv_bias[5]);
    read_para("/home/group14/code/parameters/conv6_weights_256_256_3_3.txt", conv_weights[5]);

    read_para("/home/group14/code/parameters/conv7_bias_256.txt", conv_bias[6]);
    read_para("/home/group14/code/parameters/conv7_weights_256_256_3_3.txt", conv_weights[6]);

    read_para("/home/group14/code/parameters/conv8_bias_512.txt", conv_bias[7]);
    read_para("/home/group14/code/parameters/conv8_weights_512_256_3_3.txt", conv_weights[7]);

    read_para("/home/group14/code/parameters/conv9_bias_512.txt", conv_bias[8]);
    read_para("/home/group14/code/parameters/conv9_weights_512_512_3_3.txt", conv_weights[8]);

    read_para("/home/group14/code/parameters/conv10_bias_512.txt", conv_bias[9]);
    read_para("/home/group14/code/parameters/conv10_weights_512_512_3_3.txt", conv_weights[9]);

    read_para("/home/group14/code/parameters/conv11_bias_512.txt", conv_bias[10]);
    read_para("/home/group14/code/parameters/conv11_weights_512_512_3_3.txt", conv_weights[10]);

    read_para("/home/group14/code/parameters/conv12_bias_512.txt", conv_bias[11]);
    read_para("/home/group14/code/parameters/conv12_weights_512_512_3_3.txt", conv_weights[11]);

    read_para("/home/group14/code/parameters/conv13_bias_512.txt", conv_bias[12]);
    read_para("/home/group14/code/parameters/conv13_weights_512_512_3_3.txt", conv_weights[12]);

    read_para("/home/group14/code/parameters/gemm1_bias_4096.txt", gemm_bias[0]);
    read_para("/home/group14/code/parameters/gemm1_weights_4096_25088.txt", gemm_weights[0]);
    
    read_para("/home/group14/code/parameters/gemm2_bias_4096.txt", gemm_bias[1]);
    read_para("/home/group14/code/parameters/gemm2_weights_4096_4096.txt", gemm_weights[1]);

    read_para("/home/group14/code/parameters/gemm3_bias_1000.txt", gemm_bias[2]);
    read_para("/home/group14/code/parameters/gemm3_weights_1000_4096.txt", gemm_weights[2]);

    cout << "Complete reading." << endl;
}