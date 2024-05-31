此目录包括以下文件：

- main.cu：定义主函数
- init.cu：定义模型权重与偏差初始化函数
- structure.cu：定义层与神经网络推理的封装函数
- functions.cu：定义CUDA Kernel函数
- Makefile：编译代码
- Readme.md：文件介绍



输入 "make" 生成可执行文件 "gpu"，输入 "./gpu" 执行神经网络推理测试，输入 "make clean" 清除可执行文件。

运行将输出 "Begin reading ... Complete reading." 字样，指明模型权重与偏差的读取过程。在对每个测试数据进行推理前，会输出 "Inference input[i]" 字样，指明正在对第 i 个测试数据进行操作。

函数中读取txt文件使用的是服务器小组账号下的绝对路径，运行时可能需要修改。相应文件存储在 /home/group14/code/parameters/ 文件夹下，由于文件太大，没有一并提交。