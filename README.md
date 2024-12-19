# rkface
这是一个在RV1106上使用facenet、retinaface模型进行人脸检测和识别的测试程序，部份代码参考了官方RKMPI、RKNN的代码。

## 编译
本工程默认使用VSCode + CMake + WSL方式进行交叉编译，你可以通过下列不同的方式进行编译。

### 1、WSL Ubuntu
将源码复制到WSL环境的Ubuntu系统下，在VSCode中通过WSL扩展打开源码目录，配置好交叉编译工具后直接Build
.vscode下配置有直接将编译好的程序上传到板卡的任务，可以根据你的情况进行修改。

### 2、Ubuntu
将源码复制到Ubuntu下，确保交叉编译环境可用
```
cd rkface

mkdir build && cd build
cmake ..
make 
```

## 程序的运行
将编译好的程序上传到板卡上执行./rkface，并保证其目录结构如下：

```
rkface
├── faces-----------------------人脸特定预定义，只能是字母和数字命名
|    ├── Zhang.dat
|    └── Xia.dat
└── model-----------------------RKNN模型文件 
    ├── mobilefacenet.rknn
    └── retinaface.rknn
```

如果需要板子加电自动运行，需要修改板卡的应用启动脚本。

## 以上代码，仅供参考，侵删！ 