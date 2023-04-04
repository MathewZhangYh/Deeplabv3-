DeeplabV3+
---
### 目录
1. [所需环境](#所需环境)
2. [文件下载](#文件下载)
3. [数据准备](#数据准备)
4. [程序运行](#程序运行)
5. [评估结果](#评估结果)

### 所需环境
opencv_python==4.1.2.30\
torch==1.2.0

### 文件下载
已经训练好的模型 deeplabv3+_model.pth

链接：https://pan.baidu.com/s/1uBv9sc3qg-QBzXnEifqxrQ 

提取码：mzyh 

### 数据准备
数据集下载地址：https://www.msri.org/people/members/eranb/

1、训练前将原始图片的 jpg 格式文件放在 weizmann_horse_db\horse

2、训练前将 mask 的 png 格式文件放在 weizmann_horse_db\mask

3、将 deeplabv3+_model.pth 模型文件放在 model

### 训练评估   
1、直接运行deeplabv3_plus.ipynb

2、可以载入已训练好的模型继续训练，测试程序可行性训练1个epoch

3、在验证集上进行推理，生成mask存放在detection文件夹


### 评估结果
| miou | boundary iou |
|------|--------------|
| 94.35  | 74.62        |
