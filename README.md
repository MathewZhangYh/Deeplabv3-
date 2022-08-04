Homework
---
author ： 自动化1904张宇恒U201914728
### 目录
1. [所需环境](#所需环境)
2. [文件下载](#文件下载)
3. [数据准备](#数据准备)
4. [训练步骤](#训练步骤)
5. [评估步骤](#评估步骤)
6. [评估结果](#评估结果)

### 所需环境
opencv_python==4.1.2.30\
torch==1.2.0

### 文件下载
已经训练好的模型 deeplabv3+_model.pth

链接：https://pan.baidu.com/s/1uBv9sc3qg-QBzXnEifqxrQ 

提取码：mzyh 

### 数据准备
1、训练前将原始图片的 jpg 格式文件放在 weizmann_horse_db\horse

2、训练前将 mask 的 png 格式文件放在 weizmann_horse_db\mask

3、将 deeplabv3+_model.pth 模型文件放在 model

### 训练步骤    
1、运行 annotation.ipynb 文件划分数据集

2、运行 train.ipynb 文件，进行模型训练
 

### 评估步骤
1、用于评估的模型路径在 logs 文件夹，模型文件名为 best_model.pth

2、可在训练完成后，直接运行 model_eval.ipynb

3、model_eval.ipynb 会在验证集上，生成模型预测的mask

4、或修改 model_eval.ipynb 中 "model_path" 再评估 miou 和 boundary iou


### 评估结果
| miou | boundary iou |
|------|--------------|
| 94.35  | 74.62        |
