# Comsen
## 华中科技大学光学与电子信息学院2409陈恪瑾
### 项目概述
#### 项目名称
波士顿房价预测
#### 数据来源
https://www.kaggle.com/c/house-prices-advanced-regression-techniques
#### 使用模型
基于pytroch的MLP
#### requirements
[requirements.txt](requirements.txt)
#### 主要参考文献（代码参考）
张奥森, Zachary C. Lipton, 李沐, Alexander J. Smola. 动手学深度学习[M]. 机械工业出版社, 2021.
[kaggle-house-price.ipynb](kaggle-house-price.ipynb)
### 项目成果
#### 源代码
[main.py](main.py)（基础版本）

[main_gpu.py](main_gpu.py)（完整功能版本）
#### 训练数据
[test.csv](house-prices-advanced-regression-techniques/test.csv)

[train.csv](house-prices-advanced-regression-techniques/train.csv)
#### 预测结果
[submission.csv](submission.csv)
#### 导出的onnx模型
[house_price_model.onnx](house_price_model.onnx)
#### 模型的可视化图片
[house_price_model.onnx.png](house_price_model.onnx.png)
#### 训练可视化曲线图和绘图代码
[training_plots](training_plots)

