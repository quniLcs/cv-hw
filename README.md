神经网络分类器
==============

GitHub repo 链接：<https://github.com/quniLcs/cv-nnc>

网盘链接：[百度网盘
神经网络分类器](https://pan.baidu.com/s/1hQtjDPMmQEPDIk5TRP6MwA?pwd=6bku)

使用模块说明
------------

`tensorflow`：用于加载MNIST数据集；

`numpy`：用于设置随机数种子与数学计算；

`pickle`：用于保存模型；

`matplotlib.pyplot`：用于可视化；

`warnings`：用于过滤警告信息。

代码文件说明
------------

`report.ipynb`：报告的源代码，包含超参数查找、损失函数曲线、错误率曲线、网络参数可视化等部分。

`train.py`： 定义一个函数`train`，输入五个参数，
第一个参数表示网络结构，即隐藏层的神经元数量， 第二个参数表示学习率初始值，
第三个参数表示正则化参数，
第四个参数`prin_error`表示是否打印训练过程中的训练集和测试集错误率，
第五个参数`prin_loss`表示是否打印训练过程中的训练集和测试集损失函数；
函数依照训练集对训练集和测试集进行标准化处理， 用标准正态分布初始化权重，
接着进行100000个循环的随机梯度下降，
其中学习率余弦衰减，记录20次错误率和损失函数，
最后在当前目录保存模型并返回四个参数，分别表示训练过程中的训练集和测试集的错误率和损失函数；
直接运行该文件时，调用`train([100], 1e-3, 0.05, prin_error = True)`。

`test.py`： 定义一个函数`test`，输入三个参数，即函数`test`的前三个参数；
函数从当前目录加载模型、计算错误率并将其返回；
直接运行该文件时，调用`test([100], 1e-3, 0.05)`。

`predict.py`： 定义一个函数`predict`，返回三个参数， 第一个参数表示预测值，
第二个参数表示错误率， 第三个参数表示损失函数； 用于函数`test`和`predict`。

`backprop.py`： 定义一个函数`backprop`， 使用反向传播算法计算梯度并将其返回；
用于函数`test`。

其它文件说明
------------

`mean.dat`和`std.dat`用于存储测试集的均值和标准差，进而用于标准化；

`model weights with num_hidden [100], alpha = 0.001, lambda =
0.05.dat`等文件用于保存模型，生成于代码文件夹，存储于模型文件夹。若要载入提供的模型，需要将相应的模型文件复制到代码文件夹，否则需要使用`train.py`自行生成模型。

训练和测试示例代码
------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from train import train
from test import test

train([100], 1e-3, 0.05, prin_error = True)
train_error, test_error = test([100], 1e-3, 0.05)
print('training error = %11.6f  testing error = %10.4f' % (train_error, test_error))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

超参数查找
----------

对网络结构、学习率初始值、正则化参数进行搜索：

| `num_hidden` | `alpha` | `lambd` | `error_train` | `error_test` |
|--------------|---------|---------|---------------|--------------|
| `[50]`       | `0.001` | `0.05`  | `0.087750`    | `0.0852`     |
| `[80]`       | `0.001` | `0.05`  | `0.081267`    | `0.0814`     |
| `[100]`      | `0.001` | `0.05`  | `0.080933`    | `0.0807`     |
| `[120]`      | `0.001` | `0.05`  | `0.076483`    | `0.0771`     |
| `[150]`      | `0.001` | `0.05`  | `0.075233`    | `0.0778`     |
| `[200]`      | `0.001` | `0.05`  | `0.071983`    | `0.0753`     |
| `[250]`      | `0.001` | `0.05`  | `0.068017`    | `0.0747`     |
| `[300]`      | `0.001` | `0.05`  | `0.066033`    | `0.0691`     |
| `[400]`      | `0.001` | `0.05`  | `0.061450`    | `0.0668`     |
| `[500]`      | `0.001` | `0.05`  | `0.057533`    | `0.0655`     |
| `[600]`      | `0.001` | `0.05`  | `0.054650`    | `0.0613`     |
| `[700]`      | `0.001` | `0.05`  | `0.050967`    | `0.0582`     |
| `[100]`      | `0.005` | `0.05`  | `0.083900`    | `0.0825`     |
| `[100]`      | `0.005` | `0.05`  | `0.099350`    | `0.0986`     |
| `[100]`      | `0.001` | `0.01`  | `0.087317`    | `0.0941`     |
| `[100]`      | `0.001` | `0.1`   | `0.101800`    | `0.0986`     |
