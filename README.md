# MMsegmentation车道线检测

## 一. MMsegmentation环境部署

### （1）我的环境配置

```
操作系统：Ubuntu20.04
IDE：vscode
Python: 3.6.13
PyTorch: 1.10.2+cu113
CUDA：113
GPU：NVIDIA GeForce RTX 3090
```

### （2）完整的安装脚本

#### Linux

​		这里便是一个完整安装 MMSegmentation 的脚本，使用 conda 并链接了数据集的路径（以您的数据集路径为 $DATA_ROOT 来安装）。

```shell
conda create -n open-mmlab python=3.10 -y
conda activate open-mmlab

conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  # 或者 "python setup.py develop"

mkdir data
ln -s $DATA_ROOT data
```

#### Windows (有风险)

​		这里便是一个完整安装 MMSegmentation 的脚本，使用 conda 并链接了数据集的路径（以您的数据集路径为 %DATA_ROOT% 来安装）。
注意：它必须是一个绝对路径。

```shell
conda create -n open-mmlab python=3.10 -y
conda activate open-mmlab

conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch
set PATH=full\path\to\your\cpp\compiler;%PATH%
pip install mmcv

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  # 或者 "python setup.py develop"

mklink /D data %DATA_ROOT%
```

## 二.数据集收集以及标注

### （1）数据分析

​		使用官方提供的视频，每12帧提取1帧，总共提取583张图片，剔除后84张无车道线图片，剩余499张数据样本。

![](https://s2.loli.net/2022/05/21/PcU5Y1tZBa8FLMs.png)

​	需要标注的数据区域为图片下1/3区域内的车道线。过远区域车道线不清晰，不利于模型的训练。只标注车行进的主车道线。

### （2）数据标注

​		数据标注我们选择使用labelme。其优势在于我们可以在任意地方使用该
工具。此外，它也可以帮助我们标注图像，不需要在电脑中安装或复制大型数据集。
标注方式：我们选择用多边形（Polygons）进行车道线的标注。

![](https://s2.loli.net/2022/05/21/bgeJK6hQY2R1XjW.png)

### （3）数据增强

​		在深度学习中，数据增强可以在样本数量不足或者样本质量不够好的情况下，提高样本质量，增加训练的数据量，提高模型的泛化能力，增加噪声数据，提升模型的鲁棒性。

​		我们对标注好的车道线数据进行数据增强，数据增强的同时保留原有标注数据。对每张图片进行4次数据增强，包含改变亮度、加噪声、加随机点、水平翻转4种形式的数据增强，不同形式的数据增强会随机叠加。

​		修改DataAugmentforLabelMe.py文件里的数据集路径，运行后即可得到增强的数据集。

![](https://s2.loli.net/2022/05/21/iIW3VdtZu2fK9w1.png)



### （4）数据集

​		数据集格式选择voc格式，将labelme标定好的json数据转voc格式。

在labelme虚拟环境下输入以下命令：

```
python labelme2voc.py data_dataset data_dataset_voc --labels labels.txt
```

会在当前文件夹下产生一个data_dataset_voc文件夹。

data_dataset文件夹里包含原始图片和labelme标定好的json数据。

JPEGImages保存原始图片

SegmentationClassPNG存放.png格式图片，语义（class）分割相关

运行t3v_txt.py生成trainval.txt、test.txt、train.txt、val.txt4个文件，比例可自行更改。

输入以下命令：

```
python SegmentationClassAug.py aug_dataset  data_dataset_vocaug --labels labels.txt
```

可将增强后的数据集也转换成voc数据集格式。

运行aug_txt.py生成aug.txt将data_dataset_vo和data_dataset_vocaug文件夹合并最后形成完整的数据集。

![](https://s2.loli.net/2022/05/21/V5gxF4MpYoazd81.png)

## 三.修改配置训练

### （1）修改配置

```
python tools/train.py configs/hrnet/fcn_hr18s_512x512_20k_voc12aug.py
```

会在work_dirs下生成fcn_hr18s_512x512_20k_voc12aug文件夹，在文件夹内有fcn_hr18s_512x512_20k_voc12aug.py文件，可对其进行修改。

![](https://s2.loli.net/2022/05/21/nylhYsx82DwZqRG.png)

SyncBN改为BN

![](https://s2.loli.net/2022/05/21/T9HNpoQIwFz8gGc.png)

数据集类型为PascalVOCDataset

所有data_root都修改为自己的数据集位置

Num_classes修改为自己的标签数加1

![](https://s2.loli.net/2022/05/21/9h6DsgHdYMCXBPe.png)

训练集包含数据增强

在mmsegmentation/mmseg/datasets下，找到voc.py文件，修改CLASSES和PALETTE 。

![](https://s2.loli.net/2022/05/21/qEhMJKGkawzD4OB.png) 

​	在mmsegmentation/mmseg/core/evaluation下，找到class_names.py文件，修改voc_classes()和voc_palette()。

![](https://s2.loli.net/2022/05/21/7mNHh6WBg9cz1Xj.png)

### （2）训练和测试

（1）训练

可以将fcn_hr18s_512x512_20k_voc12aug.py文件复制到根目录下，改名为fcn_hr18s_voc.py。

输入以下命令开始训练

```
python tools/train.py fcn_hr18s_voc.py
```

![](https://s2.loli.net/2022/05/21/fT6s3m4lKykh15z.png)

（2）测试

```
python tools/test.py fcn_hr18s_voc.py work_dirs/fcn_hr18s_voc/iter_20000.pth --show-dir my_output
```

会输出test.txt文件内包含的图片的测试结果

![](https://s2.loli.net/2022/05/21/BoNAf2pQjPydWIl.png)

### （3）后处理拟合车道线

加载模型检测图片获得0（背景）、1（白虚线）、2（黄实线）、3（白实线）的掩码图，遍历掩码图，分别逐行扫描3、2、1，记录每一行均值，并将均值拟合成直线。可根据掩码直接判断车道线的颜色和类别，用直线斜率判断左右车道线，代入y= 810和y=1080求与车道线的交点。

![](https://s2.loli.net/2022/05/21/njAD2gSEvGuL1kU.png)

运行generate_result.py文件可对原图进行检测在result文件夹下生成.txt文件

![](https://s2.loli.net/2022/05/21/VgZ28z7yHGhXSRE.png)

运行video_test.py可对视频进行检测生成检测后的视频，并生成.txt文件

![](https://s2.loli.net/2022/05/21/nTEKS4Le1HOgApY.png)







































