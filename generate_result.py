from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os


# np.set_printoptions(threshold=np.inf)

# 直线方程函数
def f_1(x, A, B):
    return A * x + B


# 产生结果
def generate_result(result, file):
    flag = 0
    flag1 = ['0', '0']
    flag2 = ['0', '0']
    p1 = p2 = p3 = p4 = 0
    for i in range(3, 0, -1):
        y0 = []
        x0 = []
        num = 0
        for y in range(810, 1080):  # 从810开始扫描
            a = 0
            count = 0  # 记录掩码个数
            for x in range(1920):
                if result[0][y][x] == i:
                    a += x
                    count += 1
            if a != 0:
                if count > 105:
                    flag += 1
                y0.append(y)
                x0.append(a / count)  # 求x坐标的均值
                num += count  # 掩码总个数

        # 直线拟合与绘制
        if num > 1000:
            if flag1[0] == '0' or i == 2:
                A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]  # 拟合直线A1为斜率

                if A1 < 0:  # 左车道线
                    p1 = int((810 - B1) / A1)  # 求交点
                    p3 = int((1080 - B1) / A1)
                    if i == 3:  # 白实线
                        flag1[0] = '1'
                        flag2[0] = '0'

                    elif i == 2:  # 黄线
                        flag1[0] = '1'
                        flag2[0] = '1'
                        if flag > 5:  # 黄线中包含双实线
                            flag1[0] = '2'

                    elif i == 1:  # 白虚线
                        flag1[0] = '0'
                        flag2[0] = '0'

                else:  # 右车道线
                    p2 = int((810 - B1) / A1)
                    p4 = int((1080 - B1) / A1)
                    if i == 3:
                        flag1[1] = '1'
                        flag2[1] = '0'

                    elif i == 2:
                        flag1[1] = '1'
                        flag2[1] = '1'
                        if flag > 5:  # 黄线中包含双实线
                            flag2[0] = '2'

                    elif i == 1:
                        flag1[1] = '0'
                        flag2[1] = '0'

    file.write('\n' + flag1[0] + flag1[1] + '\n')
    file.write(flag2[0] + flag2[1] + '\n')
    file.write(str(p1) + ' 810\n')
    file.write(str(p2) + ' 810\n')
    file.write(str(p3) + ' 1080\n')
    file.write(str(p4) + ' 1080\n')
    file.close()


if __name__ == '__main__':
    confing_file = './fcn_hr18s_voc.py'  # 配置文件
    checkpoints_file = 'work_dirs/fcn_hr18s_voc_aug/pth/iter_20000.pth'  # 模型
    img_file = 'work_dirs/fcn_hr18s_voc_aug/test/'  # 原始图片存放位置
    img_list = os.listdir(img_file)
    txt_file = 'work_dirs/fcn_hr18s_voc_aug/result2/'  # txt结果存放位置
    model = init_segmentor(confing_file, checkpoints_file, device='cuda:0')  # 加载模型

    for img in img_list:
        name = img.split('.')
        img_type = name[1]  # 得到图片类型、
        if img_type == 'jpg' or img_type == 'png':
            result = inference_segmentor(model, img_file + img)  # 检测图片获得掩码图
            i = int(name[0].split('t')[2])  # 得到图片下标
            file = open(txt_file + '{}.txt'.format(i * 10), "a")  # 原图片是以10帧分割的
            generate_result(result, file)  # 产生结果
            print('{}.txt'.format(i * 10))
