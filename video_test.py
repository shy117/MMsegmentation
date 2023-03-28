# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
import cv2
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os


# 直线方程函数
def f_1(x, A, B):
    return A * x + B


# 产生结果
def generate_result(result, file):
    flag1 = ['0', '0']
    flag2 = ['0', '0']
    p1 = p2 = p3 = p4 = 0
    for i in range(3, 0, -1):
        y0 = []
        x0 = []
        num = 0
        for y in range(810, 1080):  #
            a = 0
            count = 0
            for x in range(1920):
                if result[0][y][x] == i:
                    a += x
                    count += 1
            if a != 0:
                y0.append(y)
                x0.append(a / count)
                num += count

        # 直线拟合与绘制
        if num > 1000:
            if flag1[0] == '0' or i == 2:
                A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]
                y1 = np.arange(810, 1080, 1)
                x1 = (y1 - B1) / A1

                if A1 < 0:
                    p1 = int((810 - B1) / A1)
                    p3 = int((1080 - B1) / A1)
                    if i == 3:
                        flag1[0] = '1'
                        flag2[0] = '0'

                    elif i == 2:
                        flag1[0] = '1'
                        flag2[0] = '1'

                    elif i == 1:
                        flag1[0] = '0'
                        flag2[0] = '0'

                else:
                    p2 = int((810 - B1) / A1)
                    p4 = int((1080 - B1) / A1)
                    if i == 3:
                        flag1[1] = '1'
                        flag2[1] = '0'

                    elif i == 2:
                        flag1[1] = '1'
                        flag2[1] = '1'

                    elif i == 1:
                        flag1[1] = '0'
                        flag2[1] = '0'

    file.write(flag1[0] + flag1[1] + '\n')
    file.write(flag2[0] + flag2[1] + '\n')
    file.write(str(p1) + ' 810\n')
    file.write(str(p2) + ' 810\n')
    file.write(str(p3) + ' 1080\n')
    file.write(str(p4) + ' 1080\n')
    file.close()


def main():
    video = 'work_dirs/fcn_hr18s_voc_aug/video/test.mp4'
    output_file = 'work_dirs/fcn_hr18s_voc_aug/video/video3.mp4'
    config = 'fcn_hr18s_voc.py'
    checkpoint = 'work_dirs/fcn_hr18s_voc_aug/pth/iter_20000.pth'
    palette = 'cityscapes'
    txt_file = "work_dirs/fcn_hr18s_voc_aug/result1/"  # txt结果存放位置
    opacity = 0.5
    result_txt = False
    show_wait_time = 100
    show = None
    writer = None
    # build the model from a config file and a checkpoint file从配置文件和检查点文件构建模型
    model = init_segmentor(config, checkpoint, device='cuda:0')

    # build input video建立输入视频
    cap = cv2.VideoCapture(video)
    assert (cap.isOpened())
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # init output video

    output_fps = input_fps
    output_height = input_height
    output_width = input_width
    if output_file is not None:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        output_fps = output_fps if output_fps > 0 else input_fps
        output_height = output_height if output_height > 0 else int(
            input_height)
        output_width = output_width if output_width > 0 else int(
            input_width)
        writer = cv2.VideoWriter(output_file, fourcc, output_fps, (input_width, input_height), True)

    # start looping开始循
    i = 0
    print('Press "Esc", "q" or "Q" to exit.')
    try:
        while True:
            flag, frame = cap.read()
            if not flag:
                print("结束")
                break

            # test a single image测试单个图像
            result = inference_segmentor(model, frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break

            if i > 0 and i % 10 == 0 and result_txt:
                file = open(txt_file + '{}.txt'.format(i), "a")  # 原图片是以10帧分割的
                generate_result(result, file)  # 产生结果

            # blend raw image and prediction混合原始图像和预测
            draw_img = model.show_result(frame, result, opacity=opacity, wait_time=1, show=False)
            # palette=get_palette(palette),
            # show=False,
            # opacity=opacity)

            if show:
                cv2.imshow('video_demo', draw_img)
                cv2.waitKey(show_wait_time)
            if writer:
                if draw_img.shape[0] != output_height or draw_img.shape[1] != output_width:
                    draw_img = cv2.resize(draw_img, (output_width, output_height))
                writer.write(draw_img)
                i += 1
                print('frame', i)


    finally:
        if writer:
            writer.release()
        cap.release()


if __name__ == '__main__':
    main()
