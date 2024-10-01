import numpy as np
import torch
import os
import csv
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import seaborn as sns
from .utils import resize_img, cvtColor
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------#
#                           分类网络的评估
# --------------------------------------------------------------------------#

class DETECT(object):
    def __init__(self, model, input_shape, class_name, data_list, label_list, device):
        self.input_shape = input_shape
        self.model = model
        self.class_name = class_name
        self.data_list = data_list
        self.label_list = label_list
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    # -----------------------------单张图像检测-------------------------------------#
    def detect_image(self, image):
        image = cvtColor(image)
        img_data = resize_img(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(img_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            img = torch.from_numpy(image_data)
            img = img.to(self.device)
            output = self.model(img)
            preds = F.softmax(output, dim=1)
            pred_class = torch.argmax(preds).item()

        # print(self.class_name)
        print(preds)
        print(torch.argmax(preds.cpu()))

        pre_label = self.class_name[torch.argmax(preds).cpu()]
        probability = preds[0][pred_class].item()

        plt.subplot(1, 1, 1)
        plt.imshow(image)
        plt.title('Class:%s Probability:%.3f' % (pre_label, probability))
        plt.show()
        return pre_label

    # ---------------------------------数据集整体评估-------------------------------------------#
    #   目前可获得的指标有：
    #   top-1(准确率)、top-5(准确率)、mPrecision(平均精确率)、mRecall(平均召回率)、F1-score(F1分数)、
    #   混淆矩阵热力图、精确率条形图、召回率条形图
    # ----------------------------------------------------------------------------------------#
    def detect_data(self, num_classes, save_dir):
        idx = 0
        correct_1 = 0
        correct_5 = 0
        pred_class_list = []

        for img in tqdm(self.data_list):
            image = Image.open(img)
            image = cvtColor(image)
            img_data = resize_img(image, (self.input_shape[1], self.input_shape[0]))
            image_data = np.transpose(np.expand_dims(preprocess_input(np.array(img_data, np.float32)), 0), (0, 3, 1, 2))
            with torch.no_grad():
                img = torch.from_numpy(image_data)
                img = img.to(self.device)
                output = self.model(img)
                preds = F.softmax(output, dim=1)
                # -------------类别索引---------------------#
                pred_class = torch.argmax(preds).item()
            # -------------预测的label列表，用于计算混淆矩阵-----------------#
            pred_class_list.append(pred_class)
            # ---------类别名称（label）--------------------#
            pre_label = self.class_name[torch.argmax(preds).cpu()]
            # 概率
            # probability = preds[0][pred_class].item()
            # -------------------top-1--------------------------------#
            if accuracy(self.label_list[idx], pred_class):
                correct_1 += 1
            # ---------------------top-5-------------------------------#
            pred_5 = torch.argsort(preds, descending=True)
            pred_5 = pred_5[:5]
            if pred_class in pred_5:
                correct_5 += 1

            idx += 1

        # ------------------------创建混淆矩阵----------------------------------#
        #                   真实标签列表、概率列表、类别数
        # ---------------------------------------------------------------------#
        hist = fast_hist(np.array(self.label_list), np.array(pred_class_list), num_classes)
        # ---------------------计算各类别的精确率、召回率、F1分数--------------------------------#
        mPrecision = per_class_Precision(hist)
        mRecall = per_class_Recall(hist)
        mF1_score = f1_score(mPrecision, mRecall)

        # ------------------------绘制图像展示----------------------------------------#
        # ----------------------------混淆矩阵热力图----------------------------------#
        draw_matrix(hist, self.class_name, save_dir)
        # ----------------------------Recall和Precision条形图----------------------------------#
        draw_Recall(self.class_name, mRecall, save_dir)
        draw_Precision(self.class_name, mPrecision, save_dir)

        top_1 = correct_1 / idx
        top_5 = correct_5 / idx

        print(f'总样本数: {len(self.data_list)}\n正确样本数: {correct_1}')
        print(
            f'Top-1:{top_1 * 100:.3f}% | Top-5:{top_5 * 100:.3f}% | mPrecision:{np.mean(mPrecision) * 100:.3f}% | '
            f'mRecall:{np.mean(mRecall) * 100:.3f}% | F1-score:{np.mean(mF1_score) * 100:.3f}%')

        # -------需要查看各个类别的Recall和Precision取消以下注释即可-------------------------------#
        # print(f'Top-1:{top_1 * 100:.3f}% | Top-5:{top_5 * 100:.3f}%')
        # print(f'Precision:{mPrecision}')
        # print(f'Recall:{mRecall}')
        # print(f'F1-score:{mF1_score}')

# --------------------混淆矩阵---------------------------#
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

# --------------------各个类别的Recall---------------------------#
def per_class_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

# --------------------各个类别的Precision---------------------------#
def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

def f1_score(precision, recall):
    f1_scores = []
    for prec, rec in zip(precision, recall):
        if prec + rec == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * (prec * rec) / (prec + rec))
    return f1_scores

# --------------------准确率---------------------------#
def accuracy(target, pre_label):
    '''
    :param target:  真实标签
    :param pre_label: 预测标签
    '''
    correct = True if target == pre_label else False
    return correct

# -------------------混淆矩阵热力图------------------------------#
def draw_matrix(hist, class_name, save_dir):
    '''
    :param hist:    混淆矩阵
    :param class_name: 类别列表
    :param save_dir:   存储路径
    '''
    plt.figure()
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    sns.heatmap(hist, annot=True, fmt='d', cmap=cmap, linewidths=3,
                xticklabels=class_name, yticklabels=class_name, square=True)
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    # plt.show()
    plt.cla()
    plt.close("all")
    # ----------------文本信息保存---------------------------------#
    with open(os.path.join(save_dir, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in class_name])
        for i in range(len(hist)):
            writer_list.append([class_name[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)

# -------------------Recall条形图------------------------------#
def draw_Recall(class_name, mRecall, save_dir):
    '''
    :param class_name:  类别列表
    :param mRecall:     每个类别的召回率列表
    :param save_dir:    存储路径
    '''
    plt.barh(range(len(class_name)), mRecall, color='royalblue')  # 条形水平显示
    for idx, size in enumerate(mRecall):
        # ------------计算文本的 y 位置，确保文本在条形的顶端-------------#
        text_y = idx
        plt.text(size + 0.03, text_y, f'{size:.1f}', va='center', ha='center')
    # ----------------------设置图表标题和坐标轴标签---------------------#
    plt.title(f'mRecall:{np.mean(mRecall) * 100:.3f}%')
    plt.xlabel('Recall')
    plt.subplots_adjust(left=0.13, bottom=0.13)  # 增加底部边距
    # -----------------设置 y 轴的刻度标签为类别名称---------------------#
    plt.yticks(range(len(class_name)), class_name)
    # ----------由于 x 轴标签可能重叠，旋转它们以便阅读-------------------#
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(save_dir, 'Recall.png'))
    # plt.show()
    plt.close()

def preprocess_input(x):
    x /= 255
    x -= np.array([0.485, 0.456, 0.406])
    x /= np.array([0.229, 0.224, 0.225])
    return x

# -------------------Precision条形图------------------------------#
def draw_Precision(class_name, mPrecision, save_dir):
    '''
    :param mPrecision:  每个类别的精确率列表
    '''
    plt.barh(range(len(class_name)), mPrecision, color='royalblue')
    for idx, size in enumerate(mPrecision):
        text_y = idx
        plt.text(size + 0.03, text_y, f'{size:.1f}', va='center', ha='center')
    plt.title(f'mPrecision:{np.mean(mPrecision) * 100:.3f}%')
    plt.xlabel('Precision')
    plt.subplots_adjust(left=0.13, bottom=0.13)  # 增加底部边距
    plt.yticks(range(len(class_name)), class_name)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(save_dir, 'Precision.png'))
    # plt.show()
    plt.close()
