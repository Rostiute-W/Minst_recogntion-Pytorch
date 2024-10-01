import torch
from PIL import Image
from model.net import MinstNet
from utils.utils import get_classes
from utils.eval import DETECT

# ----------------------------------------------------------------#
#                             预测
# ----------------------------------------------------------------#
#   可进行单张图像预测 和 数据集整体评估
#   单张图像预测时获得 类别 和 概率
#   整体评估目前可获得的指标有：
#   top-1(准确率)、top-5(准确率)、mPrecision(平均精确率)、mRecall(平均召回率)、F1-score(F1分数)
#   混淆矩阵热力图、精确率条形图、召回率条形图
# ---------------------------------------------------------------#

# ------------------------训练设备--------------------------------#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# ------------------------模型权重文件----------------------------#
model_path = r'test3/logs/MinstNet.pth'
# -------------------------类别txt文件----------------------------#
classes_path = r'test3/logs/cls_classes.txt'
# --------------------------输入图像大小--------------------------#
Input_shape = [28, 28]
# -----------------测试集路径，评估在数据集上的指标-----------------#
data_path = r'test3/logs/cls_test.txt'
# --------------------预测文件的保存路径---------------------------#
#   当data_path != ''时生效
#   输出文件含各类别精确率、召回率条形图、混淆矩阵热力图、混淆矩阵csv文件
# ---------------------------------------------------------------#
save_path = r'test3/logs'
# -------------------获取类别列表和数量----------------------------#
class_name, num_classes = get_classes(classes_path)
# ---------------------图片和标签列表生成--------------------------#
data_list, label_list = [], []
if data_path != '':
    with open(data_path, 'r') as f:
        line = f.readlines()
        for path in line:
            img_path = path.strip().split(';')[-1]
            label = path.strip().split(';')[0]
            data_list.append(img_path)
            label_list.append(label)
# ----------------标签转为整形，用于混淆矩阵计算---------------------#
label_list = list(map(int, label_list))
# print(label_list)
# -----------------------------加载模型----------------------------#
model = MinstNet()
model.to(device)
print(model)
model.load_state_dict(torch.load(model_path))

if __name__ == "__main__":
    classfication = DETECT(model, Input_shape, class_name, data_list, label_list, device)
    if data_path != '':
        classfication.detect_data(num_classes, save_path)

    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            pre_label = classfication.detect_image(image)
            print(pre_label)
