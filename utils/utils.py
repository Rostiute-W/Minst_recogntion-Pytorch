import os
import numpy as np
import random
import numbers
from PIL import Image

# -----------------------------------------------------------------#
#                           数据集划分
# -----------------------------------------------------------------#


def read_split_data(root, outdir, val_rate=0.2):
    '''
    :param root: 根目录
    :param val_rate: 验证机比例
    :return: 训练集、训练集标签、验证集、验证集标签
    '''

    random.seed(0)  # 保证随机结果可复现
    root2 = os.path.join(root, "train")
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    sets = ["test"]

    # --------------------遍历文件夹，一个文件夹对应一个类别----------------------#
    classes = [cla for cla in os.listdir(root2) if os.path.isdir(os.path.join(root2, cla))]
    # ---------------------------排序，保证顺序一致-----------------------------#
    classes.sort()
    # --------------------------------生成对应的txt文件-------------------------#
    with open(os.path.join(outdir, 'cls_classes.txt'), 'w') as f:
        for item in classes:
            f.write(str(item) + '\n')
    kind, _ = get_classes(os.path.join(outdir, 'cls_classes.txt'))

    for se in sets:
        list_file = open(os.path.join(outdir, 'cls_' + se + '.txt'), 'w')

        datasets_path_t = os.path.join(root, se)
        types_name = os.listdir(datasets_path_t)
        for type_name in types_name:
            if type_name not in kind:
                continue
            cls_id = kind.index(type_name)

            photos_path = os.path.join(datasets_path_t, type_name)
            photos_name = os.listdir(photos_path)
            for photo_name in photos_name:
                _, postfix = os.path.splitext(photo_name)
                if postfix not in ['.jpg', '.png', '.jpeg']:
                    continue
                list_file.write(str(cls_id) + ";" + '%s' % (os.path.join(photos_path, photo_name)))
                list_file.write('\n')
        list_file.close()
    # -------------生成类别名称以及对应的标签----------------#
    class_indices = dict((k, v) for v, k in enumerate(classes))

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in classes:
        cla_path = os.path.join(root2, cla)
        # -----------------遍历获取supported支持的所有文件路径--------------------------#
        images = [os.path.join(root2, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # --------------------------获取该类别对应的索引-------------------------------#
        image_class = class_indices[cla]
        # --------------------------记录该类别的样本数量-------------------------------#
        every_class_num.append(len(images))
        # --------------------------按比例随机采样验证样本-----------------------------#
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  
                train_images_path.append(img_path)
                train_images_label.append(image_class)
      
    print(f"训练集样本数量：{len(train_images_path)}")
    print(f"验证集样本数量：{len(val_images_path)}")
                
    return train_images_path, train_images_label, val_images_path, val_images_label
 
 
#------------------------------------------------------------------# 
#                       常用函数
# -----------------------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image
    
def resize_img(image, size, Resize = True):
    w, h = size
    iw, ih = image.size
    if Resize:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        if h == w:
            new_image = resize(image, h)
        else:
            new_image = resize(image, [h ,w])
        new_image = center_crop(new_image, [h ,w])
    return new_image

def resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)
    
def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)

def crop(img, i, j, h, w):
    return img.crop((j, i, j + w, i + h))