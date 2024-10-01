import os
import idx2numpy
from PIL import Image
from tqdm import tqdm

# --------------------------------------------------------------#
#                      将IDX文件转换为图片
# --------------------------------------------------------------#

# -------------------------设置IDX文件的路径---------------------#
train_images_path = 'test3/dataset/MNIST/train-images.idx3-ubyte'
train_labels_path = 'test3/dataset/MNIST/train-labels.idx1-ubyte'
test_images_path = 'test3/dataset/MNIST/t10k-images.idx3-ubyte'
test_labels_path = 'test3/dataset/MNIST/t10k-labels.idx1-ubyte'
# -------------------------保存图片根目录------------------------#
output_dir = 'test3/dataset'
# -------------------------检查文件是否存在----------------------#
for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件未找到: {path}")
os.makedirs(output_dir, exist_ok=True)
# ---------------------------保存为PNG格式----------------------#
def save_images(images, labels, dataset_type='train'):
    for label in range(10):
        label_dir = os.path.join(output_dir, dataset_type, str(label))
        os.makedirs(label_dir, exist_ok=True)

    num_samples = images.shape[0]
    print(f"\033[31m保存 {dataset_type} 集中的 {num_samples} 张图片\033[0m")
    for idx in tqdm(range(num_samples)):
        image = images[idx]
        label = labels[idx]
        # -----------------转换为PIL图像-------------------#
        img = Image.fromarray(image).convert('L')
        img_name = f"{idx}.png"
        img_path = os.path.join(output_dir, dataset_type, str(label), img_name)
        img.save(img_path)


if __name__ == '__main__':
    # ---------------------------加载训练集图像和标签----------------------------#
    train_images = idx2numpy.convert_from_file(train_images_path)
    train_labels = idx2numpy.convert_from_file(train_labels_path)
    # ---------------------------加载测试集图像和标签----------------------------#
    test_images = idx2numpy.convert_from_file(test_images_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path)
    # -------------------------------保存训练集图像------------------------------#
    save_images(train_images, train_labels, 'train')
    # -------------------------------保存测试集图像------------------------------#
    save_images(test_images, test_labels, 'test')

    print("\033[31mMNIST数据集已成功转换为PNG格式图片\033[0m")
