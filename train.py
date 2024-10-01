import os
import torch
import torch.nn as nn
from utils.utils import read_split_data, get_lr
from utils.callbacks import LossHistory
from utils.dataloader import MyDataSet
from torchvision import transforms
from torch.utils.data import DataLoader
from model.net import MinstNet
from tqdm import tqdm
from torch.cuda.amp import autocast

# ----------------------------------------------------------------#
#                             训练
# ---------------------------------------------------------------#

# ------------------------------训练设备--------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------数据和保存路径-------------------------#
root_dir = 'test3/dataset'
out_dir = r'test3/logs'
# ----------------------------批量大小----------------------------#
batch_size = 512
# ----------------------------训练轮次----------------------------#
EPOCHS = 50
# -----------------------自动混合精度-----------------------------#
AMP = True

# ---------------------------数据读取-----------------------------#
train_dataset, train_images_label, val_dataset, val_images_label = read_split_data(root_dir, out_dir)
data_transform = {
    "train": transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
train_data_set = MyDataSet(images_path=train_dataset,
                           images_class=train_images_label,
                           input_shape=(28, 28),
                           transforms=data_transform['train'])

val_data_set = MyDataSet(images_path=val_dataset,
                         images_class=val_images_label,
                         input_shape=(28, 28),
                         transforms=data_transform['val'])

train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False,
                          collate_fn=train_data_set.collate_fn)
val_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False,
                        collate_fn=val_data_set.collate_fn)
# ---------------------------------模型加载------------------------#
model = MinstNet()
model.to(device)
# ----------------------------------优化器------------------------#
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4)
# --------------------------自动混合精度--------------------------#
if AMP:
    from torch.cuda.amp import GradScaler as GradScaler

    scaler = GradScaler()
else:
    scaler = None
# --------------------------实例化绘制曲线图------------------------#
line_history = LossHistory(out_dir, 'MinstNet')
# -------------------------最佳模型判断-----------------------------#
best_val_acc = 0
# -----------------------------模型训练-----------------------------#
for epoch in range(EPOCHS):
    train_loss = 0
    train_accuracy = 0
    val_loss = 0
    val_accuracy = 0
    # ------------------------训练阶段------------------------------#
    with tqdm(total=len(train_loader), desc=f'Train: Epoch{epoch + 1}/{EPOCHS}', postfix=dict,
              mininterval=0.2) as pbar:
        model.train()
        for iteration, data in enumerate(train_loader):
            image, target = data
            with torch.no_grad():
                image = image.to(device)
                target = target.to(device)
            optimizer.zero_grad()
            if AMP:
                with autocast():
                    output = model(image)
                    # ----------------交叉熵损失函数---------------------#
                    loss = nn.CrossEntropyLoss()(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(image)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
            # ----------------------损失值-----------------------------#
            train_loss += loss.item()

            # -----------------------统计正确个数-----------------------#
            with torch.no_grad():
                accuracy = torch.max(output, dim=1)[1]
                train_accuracy += torch.eq(accuracy, target).sum().item()

            # --------------实时更新损失及精度---------------------------#
            pbar.set_postfix(**{'total_loss': train_loss / (iteration + 1),
                                'accuracy': train_accuracy / batch_size,
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    # -------------------------------测试阶段------------------------------#
    with tqdm(total=len(val_loader), desc=f'Val: Epoch{epoch + 1} / {EPOCHS}', postfix=dict,
              mininterval=0.2) as pbar:
        model.eval()
        for iteration, data in enumerate(val_loader):
            image, target = data
            with torch.no_grad():
                image = image.to(device)
                target = target.to(device)
            output = model(image)
            loss = nn.CrossEntropyLoss()(output, target)

            val_loss += loss.item()

            accuracy = torch.max(output, dim=1)[1]
            val_accuracy += torch.eq(accuracy, target).sum().item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'accuracy': val_accuracy / batch_size,
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    # ------------------------绘制曲线-------------------------------------#
    line_history.append_loss(train_loss / len(train_loader), val_loss / len(val_loader),
                             train_accuracy / len(train_dataset), val_accuracy / len(val_dataset))

    print(
        f'\033[33mLoss: {train_loss / len(train_loader):.4f} || Accuracy: {train_accuracy / len(train_dataset):.4f} || Val loss: {val_loss / len(val_loader):.4f} || Val accuracy: {val_accuracy / len(val_dataset):.4f}\033[0m')

    # -------------------------------保存模型-----------------------------#
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_model_path = os.path.join(out_dir, f"MinstNet.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"\033[31mSave best model to best_model.pth\033[0m")