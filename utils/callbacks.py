import os
import scipy.signal
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------#
#                            绘制损失曲线-准确率曲线
# --------------------------------------------------------------------------------#

class LossHistory():
    def __init__(self, log_dir, backbone):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        self.accuracy = []
        self.val_accuracy = []
        self.path = os.path.join(self.log_dir, 'loss', backbone)

        if not os.path.exists(self.path):
            os.mkdir(self.path)

    # -------------------生成loss.txt和val_los.txt---------------------------------#
    def append_loss(self, loss, val_loss, accuracy, val_accuracy):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        self.accuracy.append(accuracy)
        self.val_accuracy.append(val_accuracy)

        with open(os.path.join(self.path, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss) + '\n')

        with open(os.path.join(self.path, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss) + '\n')

        self.curves_plot()

    # -------------------------------损失和精度曲线绘制-------------------------------------#
    def curves_plot(self):
        iters = range(1, len(self.losses) + 1)
        # -----------------------------------绘制损失曲线-----------------------------------#
        plt.figure()
        plt.plot(iters, self.losses, label='train loss', color='red', linewidth=2)
        plt.plot(iters, self.val_loss, label='val loss', color='coral', linewidth=2)
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), label='smooth train loss', color='green',
                     linestyle='--', linewidth=2)
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), label='smooth val loss', color='#8B4513',
                     linestyle='--', linewidth=2)
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.path, "epoch_loss.png"))
        plt.cla()
        plt.close("all")

        # -------------------------绘制准确率曲线-----------------------------------#
        plt.figure()
        plt.plot(iters, self.accuracy, label='acc', color='red', linewidth=2)
        plt.plot(iters, self.val_accuracy, label='val acc', color='coral', linewidth=2)

        try:
            plt.plot(iters, scipy.signal.savgol_filter(self.accuracy, num, 3), label='smooth acc', color='green',
                     linestyle='--', linewidth=2)
            plt.plot(iters, scipy.signal.savgol_filter(self.val_accuracy, num, 3), label='smooth val acc',
                     color='#8B4513', linestyle='--', linewidth=2)
        except:
            pass

        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.path, "epoch_accuracy.png"))
        plt.cla()
        plt.close("all")
