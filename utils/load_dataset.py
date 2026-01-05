# 负责：启动调用data_process的功能
from utils.data_process import load_cifar10
if __name__ == "__main__":
    print("开始加载CIFAR-10数据集...")  # 新增打印，确认代码执行
    train_loader, test_loader = load_cifar10()
    print("数据集加载完成！训练集样本数：", len(train_loader.dataset))

