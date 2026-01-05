# 负责：数据集处理+加载核心逻辑
# 负责：数据集处理+加载核心逻辑
import os  # 新增：用于路径处理
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_cifar10():
    # 训练集预处理（数据增强）
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize(  # 标准化
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # 测试集预处理（仅标准化）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # ========== 新增：修复路径问题，避免重复下载 ==========
    # 获取当前脚本所在目录，拼接绝对路径（解决相对路径不一致问题）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(current_dir, '../data')
    # 确保data文件夹存在，不存在则自动创建
    os.makedirs(data_root, exist_ok=True)

    # 自动下载/加载CIFAR-10到data文件夹
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root,  # 修改：使用拼接后的绝对路径（原：../data）
        train=True,  # 加载训练集
        download=True,  # 自动下载（不存在则下载，存在则跳过）
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root,  # 修改：使用拼接后的绝对路径（原：../data）
        train=False,  # 加载测试集
        download=True,
        transform=test_transform
    )

    # 构建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # 适配你的RTX3060显存
        shuffle=True,  # 训练集打乱
        num_workers=4  # 多线程加载（根据电脑配置调整，一般设2/4）
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,  # 测试集不打乱
        num_workers=2
    )

    return train_loader, test_loader