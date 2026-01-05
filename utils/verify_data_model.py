# utils/verify_data_model.py
# 验证CIFAR-10数据集与ResNet模型的协同工作能力
from utils.data_process import load_cifar10
from utils.model_utils import load_and_verify_model
import torch

def main():
    print("="*65)
    print("🚀 实验前核心校验：验证数据+模型的协同工作能力")
    print("="*65)

    # 步骤1：加载CIFAR-10数据集
    print("\n📌 步骤1：加载真实CIFAR-10数据集")
    train_loader, test_loader = load_cifar10()
    # 校验数据集基础信息
    data_shape = next(iter(train_loader))[0].shape
    assert data_shape == (32, 3, 32, 32), f"数据形状异常！预期(32,3,32,32)，实际{data_shape}"
    print(f"✅ 数据集加载成功！")
    print(f"   - 训练集样本数：{len(train_loader.dataset)}（预期50000）")
    print(f"   - 测试集样本数：{len(test_loader.dataset)}（预期10000）")
    print(f"   - 单批数据形状：{data_shape}（符合模型输入要求）")

    # 步骤2：导入并验证ResNet模型
    print("\n📌 步骤2：导入并验证ResNet模型")
    model = load_and_verify_model(model_name="resnet20")

    # 步骤3：核心校验：验证数据+模型协同工作能力
    print("\n📌 步骤3：验证数据+模型协同工作能力")
    # 1. 设备协同：确保数据和模型在同一设备（GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   - 统一设备：数据/模型均移至 {device}")
    # 2. 取真实数据并移至目标设备
    data_iter = iter(train_loader)
    images, _ = next(data_iter)
    images = images.to(device)
    # 3. 执行前向传播（模拟训练时的核心计算流程）
    with torch.no_grad():  # 禁用梯度，仅校验不训练
        outputs = model(images)
    # 4. 校验协同结果（形状+任务匹配）
    assert outputs.shape == (32, 10), f"协同工作失败！输出形状预期(32,10)，实际{outputs.shape}"
    print(f"✅ 数据+模型协同工作能力验证通过！")
    print(f"   - 模型输出形状：{outputs.shape}（匹配CIFAR-10 10分类任务）")

    print("\n" + "="*65)
    print("🎉 所有校验通过！数据与模型可正常配合工作，可启动正式实验！")
    print("="*65)

if __name__ == "__main__":
    main()