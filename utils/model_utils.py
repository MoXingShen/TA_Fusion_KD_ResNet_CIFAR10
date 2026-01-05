import torch
from models.resnet_cifar import ResNet20, ResNet110

def load_and_verify_model(model_name="resnet20"):
    """
    导入模型+验证核心逻辑（适配你的CIFAR-10数据）
    :param model_name: 模型名（resnet20/resnet110）
    :return: 初始化后的模型
    """
    # 适配你的RTX3060 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 当前运行设备：{device}")

    # 导入并初始化模型
    if model_name == "resnet20":
        model = ResNet20().to(device)
        expected_params = 272474  # ResNet20标准参数数
    elif model_name == "resnet110":
        model = ResNet110().to(device)
        expected_params = 1711626  # ResNet110标准参数数
    else:
        raise ValueError(f"仅支持resnet20/resnet110，你输入的是：{model_name}")

    # 验证模型参数（避免定义错误）
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == expected_params, f"模型参数错误！预期{expected_params}，实际{total_params}"
    print(f"✅ {model_name}模型导入成功！总参数数：{total_params:,}")

    # 测试前向传播（模拟你的CIFAR-10数据形状：32x32x3）
    test_input = torch.randn(32, 3, 32, 32).to(device)  # 匹配你的batch_size=32
    with torch.no_grad():
        output = model(test_input)
    assert output.shape == (32, 10), f"前向传播错误！预期(32,10)，实际{output.shape}"
    print(f"✅ 模型前向传播验证成功！输出形状：{output.shape}（匹配你的batch_size=32）")

    return model

def save_model(model, save_path="../checkpoints/model.pth"):
    """保存模型权重（相对路径，适配你的项目结构）"""
    torch.save(model.state_dict(), save_path)
    print(f"💾 模型已保存到：{save_path}")