from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import visdom
import time

def main():
    viz = visdom.Visdom()
    # 数据预处理（适配 ResNeXt 的输入要求）

    transform_train = transforms.Compose([
        transforms.Resize(224),          # ResNeXt 默认输入为 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据集
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # 创建 DataLoader
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory= True)

    # 加载预训练 ResNeXt（自动下载权重）
    model = models.resnext50_32x4d(weights="DEFAULT")
    
    # 冻结所有卷积层
    for param in model.parameters():
        param.requires_grad = False
    # 仅训练全连接层
    model.fc.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 修改最后一层（适配 CIFAR-10 的 10 类别）
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Using {} device training.".format(device.type))
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    # 学习率调整策略（每 30 个 epoch 下降 10 倍）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    cur_batch_win = None
    cur_batch_win_opts = {
        'title': 'Epoch Loss Trace',
        'xlabel': 'Batch Number',
        'ylabel': 'Loss',
        'width': 1200,
        'height': 600,
    }

    # 训练轮数
    num_epochs = 100

    for epoch in range(num_epochs):
        
        model.train()
        loss_list, batch_list = [], []
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            start_time = time.time()
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss_list.append(loss.detach().cpu().item())
            batch_list.append(i+1)
            
            if i % 10 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))
                
            if viz.check_connection():
                cur_batch_win = viz.line(
                    torch.Tensor(loss_list), 
                    torch.Tensor(batch_list),
                    win=cur_batch_win, 
                    name='current_batch_loss',
                    update=(None if cur_batch_win is None else 'replace'),                 
                    opts=cur_batch_win_opts
                )
            # 反向传播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_time = time.time() - start_time
            print(f"Epoch 耗时： {epoch_time / 60:.2f} 分钟")
        # 更新学习率
        scheduler.step()
        
        
        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 打印统计信息
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Test Acc: {100*correct/total:.2f}%')
        
if __name__ == "__main__":
    main()