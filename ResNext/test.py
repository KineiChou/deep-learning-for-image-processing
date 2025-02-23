import torch
def main():

    print(torch.cuda.is_available())  # 输出 True 表示CUDA可用
    print(torch.cuda.get_device_name(0))  # 输出显卡型号（如 "NVIDIA GeForce RTX 3090"）
    
if __name__ == "__main__":
    main()