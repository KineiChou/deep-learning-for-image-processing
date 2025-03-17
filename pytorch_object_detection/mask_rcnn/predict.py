import os
import time
import json

import numpy as np
from PIL import Image
import pandas as pd
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnext50_32x4d_fpn_backbone
from draw_box_utils import draw_objs


def create_model(num_classes, box_thresh=0.5):
    backbone = resnext50_32x4d_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def calculate_equivalent_diameters(masks, pixel_to_meter_ratio, output_excel_path):
    """
    计算Mask R-CNN预测实例的等效直径并保存到Excel
    
    参数：
    masks : numpy.ndarray
        Mask R-CNN输出的掩膜数组，形状为[H, W, N]，N为实例数量
    pixel_to_meter_ratio : float
        像素到现实世界的转换比例（米/像素）
    output_excel_path : str
        输出Excel文件路径
    
    返回：
    pandas.DataFrame
        包含计算结果的数据框
    """
    # 初始化结果存储
    results = []
    
    # 转换掩膜数据类型（确保为二值图像）
    masks = masks.astype(bool)
    
    # 遍历每个实例
    for i in range(masks.shape[0]):
        mask = masks[i, :, :]
        
        try:
            # 计算掩膜属性
            labeled = label(mask)
            regions = regionprops(labeled)
            
            # 筛选有效区域（处理可能的分割碎片）
            main_region = max(regions, key=lambda x: x.area)
            
            # 计算像素面积
            pixel_area = main_region.area
            
            # 转换为实际面积（平方米）
            real_area = pixel_area * (pixel_to_meter_ratio ** 2)
            
            # 计算等效直径
            equivalent_diameter = 2 * np.sqrt(real_area / np.pi)
            
            # 记录结果
            results.append({
                "Instance_ID": i+1,
                "Pixel_Area": pixel_area,
                "Real_Area_m2": real_area,
                "Equivalent_Diameter_m": equivalent_diameter,
                "Bbox_Height_m": (main_region.bbox[2] - main_region.bbox[0]) * pixel_to_meter_ratio,
                "Bbox_Width_m": (main_region.bbox[3] - main_region.bbox[1]) * pixel_to_meter_ratio
            })
            
        except Exception as e:
            print(f"处理实例 {i+1} 时发生错误: {str(e)}")
            continue

    # 转换为数据框
    df = pd.DataFrame(results)
    
    # 保存到Excel
    df.to_excel(output_excel_path, index=False)
    
    return df

def main():
    num_classes = 1  # 不包含背景
    box_thresh = 0.5
    weights_path = "./save_weights/model_35.pth"
    img_path = "./2-1.png"
    label_json_path = './coco91_indices.json'
    pixel_ratio = 0.3/440  # 假设每个像素对应0.005米（5毫米/像素）
    output_path = "实例尺寸分析1.xlsx"

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu',weights_only=False)
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # load image
    assert os.path.exists(img_path), f"{img_path} does not exits."
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        predict_mask = predictions["masks"].to("cpu").numpy()
        predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")
            return
        
        df = calculate_equivalent_diameters(
        predict_mask,
        pixel_ratio,
        output_path
    )
        print("计算结果：")
        print(df)

        plot_img = draw_objs(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             category_index=category_index,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        plot_img.save("test_result.jpg")


if __name__ == '__main__':
    main()

