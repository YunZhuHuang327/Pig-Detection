#!/usr/bin/env python3
"""
根据submission_kaggle.csv可视化测试图片的检测结果
在每张图片上画出边界框和置信度分数
"""

import cv2
import csv
from pathlib import Path
import numpy as np
from tqdm import tqdm


def draw_predictions(image_path, detections, output_path, conf_threshold=0.3):
    """
    在图片上画出检测框
    
    Args:
        image_path: 输入图片路径
        detections: 检测结果列表 [(conf, left, top, width, height, class), ...]
        output_path: 输出图片路径
        conf_threshold: 置信度阈值（用于颜色编码）
    """
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠ 无法读取图片: {image_path}")
        return False
    
    # 为每个检测框画边界框和标签
    for det in detections:
        conf, left, top, width, height, class_id = det
        
        # 转换为整数坐标
        x1 = int(left)
        y1 = int(top)
        x2 = int(left + width)
        y2 = int(top + height)
        
        # 根据置信度设置颜色（绿色=高置信度，黄色=中等，红色=低）
        if conf >= 0.7:
            color = (0, 255, 0)  # 绿色
        elif conf >= 0.5:
            color = (0, 255, 255)  # 黄色
        else:
            color = (0, 165, 255)  # 橙色
        
        # 画边界框
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # 添加标签（置信度分数）
        label = f"{conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # 计算文字大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # 画文字背景
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width + 5, y1),
            color,
            -1
        )
        
        # 画文字
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),  # 黑色文字
            font_thickness
        )
    
    # 在图片顶部添加检测统计
    stats_text = f"Detections: {len(detections)}"
    cv2.putText(
        img,
        stats_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )
    
    # 保存图片
    cv2.imwrite(str(output_path), img)
    return True


def parse_prediction_string(pred_string):
    """
    解析PredictionString
    格式: conf_1 left_1 top_1 width_1 height_1 class_1 conf_2 ...
    """
    if not pred_string or pred_string.strip() == '':
        return []
    
    parts = pred_string.split()
    detections = []
    
    # 每6个数字一组
    for i in range(0, len(parts), 6):
        if i + 5 < len(parts):
            conf = float(parts[i])
            left = float(parts[i + 1])
            top = float(parts[i + 2])
            width = float(parts[i + 3])
            height = float(parts[i + 4])
            class_id = int(parts[i + 5])
            
            detections.append((conf, left, top, width, height, class_id))
    
    return detections


def visualize_all_predictions(csv_file, test_dir, output_dir, conf_threshold=0.3):
    """
    可视化所有预测结果
    
    Args:
        csv_file: submission_kaggle.csv文件路径
        test_dir: 测试图片目录
        output_dir: 输出目录
        conf_threshold: 最低置信度阈值（低于此值的不画出）
    """
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("开始可视化预测结果...")
    print(f"输入CSV: {csv_file}")
    print(f"测试图片目录: {test_dir}")
    print(f"输出目录: {output_dir}")
    print(f"置信度阈值: {conf_threshold}")
    print()
    
    # 读取CSV文件
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"总图片数: {len(rows)}")
    
    # 统计信息
    total_detections = 0
    images_with_detections = 0
    failed_images = []
    
    # 处理每张图片
    for row in tqdm(rows, desc="可视化进度"):
        image_id = row['Image_ID']
        pred_string = row['PredictionString']
        
        # 构造图片文件名（8位数字补零）
        image_name = f"{int(image_id):08d}.jpg"
        image_path = test_dir / image_name
        
        if not image_path.exists():
            failed_images.append(image_name)
            continue
        
        # 解析检测结果
        detections = parse_prediction_string(pred_string)
        
        # 过滤低置信度的检测
        detections = [d for d in detections if d[0] >= conf_threshold]
        
        if detections:
            images_with_detections += 1
            total_detections += len(detections)
        
        # 画出检测框
        output_path = output_dir / image_name
        success = draw_predictions(image_path, detections, output_path, conf_threshold)
        
        if not success:
            failed_images.append(image_name)
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("可视化完成！")
    print("=" * 80)
    print(f"处理图片数: {len(rows)}")
    print(f"有检测的图片: {images_with_detections}")
    print(f"总检测框数: {total_detections}")
    print(f"平均每张: {total_detections / len(rows):.1f} 个检测框")
    print(f"输出目录: {output_dir}")
    
    if failed_images:
        print(f"\n⚠ {len(failed_images)} 张图片处理失败")
        if len(failed_images) <= 10:
            for img in failed_images:
                print(f"  - {img}")
    
    return output_dir


def create_sample_grid(output_dir, num_samples=9):
    """
    创建一个网格图，显示若干样本结果
    """
    output_dir = Path(output_dir)
    image_files = sorted(list(output_dir.glob("*.jpg")))[:num_samples]
    
    if not image_files:
        print("没有找到可视化图片")
        return
    
    print(f"\n创建样本网格图（前{len(image_files)}张）...")
    
    # 读取所有图片
    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            # 调整大小以便显示
            img = cv2.resize(img, (400, 300))
            images.append(img)
    
    if not images:
        return
    
    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(len(images))))
    
    # 创建网格
    rows = []
    for i in range(0, len(images), grid_size):
        row_images = images[i:i+grid_size]
        # 补齐不足的位置
        while len(row_images) < grid_size:
            row_images.append(np.zeros_like(images[0]))
        row = np.hstack(row_images)
        rows.append(row)
    
    grid = np.vstack(rows)
    
    # 保存网格图
    grid_path = output_dir.parent / f"sample_grid_{grid_size}x{grid_size}.jpg"
    cv2.imwrite(str(grid_path), grid)
    print(f"✓ 样本网格图已保存: {grid_path}")


if __name__ == "__main__":
    # 文件路径
    csv_file = "/DATA1/yunzhu/DETR+1024/CV_HW1/runs/train/rfdetr_small_pig_detection_multigpu_20251010_122140/submission_kaggle.csv"
    test_dir = "test/img"
    output_dir = "runs/train/rfdetr_small_pig_detection_multigpu_20251010_122140/visualizations"
    
    # 可视化所有图片
    output_path = visualize_all_predictions(
        csv_file=csv_file,
        test_dir=test_dir,
        output_dir=output_dir,
        conf_threshold=0.3  # 只显示置信度 >= 0.3 的检测
    )
    
    # 创建样本网格图
    create_sample_grid(output_path, num_samples=16)
    
    print(f"\n可以查看可视化结果:")
    print(f"  单张图片: {output_dir}/")
    print(f"  网格预览: {output_dir}/../sample_grid_4x4.jpg")
