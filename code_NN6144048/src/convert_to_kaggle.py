#!/usr/bin/env python3
"""
将submission.txt转换为Kaggle提交格式
从：image_name x1 y1 x2 y2 score ...
到：Image_ID,PredictionString
    1,conf_1 bb_left_1 bb_top_1 bb_width_1 bb_height_1 class_1 conf_2 ...
"""

import csv
from pathlib import Path

def convert_to_kaggle_format(input_file, output_file):
    """
    转换格式
    
    当前格式：
        00000001.jpg x1 y1 x2 y2 score x1 y1 x2 y2 score ...
    
    Kaggle格式：
        Image_ID,PredictionString
        1,conf_1 bb_left_1 bb_top_1 bb_width_1 bb_height_1 class_1 conf_2 bb_left_2 ...
    
    注意：
    - 图片ID从文件名提取（去掉.jpg）
    - 坐标从 (x1,y1,x2,y2) 转换为 (left,top,width,height)
    - 顺序改为: conf, left, top, width, height, class
    - class固定为0（代表猪）
    """
    
    print("开始转换格式...")
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")
    
    results = []
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 1:
                continue
            
            # 获取图片名称和ID
            image_name = parts[0]
            # 从文件名提取ID（例如：00000001.jpg -> 1）
            image_id = int(image_name.replace('.jpg', ''))
            
            # 转换所有检测框
            prediction_parts = []
            num_detections = (len(parts) - 1) // 5
            
            for i in range(num_detections):
                idx = 1 + i * 5
                if idx + 4 < len(parts):
                    x1 = float(parts[idx])
                    y1 = float(parts[idx + 1])
                    x2 = float(parts[idx + 2])
                    y2 = float(parts[idx + 3])
                    score = float(parts[idx + 4])
                    
                    # 转换坐标格式：(x1,y1,x2,y2) -> (left,top,width,height)
                    bb_left = x1
                    bb_top = y1
                    bb_width = x2 - x1
                    bb_height = y2 - y1
                    class_id = 0  # 固定为0（猪）
                    
                    # Kaggle格式顺序：conf left top width height class
                    prediction_parts.extend([
                        str(score),
                        str(bb_left),
                        str(bb_top),
                        str(bb_width),
                        str(bb_height),
                        str(class_id)
                    ])
            
            # 用空格连接所有检测（注意：不是逗号）
            prediction_string = ' '.join(prediction_parts)
            
            results.append({
                'Image_ID': image_id,
                'PredictionString': prediction_string
            })
            
            if line_num % 500 == 0:
                print(f"  已处理 {line_num} 张图片...")
    
    # 按Image_ID排序
    results.sort(key=lambda x: x['Image_ID'])
    
    # 写入CSV文件
    print(f"\n写入CSV文件...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Image_ID', 'PredictionString'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ 转换完成！")
    print(f"  总图片数: {len(results)}")
    print(f"  输出文件: {output_file}")
    
    # 显示前几行作为示例
    print(f"\n前3行示例：")
    print("-" * 80)
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 4:  # header + 3行
                if i == 0:
                    print(f"Header: {line.strip()}")
                else:
                    # 截断过长的PredictionString
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        pred = parts[1][:100] + '...' if len(parts[1]) > 100 else parts[1]
                        print(f"Row {i}: Image_ID={parts[0]}, PredictionString={pred}")


def verify_format(csv_file, original_file):
    """验证转换后的格式是否正确"""
    
    print(f"\n验证格式...")
    
    # 读取CSV
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"✓ CSV格式正确")
    print(f"  行数: {len(rows)}")
    
    # 检查第一行
    if rows:
        first_row = rows[0]
        pred_parts = first_row['PredictionString'].split()
        num_detections = len(pred_parts) // 6
        
        print(f"\n第一张图片 (Image_ID={first_row['Image_ID']}):")
        print(f"  检测框数量: {num_detections}")
        
        if num_detections > 0:
            print(f"  第一个检测框:")
            print(f"    confidence: {pred_parts[0]}")
            print(f"    bb_left: {pred_parts[1]}")
            print(f"    bb_top: {pred_parts[2]}")
            print(f"    bb_width: {pred_parts[3]}")
            print(f"    bb_height: {pred_parts[4]}")
            print(f"    class: {pred_parts[5]}")
    
    # 对比原始文件
    print(f"\n对比原始文件...")
    with open(original_file, 'r') as f:
        first_line = f.readline().strip().split()
        orig_detections = (len(first_line) - 1) // 5
    
    print(f"  原始检测数: {orig_detections}")
    print(f"  转换后检测数: {num_detections}")
    
    if orig_detections == num_detections:
        print(f"  ✓ 检测数量一致")
    else:
        print(f"  ⚠ 检测数量不一致！")
    
    return True


if __name__ == "__main__":
    # 文件路径
    input_file = "/DATA1/yunzhu/DETR+1024/CV_HW1/runs/train/rfdetr_small_pig_detection_multigpu_20251010_110627/submission.txt"
    output_file = "/DATA1/yunzhu/DETR+1024/CV_HW1/runs/train/rfdetr_small_pig_detection_multigpu_20251010_110627/submission_kaggle.csv"
    
    # 转换
    convert_to_kaggle_format(input_file, output_file)
    
    # 验证
    verify_format(output_file, input_file)
    
    print("\n" + "=" * 80)
    print("转换完成！可以提交到Kaggle了！")
    print("=" * 80)
    print(f"\n文件位置: {output_file}")
