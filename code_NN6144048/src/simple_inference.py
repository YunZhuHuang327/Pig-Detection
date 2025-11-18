#!/usr/bin/env python3
"""
簡單的RF-DETR推理脚本 - 直接使用checkpoint進行推理
"""

import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 設置 PyTorch CUDA 記憶體優化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 添加RF-DETR路径
sys.path.insert(0, str(Path("rf-detr").absolute()))

from rfdetr import RFDETRSmall
import supervision as sv

def load_model_from_checkpoint(checkpoint_path):
    """
    從checkpoint載入模型
    """
    print(f"\n載入模型: {checkpoint_path}")
    
    # 載入checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 獲取模型配置
    args = checkpoint.get('args', None)
    num_classes = args.num_classes if args and hasattr(args, 'num_classes') else 1
    class_names = args.class_names if args and hasattr(args, 'class_names') else ['pig']
    
    print(f"  類別數: {num_classes}")
    print(f"  類別名稱: {class_names}")
    
    # 創建模型（不載入預訓練權重）
    model = RFDETRSmall(pretrain_weights=None)
    model.model.class_names = class_names
    
    # 重新初始化detection head以匹配正確的類別數
    model.model.reinitialize_detection_head(num_classes)
    
    # 載入state_dict
    state_dict = checkpoint['model']
    
    # 處理DDP的"module."前綴（如果有的話）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # 載入權重
    missing_keys, unexpected_keys = model.model.model.load_state_dict(new_state_dict, strict=False)
    
    # 報告載入狀態
    if len(unexpected_keys) > 0:
        print(f"  ⚠ Unexpected keys: {len(unexpected_keys)}")
    if len(missing_keys) > 0:
        important_missing = [k for k in missing_keys if 'ema' not in k.lower()]
        if important_missing:
            print(f"  ⚠ Missing keys: {len(important_missing)}")
    
    model.model.model.eval()
    
    print("✓ 模型載入成功")
    return model


def run_inference(checkpoint_path, test_dir="test/img", output_dir=None, threshold=0.5):
    """
    在測試集上運行推理
    """
    print("\n" + "=" * 70)
    print("RF-DETR 測試集推理")
    print("=" * 70)
    
    # 載入模型
    model = load_model_from_checkpoint(checkpoint_path)
    
    # 設置輸出目錄
    if output_dir is None:
        output_dir = Path(checkpoint_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取測試圖片
    test_path = Path(test_dir)
    test_images = sorted(test_path.glob("*.jpg"))
    
    print(f"\n測試集: {test_dir}")
    print(f"找到 {len(test_images)} 張測試圖片")
    print(f"置信度閾值: {threshold}")
    print(f"輸出目錄: {output_dir}")
    
    if len(test_images) == 0:
        print("❌ 沒有找到測試圖片")
        return
    
    # 推理
    print("\n開始推理...")
    all_predictions = []
    submission_lines = []
    total_detections = 0
    
    for img_path in tqdm(test_images, desc="處理"):
        try:
            # 讀取圖片
            image = Image.open(img_path).convert('RGB')
            
            # 推理 (返回supervision.Detections對象)
            results = model.predict(image, threshold=threshold)
            
            # 解析結果
            predictions = {
                'image_id': img_path.stem,
                'file_name': img_path.name,
                'detections': []
            }
            
            submission_line = img_path.name
            
            # 檢查檢測結果
            if results is not None and len(results.xyxy) > 0:
                num_det = len(results.xyxy)
                total_detections += num_det
                
                for i in range(num_det):
                    bbox = results.xyxy[i]
                    x1, y1, x2, y2 = bbox
                    score = float(results.confidence[i]) if results.confidence is not None else 0.0
                    
                    predictions['detections'].append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'score': score,
                        'category': 'pig',
                        'category_id': 0
                    })
                    
                    submission_line += f" {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {score:.6f}"
            
            all_predictions.append(predictions)
            submission_lines.append(submission_line)
            
        except Exception as e:
            print(f"\n⚠ 處理 {img_path.name} 時出錯: {e}")
            all_predictions.append({
                'image_id': img_path.stem,
                'file_name': img_path.name,
                'detections': []
            })
            submission_lines.append(img_path.name)
    
    # 保存結果
    print(f"\n總檢測數: {total_detections}")
    print(f"平均每張圖片: {total_detections / len(test_images):.1f} 個目標")
    
    # 保存predictions.json
    predictions_file = output_dir / "predictions.json"
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 預測結果已保存: {predictions_file}")
    
    # 保存submission.txt
    submission_file = output_dir / "submission.txt"
    with open(submission_file, 'w') as f:
        f.write('\n'.join(submission_lines))
    print(f"✓ 提交文件已保存: {submission_file}")
    
    print("\n" + "=" * 70)
    print("推理完成！")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RF-DETR測試集推理")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/train/rfdetr_small_pig_detection_multigpu_20251010_122140/checkpoint_best_regular.pth",
        help="模型checkpoint路徑"
    )
    parser.add_argument("--test_dir", type=str, default="test/img", help="測試圖片目錄")
    parser.add_argument("--output_dir", type=str, help="輸出目錄（如果不指定，使用checkpoint所在目錄）")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度閾值")
    
    args = parser.parse_args()
    
    # 檢查checkpoint
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        sys.exit(1)
    '''
    # 檢查checkpoint，使用指定的path
    checkpoint_path_str = "/DATA1/yunzhu/DETR+1024/CV_HW1/runs/train/rfdetr_small_pig_detection_multigpu_20251010_110627/checkpoint_best_ema.pth"
    checkpoint_path = Path(checkpoint_path_str) # <--- 把字串轉換成 Path 物件
    if not checkpoint_path.exists():
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        sys.exit(1)
    '''
    
    # 運行推理
    run_inference(
        checkpoint_path=checkpoint_path,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        threshold=args.threshold
    )
