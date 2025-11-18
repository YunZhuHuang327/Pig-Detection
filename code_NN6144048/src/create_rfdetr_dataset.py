"""
创建RF-DETR数据集
将train/img分割为8:2的训练和验证集，并生成COCO格式的标注文件
DETR系列模型使用COCO格式的JSON标注
"""

import os
import json
import shutil
from pathlib import Path
import random
from PIL import Image
from datetime import datetime

def parse_gt_file(gt_path):
    """解析gt.txt文件
    格式: image_id, x, y, width, height
    返回: {image_id: [(x, y, w, h), ...], ...}
    """
    annotations = {}
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 5:
                continue
            
            img_id = int(parts[0])
            x, y, w, h = map(int, parts[1:])
            
            if img_id not in annotations:
                annotations[img_id] = []
            annotations[img_id].append((x, y, w, h))
    
    return annotations

def create_coco_annotations(img_list, annotations_dict, img_dir, category_id=1):
    """
    创建COCO格式的标注
    
    Args:
        img_list: 图片文件名列表
        annotations_dict: {image_id: [(x, y, w, h), ...]} 格式的标注字典
        img_dir: 图片目录路径
        category_id: 类别ID (默认1代表pig)
    
    Returns:
        COCO格式的字典
    """
    coco_format = {
        "info": {
            "description": "Pig Detection Dataset for RF-DETR",
            "version": "1.0",
            "year": 2025,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "pig",
                "supercategory": "animal"
            }
        ]
    }
    
    annotation_id = 1
    
    for idx, img_filename in enumerate(img_list, start=1):
        # 从文件名提取图片ID (例如: 00000001.jpg -> 1)
        img_id = int(img_filename.split('.')[0])
        
        # 获取图片尺寸
        img_path = os.path.join(img_dir, img_filename)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Warning: Cannot read image {img_filename}: {e}")
            continue
        
        # 添加图片信息
        image_info = {
            "id": idx,  # COCO格式中的图片ID
            "file_name": img_filename,
            "width": width,
            "height": height,
            "original_id": img_id  # 保留原始图片ID
        }
        coco_format["images"].append(image_info)
        
        # 添加该图片的所有标注
        if img_id in annotations_dict:
            for bbox in annotations_dict[img_id]:
                x, y, w, h = bbox
                
                # DETR使用的是COCO格式bbox: [x_min, y_min, width, height]
                # 确保bbox在图片范围内
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                
                if w <= 0 or h <= 0:
                    continue
                
                annotation = {
                    "id": annotation_id,
                    "image_id": idx,  # 关联到COCO格式的图片ID
                    "category_id": category_id,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                    "segmentation": []  # DETR主要使用bbox，segmentation可为空
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1
    
    return coco_format

def create_rfdetr_dataset(
    train_img_dir,
    gt_file,
    output_dir,
    train_ratio=0.8,
    seed=42
):
    """
    创建RF-DETR数据集
    
    Args:
        train_img_dir: 训练图片目录
        gt_file: GT标注文件路径
        output_dir: 输出数据集目录
        train_ratio: 训练集比例 (默认0.8)
        seed: 随机种子
    """
    random.seed(seed)
    
    print("=" * 60)
    print("Creating RF-DETR Dataset")
    print("=" * 60)
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    (output_path / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "annotations").mkdir(parents=True, exist_ok=True)
    
    # 解析GT文件
    print(f"\n[1/5] Parsing GT file: {gt_file}")
    annotations = parse_gt_file(gt_file)
    print(f"   Found annotations for {len(annotations)} images")
    total_boxes = sum(len(boxes) for boxes in annotations.values())
    print(f"   Total bounding boxes: {total_boxes}")
    
    # 获取所有图片文件
    print(f"\n[2/5] Scanning images from: {train_img_dir}")
    all_images = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.jpg')])
    print(f"   Found {len(all_images)} images")
    
    # 随机打乱并分割
    print(f"\n[3/5] Splitting dataset (train:val = {train_ratio}:{1-train_ratio})")
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)
    train_images = sorted(all_images[:split_idx])
    val_images = sorted(all_images[split_idx:])
    
    print(f"   Training set: {len(train_images)} images")
    print(f"   Validation set: {len(val_images)} images")
    
    # 复制图片文件
    print(f"\n[4/5] Copying images...")
    for img in train_images:
        shutil.copy2(
            os.path.join(train_img_dir, img),
            output_path / "train" / img
        )
    print(f"   Copied {len(train_images)} training images")
    
    for img in val_images:
        shutil.copy2(
            os.path.join(train_img_dir, img),
            output_path / "val" / img
        )
    print(f"   Copied {len(val_images)} validation images")
    
    # 生成COCO格式标注
    print(f"\n[5/5] Generating COCO format annotations...")
    
    # 训练集标注
    train_coco = create_coco_annotations(
        train_images,
        annotations,
        train_img_dir
    )
    train_ann_file = output_path / "annotations" / "instances_train.json"
    with open(train_ann_file, 'w') as f:
        json.dump(train_coco, f, indent=2)
    print(f"   Training annotations saved to: {train_ann_file}")
    print(f"   - Images: {len(train_coco['images'])}")
    print(f"   - Annotations: {len(train_coco['annotations'])}")
    
    # 验证集标注
    val_coco = create_coco_annotations(
        val_images,
        annotations,
        train_img_dir
    )
    val_ann_file = output_path / "annotations" / "instances_val.json"
    with open(val_ann_file, 'w') as f:
        json.dump(val_coco, f, indent=2)
    print(f"   Validation annotations saved to: {val_ann_file}")
    print(f"   - Images: {len(val_coco['images'])}")
    print(f"   - Annotations: {len(val_coco['annotations'])}")
    
    # 创建数据集配置文件
    config = {
        "dataset_name": "pig_detection_rfdetr",
        "task": "object_detection",
        "format": "COCO",
        "num_classes": 1,
        "classes": ["pig"],
        "train": {
            "images": "train/",
            "annotations": "annotations/instances_train.json",
            "num_images": len(train_images),
            "num_annotations": len(train_coco['annotations'])
        },
        "val": {
            "images": "val/",
            "annotations": "annotations/instances_val.json",
            "num_images": len(val_images),
            "num_annotations": len(val_coco['annotations'])
        },
        "split_ratio": {
            "train": train_ratio,
            "val": 1 - train_ratio
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            "Dataset format: COCO (compatible with DETR/RF-DETR)",
            "Bounding box format: [x_min, y_min, width, height]",
            "Category ID: 1 (pig)",
            "Images are in JPG format"
        ]
    }
    
    config_file = output_path / "dataset_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n   Dataset configuration saved to: {config_file}")
    
    # 创建README
    readme_content = f"""# RF-DETR Pig Detection Dataset

## Dataset Information
- **Task**: Object Detection
- **Format**: COCO JSON
- **Number of Classes**: 1 (pig)
- **Created**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Structure
```
rfdetr_dataset/
├── train/                          # Training images ({len(train_images)} images)
├── val/                            # Validation images ({len(val_images)} images)
├── annotations/
│   ├── instances_train.json        # Training annotations (COCO format)
│   └── instances_val.json          # Validation annotations (COCO format)
├── dataset_config.json             # Dataset configuration
└── README.md                       # This file
```

## Dataset Statistics
- **Total Images**: {len(all_images)}
- **Training Set**: {len(train_images)} images ({len(train_coco['annotations'])} annotations)
- **Validation Set**: {len(val_images)} images ({len(val_coco['annotations'])} annotations)
- **Split Ratio**: {train_ratio*100:.0f}% train / {(1-train_ratio)*100:.0f}% val

## Annotation Format
COCO JSON format compatible with DETR/RF-DETR:
- **Bounding Box**: [x_min, y_min, width, height]
- **Category ID**: 1 (pig)
- **Coordinate System**: Top-left origin

## Usage with RF-DETR

### 1. Training
```python
from torchvision.datasets import CocoDetection

# Load training dataset
train_dataset = CocoDetection(
    root='rfdetr_dataset/train',
    annFile='rfdetr_dataset/annotations/instances_train.json'
)

# Load validation dataset
val_dataset = CocoDetection(
    root='rfdetr_dataset/val',
    annFile='rfdetr_dataset/annotations/instances_val.json'
)
```

### 2. Configuration
Update your RF-DETR config file:
```yaml
data:
  train_dataloader:
    dataset:
      img_folder: rfdetr_dataset/train
      ann_file: rfdetr_dataset/annotations/instances_train.json
  val_dataloader:
    dataset:
      img_folder: rfdetr_dataset/val
      ann_file: rfdetr_dataset/annotations/instances_val.json

model:
  num_classes: 1  # pig only
```

## Notes
- All images are in JPG format
- Bounding boxes are in absolute pixel coordinates
- No image preprocessing has been applied (resize/normalize should be done during training)
- Dataset is compatible with all DETR-based models (DETR, Deformable-DETR, RF-DETR, etc.)
"""
    
    readme_file = output_path / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"   README saved to: {readme_file}")
    
    print("\n" + "=" * 60)
    print("Dataset creation completed successfully!")
    print("=" * 60)
    print(f"\nDataset location: {output_path.absolute()}")
    print(f"\nTo use this dataset with RF-DETR:")
    print(f"  - Training images: {output_path / 'train'}")
    print(f"  - Training annotations: {output_path / 'annotations' / 'instances_train.json'}")
    print(f"  - Validation images: {output_path / 'val'}")
    print(f"  - Validation annotations: {output_path / 'annotations' / 'instances_val.json'}")
    print("\n")

if __name__ == "__main__":
    # 配置路径
    TRAIN_IMG_DIR = "train/img"
    GT_FILE = "train/gt.txt"
    OUTPUT_DIR = "rfdetr_dataset"
    TRAIN_RATIO = 0.8
    SEED = 42
    
    # 创建数据集
    create_rfdetr_dataset(
        train_img_dir=TRAIN_IMG_DIR,
        gt_file=GT_FILE,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        seed=SEED
    )
