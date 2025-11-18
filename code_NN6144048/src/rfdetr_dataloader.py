"""
RF-DETR数据加载器示例
展示如何使用COCO格式数据集训练RF-DETR
"""

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from PIL import Image
import numpy as np

class CocoDetectionWrapper(CocoDetection):
    """
    COCO数据集包装器，适配DETR模型输入格式
    """
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
    
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        
        # 转换标注格式为DETR所需格式
        target = {'image_id': image_id, 'annotations': target}
        
        # 处理边界框和标签
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for obj in target['annotations']:
            # COCO格式: [x_min, y_min, width, height]
            bbox = obj['bbox']
            # 转换为 [x_min, y_min, x_max, y_max]
            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h
            
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(obj['category_id'])
            areas.append(obj.get('area', w * h))
            iscrowd.append(obj.get('iscrowd', 0))
        
        # 转换为tensor
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': areas,
            'iscrowd': iscrowd,
            'orig_size': torch.as_tensor([int(img.height), int(img.width)]),
            'size': torch.as_tensor([int(img.height), int(img.width)])
        }
        
        # 应用transforms
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        return img, target


class Compose:
    """组合多个transforms"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """转换PIL Image到Tensor"""
    def __call__(self, image, target):
        image = T.ToTensor()(image)
        return image, target


class Normalize:
    """归一化"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target):
        image = T.Normalize(self.mean, self.std)(image)
        return image, target


class Resize:
    """调整图片大小"""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, target):
        # size: (height, width)
        h, w = self.size
        
        # 获取原始尺寸
        orig_w, orig_h = image.size
        
        # 调整图片大小
        image = T.Resize((h, w))(image)
        
        # 调整边界框
        if len(target['boxes']) > 0:
            boxes = target['boxes']
            scale_x = w / orig_w
            scale_y = h / orig_h
            
            boxes[:, 0::2] *= scale_x  # x坐标
            boxes[:, 1::2] *= scale_y  # y坐标
            
            target['boxes'] = boxes
        
        target['size'] = torch.tensor([h, w])
        
        return image, target


class RandomHorizontalFlip:
    """随机水平翻转"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            image = T.functional.hflip(image)
            
            if len(target['boxes']) > 0:
                boxes = target['boxes']
                w = image.size[0] if isinstance(image, Image.Image) else image.shape[2]
                
                # 翻转x坐标
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes
        
        return image, target


def make_transforms(image_set='train'):
    """创建数据增强pipeline"""
    
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if image_set == 'train':
        return Compose([
            RandomHorizontalFlip(p=0.5),
            Resize((360, 640)),  # (height, width)
            normalize,
        ])
    
    if image_set == 'val':
        return Compose([
            Resize((360, 640)),
            normalize,
        ])
    
    raise ValueError(f'unknown image set: {image_set}')


def build_dataset(image_set, dataset_root='rfdetr_dataset'):
    """
    构建数据集
    
    Args:
        image_set: 'train' 或 'val'
        dataset_root: 数据集根目录
    
    Returns:
        CocoDetectionWrapper dataset
    """
    img_folder = f'{dataset_root}/{image_set}'
    ann_file = f'{dataset_root}/annotations/instances_{image_set}.json'
    
    dataset = CocoDetectionWrapper(
        img_folder,
        ann_file,
        transforms=make_transforms(image_set)
    )
    
    return dataset


def collate_fn(batch):
    """
    自定义collate函数，处理不同数量的边界框
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # 将图片堆叠成batch
    images = torch.stack(images, dim=0)
    
    return images, targets


def create_dataloaders(
    dataset_root='rfdetr_dataset',
    batch_size=2,
    num_workers=4
):
    """
    创建训练和验证数据加载器
    
    Args:
        dataset_root: 数据集根目录
        batch_size: batch大小
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader
    """
    # 构建数据集
    train_dataset = build_dataset('train', dataset_root)
    val_dataset = build_dataset('val', dataset_root)
    
    print(f"Training dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def test_dataloader():
    """测试数据加载器"""
    print("=" * 70)
    print("Testing RF-DETR DataLoader")
    print("=" * 70)
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        dataset_root='rfdetr_dataset',
        batch_size=2,
        num_workers=0  # 测试时使用0避免多进程问题
    )
    
    # 测试训练集
    print("\n[1] Testing Training DataLoader...")
    images, targets = next(iter(train_loader))
    
    print(f"   Batch images shape: {images.shape}")
    print(f"   Number of targets: {len(targets)}")
    print(f"   First target keys: {targets[0].keys()}")
    print(f"   First target boxes shape: {targets[0]['boxes'].shape}")
    print(f"   First target labels shape: {targets[0]['labels'].shape}")
    print(f"   First target labels: {targets[0]['labels']}")
    
    # 检查数值范围
    print(f"\n   Image value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"   First image mean: {images[0].mean():.3f}")
    print(f"   First image std: {images[0].std():.3f}")
    
    # 测试验证集
    print("\n[2] Testing Validation DataLoader...")
    images, targets = next(iter(val_loader))
    
    print(f"   Batch images shape: {images.shape}")
    print(f"   Number of targets: {len(targets)}")
    print(f"   First target boxes shape: {targets[0]['boxes'].shape}")
    
    # 统计信息
    print("\n[3] Dataset Statistics...")
    total_boxes_train = 0
    for images, targets in train_loader:
        for target in targets:
            total_boxes_train += len(target['boxes'])
    
    print(f"   Total boxes in training set: {total_boxes_train}")
    print(f"   Average boxes per batch: {total_boxes_train / len(train_loader):.2f}")
    
    print("\n" + "=" * 70)
    print("DataLoader Test Complete!")
    print("=" * 70)
    print("\n✓ DataLoader is working correctly!")
    print("✓ Ready for RF-DETR training!")


if __name__ == '__main__':
    test_dataloader()
