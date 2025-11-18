"""
RF-DETR 猪隻检测模型配置示例
基于COCO数据集格式的配置文件
"""

# 数据集配置
dataset_config = {
    'type': 'CocoDetection',
    'root': 'rfdetr_dataset',
    'train': {
        'img_folder': 'rfdetr_dataset/train',
        'ann_file': 'rfdetr_dataset/annotations/instances_train.json'
    },
    'val': {
        'img_folder': 'rfdetr_dataset/val', 
        'ann_file': 'rfdetr_dataset/annotations/instances_val.json'
    },
    'test': {
        'img_folder': 'test/img',  # 测试集图片目录
        'ann_file': None  # 测试集无标注
    }
}

# 模型配置
model_config = {
    'type': 'RFDETR',
    'backbone': 'resnet50',  # 可选: resnet50, resnet101, swin_tiny, swin_small
    'num_classes': 1,  # 只有一个类别: pig
    'num_queries': 300,  # DETR查询数量，可根据图片中目标数量调整
    'aux_loss': True,  # 使用辅助损失
    'with_box_refine': True,  # 使用边界框细化
    'two_stage': False,  # 是否使用两阶段
}

# 训练配置
train_config = {
    'batch_size': 2,  # 根据GPU内存调整 (DETR需要较大内存)
    'num_workers': 4,
    'epochs': 50,
    'lr': 1e-4,  # 学习率
    'weight_decay': 1e-4,
    'lr_backbone': 1e-5,  # backbone学习率 (通常比主网络小)
    'lr_drop': 40,  # 学习率衰减epoch
    'clip_max_norm': 0.1,  # 梯度裁剪
    
    # 优化器
    'optimizer': 'AdamW',
    
    # 学习率调度器
    'lr_scheduler': 'step',
    
    # 数据增强
    'augmentation': {
        'resize': [640, 360],  # 保持原图比例
        'random_horizontal_flip': 0.5,
        'random_select': True,
        'random_resize': [[480, 270], [512, 288], [544, 306], [576, 324], [608, 342], [640, 360]],
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    },
    
    # 损失权重
    'loss_weights': {
        'loss_ce': 2.0,  # 分类损失权重
        'loss_bbox': 5.0,  # L1边界框损失权重  
        'loss_giou': 2.0,  # GIoU损失权重
    },
    
    # Matcher配置
    'matcher': {
        'cost_class': 2.0,  # 分类代价
        'cost_bbox': 5.0,  # L1代价
        'cost_giou': 2.0,  # GIoU代价
    }
}

# 评估配置
eval_config = {
    'batch_size': 1,
    'metrics': ['mAP', 'AP50', 'AP75'],  # COCO评估指标
    'iou_thresholds': [0.5, 0.75],  # IoU阈值
}

# 推理配置
inference_config = {
    'confidence_threshold': 0.5,  # 置信度阈值
    'nms_threshold': 0.5,  # NMS阈值（DETR通常不需要NMS）
    'max_detections': 100,  # 最大检测数量
}

# 保存配置
save_config = {
    'checkpoint_dir': 'runs/train/rfdetr_pig_detection',
    'save_interval': 5,  # 每5个epoch保存一次
    'keep_last_n': 3,  # 保留最后3个checkpoint
}

# 日志配置  
log_config = {
    'log_dir': 'runs/train/rfdetr_pig_detection/logs',
    'log_interval': 50,  # 每50个iteration打印一次
    'tensorboard': True,
    'wandb': False,  # 可选: 使用Weights & Biases
}

# 完整配置
config = {
    'dataset': dataset_config,
    'model': model_config,
    'train': train_config,
    'eval': eval_config,
    'inference': inference_config,
    'save': save_config,
    'log': log_config,
}

if __name__ == '__main__':
    import json
    
    # 保存为JSON配置文件
    with open('rfdetr_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("Configuration saved to: rfdetr_config.json")
    print("\nKey configurations:")
    print(f"  - Model: {model_config['type']} with {model_config['backbone']}")
    print(f"  - Classes: {model_config['num_classes']} (pig)")
    print(f"  - Training images: 1012")
    print(f"  - Validation images: 254")
    print(f"  - Batch size: {train_config['batch_size']}")
    print(f"  - Epochs: {train_config['epochs']}")
    print(f"  - Learning rate: {train_config['lr']}")
