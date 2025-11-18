#!/usr/bin/env python3
"""
RF-DETR 多GPU训练脚本（优化版）
支持4张RTX 2080 Ti (11GB each)

请使用 torchrun 启动以进行高效的分布式训练:
torchrun --nproc_per_node=4 train_rfdetr_multigpu.py
"""

import os
import sys
import json
import shutil
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import datetime

# 設置 PyTorch CUDA 記憶體優化（在導入其他模組前設置）
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 添加RF-DETR路径
rfdetr_path = Path("rf-detr")
if rfdetr_path.exists():
    sys.path.insert(0, str(rfdetr_path.absolute()))
else:
    # 嘗試相對於腳本位置查找
    script_dir = Path(__file__).parent
    rfdetr_path = script_dir / "rf-detr"
    if rfdetr_path.exists():
        sys.path.insert(0, str(rfdetr_path.absolute()))

try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
    from rfdetr.config import TrainConfig
    # 只在主進程打印成功信息
    rank = int(os.environ.get('RANK', '0'))
    if rank == 0:
        # 檢查是否使用 torchrun 啟動
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            world_size = os.environ.get('WORLD_SIZE')
            print(f"✓ RF-DETR已安装 (使用 torchrun，{world_size} 個 GPU)")
        else:
            print("✓ RF-DETR已安装 (單GPU模式)")
except ImportError as e:
    print(f"❌ 找不到RF-DETR: {e}")
    print(f"當前sys.path: {sys.path[:3]}")
    print(f"嘗試的路徑: {rfdetr_path.absolute() if rfdetr_path else 'N/A'}")
    sys.exit(1)


def convert_to_roboflow_format(dataset_path, output_path):
    """
    转换COCO格式数据集到Roboflow格式
    同时修正category_id（从1改为0）
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    print(f"\n转换数据集格式到: {output_path}")
    
    # 创建Roboflow格式目录
    (output_path / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "valid").mkdir(parents=True, exist_ok=True)
    (output_path / "test").mkdir(parents=True, exist_ok=True)
    
    # 复制训练集（图片直接放在train/目录下，不是train/images/）
    print("  复制训练集...")
    for img_file in (dataset_path / "train").glob("*.jpg"):
        shutil.copy2(img_file, output_path / "train" / img_file.name)
    
    # 修正训练集annotations的category_id（从1改为0，因为RF-DETR期望从0开始）
    train_ann = json.load(open(dataset_path / "annotations" / "instances_train.json"))
    train_ann['categories'][0]['id'] = 0
    for ann in train_ann['annotations']:
        ann['category_id'] = 0
    with open(output_path / "train" / "_annotations.coco.json", 'w') as f:
        json.dump(train_ann, f)
    
    # 复制验证集（图片直接放在valid/目录下）
    print("  复制验证集...")
    for img_file in (dataset_path / "val").glob("*.jpg"):
        shutil.copy2(img_file, output_path / "valid" / img_file.name)
    
    # 修正验证集annotations的category_id
    val_ann = json.load(open(dataset_path / "annotations" / "instances_val.json"))
    val_ann['categories'][0]['id'] = 0
    for ann in val_ann['annotations']:
        ann['category_id'] = 0
    with open(output_path / "valid" / "_annotations.coco.json", 'w') as f:
        json.dump(val_ann, f)
    
    # 创建测试集（使用验证集作为测试集用于训练评估）
    print("  使用验证集作为测试集（用于训练评估）...")
    for img_file in (dataset_path / "val").glob("*.jpg"):
        shutil.copy2(img_file, output_path / "test" / img_file.name)
    with open(output_path / "test" / "_annotations.coco.json", 'w') as f:
        json.dump(val_ann, f)
    
    print(f"✓ 数据集格式转换完成: {output_path}")
    return str(output_path.absolute())


def create_training_config(
    dataset_dir,
    output_dir,
    epochs=50,
    batch_size=16,  # 这是总 batch size, 会被均分到各个GPU
    lr=1e-4,
    resolution=512,  # Small模型使用512x512
    model_size="small",  # 默認使用Small模型
    grad_accum_steps=1,
    device="cuda:0"  # **修改点**: 接受特定设备作为参数
):
    """
    创建多GPU训练配置
    """
    config = TrainConfig(
        # 数据集配置
        dataset_file="roboflow",
        dataset_dir=dataset_dir,
        
        # 训练参数（多GPU优化）
        epochs=epochs,
        batch_size=batch_size,  # 总batch size
        lr=lr,
        lr_encoder=lr * 1.5,  # encoder學習率稍高
        weight_decay=1e-4,
        grad_accum_steps=grad_accum_steps,
        
        # 模型配置
        # resolution 和 pretrain_weights 由模型自己處理，不在這裡設置
        
        # 输出配置
        output_dir=output_dir,
        checkpoint_interval=10,
        
        # EMA 配置（多GPU訓練中可能有問題，暫時禁用）
        use_ema=False,  # 禁用 EMA 避免分布式問題
        
        # 多GPU配置
        num_workers=8,  # 更多worker
        
        # 學習率調度
        warmup_epochs=5,
        lr_drop=40,  # 在第40個epoch降低學習率
        
        # 其他
        class_names=["pig"],
        run_test=True,
        tensorboard=True,
        multi_scale=True,  # 多尺度訓練
        early_stopping=False,  # 禁用早停，避免checkpoint問題
    )
    
    return config


def train_rfdetr_multigpu(
    dataset_root="rfdetr_dataset",
    model_size="small",  # 默認使用Small模型
    epochs=50,
    batch_size=16,
    lr=1e-4,
    resolution=512,  # Small模型使用512x512
    use_pretrained=True,
    save_dir="runs/train",
    num_gpus=4
):
    """
    多GPU训练RF-DETR模型
    """
    
    # **修改点**: 檢查分布式環境，但不初始化（RF-DETR會自己初始化）
    is_distributed = False
    rank = 0
    local_rank = 0
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        is_distributed = True
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        device = f"cuda:{local_rank}"
        
        # 只在主进程 (rank 0) 打印信息，避免日志混乱
        if rank != 0:
            import builtins
            builtins.print = lambda *args, **kwargs: None
        
        print(f"✅ 檢測到分布式訓練環境。當前進程 Rank: {rank}, 使用 GPU: {device}")
        print(f"   (注意: 分布式初始化將由 RF-DETR 內部處理)")
    else:
        # 如果没有用 torchrun 启动，则回退到单设备模式
        print("⚠ 未檢測到分布式環境，將在單個可用GPU上運行。")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print("RF-DETR 猪隻检测模型训练（多GPU版本）")
    print("=" * 70)
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    # **修改点**: 移除手动设置 CUDA_VISIBLE_DEVICES，由 torchrun 管理
    if is_distributed:
        print(f"\n總共使用 {os.environ['WORLD_SIZE']} 個 GPU 進行訓練。")
    
    if rank == 0:  # 只在主進程打印
        print(f"\n當前進程可用GPU信息:")
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU: {props.name}")
        print(f"    内存: {props.total_memory / (1024**3):.2f} GB")
    
    # 创建输出目录（带时间戳）- 只在主进程创建
    output_dir = Path(".")
    if not is_distributed or rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f"rfdetr_{model_size}_pig_detection_multigpu_{timestamp}"
        output_dir = Path(save_dir) / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保所有进程都获得了输出目录的路径
    if is_distributed:
        # 需要等RF-DETR初始化dist後才能使用broadcast
        # 這裡先用簡單的方式：從環境變量傳遞
        if rank == 0:
            os.environ['OUTPUT_DIR'] = str(output_dir)
        # 使用小延遲確保環境變量設置完成
        import time
        time.sleep(1)
        if rank != 0:
            output_dir = Path(os.environ.get('OUTPUT_DIR', '.'))

    if rank == 0:
        print(f"\n训练输出目录: {output_dir}")
    
    # 转换数据集格式
    dataset_path = Path(dataset_root)
    roboflow_dataset_path = Path("rfdetr_dataset_roboflow")
    
    if not roboflow_dataset_path.exists():
        if not is_distributed or rank == 0:
            dataset_dir = convert_to_roboflow_format(dataset_path, roboflow_dataset_path)
            # 設置環境變量供其他進程使用
            os.environ['DATASET_DIR'] = dataset_dir
    else:
        if rank == 0:
            print(f"\n✓ 使用现有的Roboflow格式数据集: {roboflow_dataset_path}")
    
    # 等待數據集轉換完成
    if is_distributed:
        import time
        time.sleep(2)
        
    dataset_dir = os.environ.get('DATASET_DIR', str(roboflow_dataset_path.absolute()))
    if not dataset_dir:
        dataset_dir = str(roboflow_dataset_path.absolute())

    # 读取数据集信息
    train_ann_file = roboflow_dataset_path / "train" / "_annotations.coco.json"
    with open(train_ann_file) as f:
        train_data = json.load(f)
    
    if rank == 0:
        print(f"\n数据集信息:")
        print(f"  训练图片: {len(train_data['images'])}")
        print(f"  训练标注: {len(train_data['annotations'])}")
        print(f"  类别数: {len(train_data['categories'])}")
        print(f"  类别名称: {[c['name'] for c in train_data['categories']]}")
    
    # 计算实际batch size
    effective_gpus = int(os.environ.get('WORLD_SIZE', '1')) if is_distributed else 1
    actual_batch_per_gpu = batch_size // effective_gpus
    
    if rank == 0:
        print(f"\nBatch配置:")
        print(f"  GPU数量: {effective_gpus}")
        print(f"  总Batch Size: {batch_size}")
        print(f"  每GPU Batch Size: {actual_batch_per_gpu}")
    
    # 初始化模型
    model_class_map = {
        "nano": (RFDETRNano, "RF-DETR-NANO"),
        "small": (RFDETRSmall, "RF-DETR-SMALL"),
        "medium": (RFDETRMedium, "RF-DETR-MEDIUM")
    }
    
    model_class, model_name = model_class_map.get(model_size, (RFDETRNano, "RF-DETR-NANO"))
    
    if rank == 0:
        print(f"\n初始化模型: {model_name}")
    model = model_class()
    if rank == 0:
        print(f"  分辨率: {resolution}x{resolution}")
        print(f"  预训练: {'是' if use_pretrained else '否'}")
    
    # 创建训练配置
    train_config = create_training_config(
        dataset_dir=dataset_dir,
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        resolution=resolution,
        model_size=model_size,
        grad_accum_steps=1,
        device=device # **修改点**: 传入当前进程的GPU设备
    )
    
    # 保存配置 (仅主进程)
    if not is_distributed or rank == 0:
        config_dict = {
            "model_size": model_size,
            "dataset_root": dataset_root,
            "epochs": epochs,
            "batch_size": batch_size,
            "batch_per_gpu": actual_batch_per_gpu,
            "num_gpus": effective_gpus,
            "lr": lr,
            "resolution": resolution,
            "use_pretrained": use_pretrained,
            "save_dir": save_dir,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        config_file = output_dir / "train_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"  配置已保存: {config_file}")
    
    # 开始训练
    if rank == 0:
        print("\n" + "=" * 70)
        print("开始训练...")
        print("=" * 70)
    
    try:
        model.train_from_config(train_config)
        if rank == 0:
            print("\n✓ 训练完成！")
        
        # 训练完成后在测试集上运行推理 (仅主进程)
        if not is_distributed or rank == 0:
            print("\n" + "=" * 70)
            print("在测试集上运行推理...")
            print("=" * 70)
            
            run_inference_on_test_set(model, output_dir, model_size)
            
    except Exception as e:
        if rank == 0:
            print(f"\n❌ 训练出错: {e}")
            import traceback
            traceback.print_exc()
    finally:
        # RF-DETR會自己清理分布式環境，我們不需要手動清理
        pass

    if not is_distributed or rank == 0:
        print("\n" + "=" * 70)
        print("训练和推理完成！")
        print("=" * 70)
        print(f"\n输出目录: {output_dir}")
        print(f"  - best_checkpoint.pth: 最佳模型")
        print(f"  - predictions.json: 测试集预测结果")
        print(f"  - submission.txt: 提交格式结果")


def run_inference_on_test_set(model, output_dir, model_size="small"):
    """
    在真实测试集上运行推理（无标注文件）
    """
    test_dir = Path("test/img")
    if not test_dir.exists():
        print(f"❌ 测试集目录不存在: {test_dir}")
        return
    
    test_images = sorted(test_dir.glob("*.jpg"))
    print(f"找到 {len(test_images)} 张测试图片")
    
    # 加载最佳模型
    best_checkpoint = output_dir / "best_checkpoint.pth"
    if not best_checkpoint.exists():
        print("⚠ 未找到best_checkpoint.pth，使用当前模型")
    else:
        print(f"加载最佳模型: {best_checkpoint}")
        # 根據模型大小選擇正確的類
        model_class_map = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "medium": RFDETRMedium
        }
        model_class = model_class_map.get(model_size, RFDETRSmall)
        try:
            # 嘗試使用模型類別載入
            model = model_class(pretrain_weights=str(best_checkpoint))
        except:
            print("⚠ 無法載入checkpoint，使用當前模型")
    
    # 运行推理
    all_predictions = []
    submission_lines = []
    
    print("开始推理...")
    for img_path in test_images:
        # 预测
        results = model.predict(str(img_path), conf_threshold=0.3)
        
        # 保存预测结果
        predictions = {
            "image_id": img_path.name,
            "detections": []
        }
        
        submission_line = img_path.name
        
        if hasattr(results, 'boxes') and len(results.boxes) > 0:
            for box, score in zip(results.boxes, results.scores):
                x1, y1, x2, y2 = box
                predictions["detections"].append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(score),
                    "category": "pig"
                })
                submission_line += f" {x1} {y1} {x2} {y2} {score}"
        
        all_predictions.append(predictions)
        submission_lines.append(submission_line)
    
    # 保存结果
    predictions_file = output_dir / "predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    print(f"✓ 预测结果已保存: {predictions_file}")
    
    submission_file = output_dir / "submission.txt"
    with open(submission_file, 'w') as f:
        f.write('\n'.join(submission_lines))
    print(f"✓ 提交文件已保存: {submission_file}")


if __name__ == "__main__":
    # 只在主進程打印配置
    rank = int(os.environ.get('RANK', '0'))
    
    if rank == 0:
        print("\n训练配置:")
    
    config = {
        "model_size": "small",        # nano, small, medium - 使用Small模型
        "dataset_root": "rfdetr_dataset",
        "epochs": 50,
        "batch_size": 4,              # 減小 batch size：8 (每張GPU 2張圖)
        "lr": 1e-4,
        "resolution": 512,            # Small模型使用512x512
        "use_pretrained": True,
        "save_dir": "runs/train",
        "num_gpus": 4                 # 这个参数现在仅供参考，实际GPU数由torchrun命令决定
    }
    
    if rank == 0:
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("\n💡 記憶體和分布式訓練優化:")
        print(f"   - Small 模型使用 512x512 解析度")
        print(f"   - Batch size = 4 (每張GPU 1張圖)")
        print(f"   - 禁用 EMA 避免分布式 checkpoint 問題")
        print(f"   - 使用 4 GPU 進行分布式訓練")
    
    train_rfdetr_multigpu(**config)
