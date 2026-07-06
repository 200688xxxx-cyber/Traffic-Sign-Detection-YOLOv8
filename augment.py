import cv2
import shutil
import random
import numpy as np
from pathlib import Path

# ==================================
# 项目路径
# ==================================

# ⚠️ 注意：这里我帮你改回了你实际的桌面路径
ROOT = Path(r"C:/Users/Lenovo/Desktop/1/2/yolov8/ultralytics/cfg/datasets")

INPUT_IMG_DIR = ROOT / "output" / "images" / "train"
INPUT_LABEL_DIR = ROOT / "output" / "labels" / "train"

OUTPUT_IMG_DIR = ROOT / "output_aug" / "images" / "train"
OUTPUT_LABEL_DIR = ROOT / "output_aug" / "labels" / "train"

# ==================================
# 参数配置
# ==================================

# 需要拯救的弱势类别 ID
WEAK_CLASS_IDS = [6,10,11,12,13,14,15,19,20]

# 包含弱势目标的图片，每张生成几张增强图
WEAK_AUG_PER_IMAGE = 5

# 当某个类别的增强数量达到这个值后，针对该类别的特供增强就会停止。
MAX_AUG_PER_WEAK_CLASS = 200

# 普通图片生成几张（0表示不增强）
NORMAL_AUG_PER_IMAGE = 1

# 全量扫描限制
MAX_IMAGES = 999999

# ==================================
# 创建目录
# ==================================

OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)


# ==================================
# 增强函数
# ==================================

def random_brightness(img):
    factor = random.uniform(0.6, 1.4)
    result = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return result


def gaussian_blur(img):
    kernel = random.choice([3, 5])
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def add_noise(img):
    noise = np.random.normal(0, 10, img.shape)
    result = img.astype(np.float32) + noise
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def augment_image(img):
    aug = img.copy()
    if random.random() < 0.7:
        aug = random_brightness(aug)
    if random.random() < 0.5:
        aug = gaussian_blur(aug)
    if random.random() < 0.5:
        aug = add_noise(aug)
    return aug


# ==================================
# 获取图片并初始化计数器
# ==================================

image_files = (
        list(INPUT_IMG_DIR.glob("*.jpg")) +
        list(INPUT_IMG_DIR.glob("*.png"))
)

image_files = image_files[:MAX_IMAGES]

# 🌟 初始化类别追踪器：记录每个弱势类别已经生成了多少张
weak_class_counter = {cls_id: 0 for cls_id in WEAK_CLASS_IDS}

print("=" * 50)
print("开始扫描并进行增强")
print(f"目标弱势 ID: {WEAK_CLASS_IDS}")
print(f"单类最高生成上限: {MAX_AUG_PER_WEAK_CLASS} 张")
print("=" * 50)

success_count = 0
skip_count = 0
weak_found_count = 0

# ==================================
# 开始增强
# ==================================

for idx, img_path in enumerate(image_files, start=1):

    label_path = INPUT_LABEL_DIR / f"{img_path.stem}.txt"

    if not label_path.exists():
        skip_count += 1
        continue

    # --- 读取标签并判断 ---
    is_weak_target = False
    valid_classes_in_this_img = set()  # 记录这张图里包含哪些尚未满额的弱势类

    try:
        # 加入 encoding='utf-8' 防止 Windows 默认 GBK 读取报错
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if not line.strip():
                    continue
                class_id = int(line.split()[0])

                # 如果是弱势目标
                if class_id in WEAK_CLASS_IDS:
                    # 🌟 核心拦截逻辑：检查这个类别是否已经“满额”了
                    if weak_class_counter[class_id] < MAX_AUG_PER_WEAK_CLASS:
                        is_weak_target = True
                        valid_classes_in_this_img.add(class_id)

    except Exception as e:
        print(f"❌ 读取标签失败 {label_path.name}: {e}")
        skip_count += 1
        continue

    # 确定增强次数
    if is_weak_target:
        aug_times = WEAK_AUG_PER_IMAGE
        weak_found_count += 1
        # 🌟 记账：这几个未满额的弱势类，即将各自获得 aug_times 张新图片
        for cid in valid_classes_in_this_img:
            weak_class_counter[cid] += aug_times
    else:
        aug_times = NORMAL_AUG_PER_IMAGE

    if aug_times == 0:
        continue

    # --- 执行图片增强与保存 ---
    img = cv2.imread(str(img_path))
    if img is None:
        skip_count += 1
        continue

    for i in range(aug_times):
        aug_img = augment_image(img)
        new_img_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
        new_label_name = f"{img_path.stem}_aug_{i}.txt"

        save_img_path = OUTPUT_IMG_DIR / new_img_name
        save_label_path = OUTPUT_LABEL_DIR / new_label_name

        cv2.imwrite(str(save_img_path), aug_img)
        shutil.copy(label_path, save_label_path)

        success_count += 1
        # 实时进度播报
        if success_count % 50 == 0 :print(f"✅ 生成 {new_img_name} (累计: {success_count}张)")

# ==================================
# 输出结果与明细
# ==================================

print("\n")
print("=" * 50)
print("🎯 数据增强完成！")
print(f"🔍 扫描图片总数: {len(image_files)} 张")
print(f"💡 触发增强的弱势图片: {weak_found_count} 张")
print("-" * 50)
print("📊 弱势类别增强明细 (Class ID : 新增数量):")
# 打印每个类别的最终生成情况
for cid, count in weak_class_counter.items():
    print(f"   类别 [{cid}] : 新增 {count} 张")
print("-" * 50)
print(f"✅ 总计生成新图片: {success_count} 张")
print("=" * 50)