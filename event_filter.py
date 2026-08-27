import cv2
import numpy as np
import pandas as pd
import os

# ===================== 【基础路径配置】请直接使用 =====================
base_root = "./data"

target_folders = [0,1]  # 如需处理多个文件夹，可改为 [1, 2, 3]

stream_dir = "output_stream"
seg_dir = "output_seg"
filter_dir = "filter_event"
# ====================================================================

def process_single_folder(folder_name):
    """
    处理单个文件夹（如 1）：
    - 读取 output_stream/1 下的所有 CSV 事件流
    - 对每个 CSV，匹配 output_seg1/1 下同名的 PNG 掩码
    - 过滤后保存到 output_filter_event/1，文件名与原 CSV 保持一致
    """
    # 拼接路径
    stream_folder = os.path.join(base_root, stream_dir, str(folder_name))
    seg_folder = os.path.join(base_root, seg_dir, str(folder_name))
    out_folder = os.path.join(base_root, filter_dir, str(folder_name))

    # 创建输出目录
    os.makedirs(out_folder, exist_ok=True)

    # 获取该文件夹下所有事件流
    event_files = [f for f in os.listdir(stream_folder) if f.endswith(".csv")]
    if not event_files:
        print(f"⚠️  {stream_folder} 下无 CSV 文件，跳过")
        return

    print(f"\n========================================")
    print(f"📂 正在处理文件夹：{folder_name}")
    print(f"📄 事件流数量：{len(event_files)}")

    # 遍历每个事件流，匹配同名掩码
    for idx, csv_filename in enumerate(event_files, 1):
        print(f"\n--- 处理 [{idx}/{len(event_files)}] {csv_filename} ---")

        # 构造对应的掩码文件名（替换后缀）
        base_name = os.path.splitext(csv_filename)[0]
        mask_filename = f"{base_name}.png"
        mask_path = os.path.join(seg_folder, mask_filename)

        # 检查掩码是否存在
        if not os.path.exists(mask_path):
            print(f"⚠️  未找到对应的掩码 {mask_filename}，跳过该文件")
            continue

        # 1. 读取并处理掩码
        vascular_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if vascular_mask is None:
            print(f"❌ 掩码读取失败：{mask_path}，跳过")
            continue

        _, binary_mask = cv2.threshold(vascular_mask, 127, 255, cv2.THRESH_BINARY)
        mask_h, mask_w = binary_mask.shape
        print(f"✅ 掩码尺寸：{mask_w}x{mask_h} | 血管像素：{np.sum(binary_mask == 255)}")

        # 2. 读取事件流
        event_path = os.path.join(stream_folder, csv_filename)
        events = pd.read_csv(event_path)
        print(f"✅ 原始事件数：{len(events)}")

        # 3. 过滤超出图像范围的坐标
        valid_x = (events['x'] >= 0) & (events['x'] < mask_w)
        valid_y = (events['y'] >= 0) & (events['y'] < mask_h)
        events = events[valid_x & valid_y].copy()

        if len(events) == 0:
            print(f"⚠️  有效坐标为空，跳过该文件")
            continue

        # 4. 掩码过滤（只保留血管区域事件）
        events['mask_val'] = binary_mask[events['y'].astype(int), events['x'].astype(int)]
        filtered = events[events['mask_val'] == 255].drop(columns=['mask_val'])

        # 5. 保存过滤后的事件流（文件名与原 CSV 一致）
        out_path = os.path.join(out_folder, csv_filename)
        filtered.to_csv(out_path, index=False)

        ratio = len(filtered) / len(events)
        print(f"✅ 过滤完成 | 保留：{len(filtered)} 个事件 | 占比：{ratio:.2%}")
        print(f"✅ 已保存到：{out_path}")

# ===================== 批量执行 =====================
if __name__ == "__main__":
    for folder in target_folders:
        try:
            process_single_folder(folder)
        except Exception as e:
            print(f"❌ 处理文件夹 {folder} 失败：{str(e)}")

    print("\n🎉 所有文件夹和事件流处理完毕！")