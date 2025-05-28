import argparse
from fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy
import os
import glob
import cv2
import numpy as np
import laspy
import open3d as o3d
import torch.nn.functional as F
import pdb

def mask_to_pointcloud(mask, mask_index, las_path, save_path1,save_path2):
    # 读取 las 点云
    las = laspy.read(las_path)
    x, y, z = las.x, las.y, las.z
    scale = 0.7

    # 与 BEV 图一致的投影方式
    x_norm = ((x - x.min()) / scale).astype(np.int32)
    y_norm = ((y - y.min()) / scale).astype(np.int32)
    img_h = y_norm.max() + 1
    y_norm_flip = img_h - 1 - y_norm

    point_pixels = np.stack([y_norm_flip, x_norm], axis=1)

    # 掩码转 numpy
    mask_np = mask[mask_index].cpu().numpy()
    foreground_pixels = np.argwhere(mask_np == 1)
    foreground_set = set(map(tuple, foreground_pixels))

    # 找出落在 mask 区域内的点
    mask_indices = np.array([i for i, pix in enumerate(point_pixels) if tuple(pix) in foreground_set])

    if len(mask_indices) == 0:
        print(f"[!] 掩码 {mask_index} 没有对应点，跳过。")
        return

    # 保存对应点云
    masked_points = np.vstack((x[mask_indices], y[mask_indices], z[mask_indices])).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(masked_points)
    os.makedirs(os.path.dirname(save_path1), exist_ok=True)
    
    o3d.io.write_point_cloud(save_path1, pcd)


    # 提取屋顶点（最高点下 2 米以内）
    z_max = masked_points[:, 2].max()
    mask_roof = masked_points[:, 2] >= z_max - 5.0
    roof_points = masked_points[mask_roof]
    roof_pcd = o3d.geometry.PointCloud()
    roof_pcd.points = o3d.utility.Vector3dVector(roof_points)
    os.makedirs(os.path.dirname(save_path2), exist_ok=True)
    o3d.io.write_point_cloud(save_path2, roof_pcd)


'''
def recover_pointcloud_from_ann(ann, las_path, output_dir, prefix="", scale=0.7):
    """
    根据 FastSAM 分割掩码恢复点云，并提取屋顶点云。

    参数:
        ann (Tensor or ndarray): [N, H, W] 掩码张量
        las_path (str): 原始点云路径 (.las)
        output_dir (str): 输出目录
        prefix (str): 输出文件名前缀
        scale (float): BEV 缩放系数，需与生成 BEV 时一致
    """
    # 加载点云
    las = laspy.read(las_path)
    x, y, z = las.x, las.y, las.z
    coords = np.vstack((x, y, z)).T

    # 映射到 BEV 坐标系
    x_min, y_min = x.min(), y.min()
    x_norm = ((x - x_min) / scale).astype(np.int32)
    y_norm = ((y - y_min) / scale).astype(np.int32)
    img_h = y_norm.max() + 1
    y_norm_flip = img_h - 1 - y_norm  # 图像Y轴翻转
    point_pixels = np.stack([y_norm_flip, x_norm], axis=1)

    # 准备输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 掩码张量转换为 numpy
    ann_np = ann.cpu().numpy() if torch.is_tensor(ann) else ann
    N = ann_np.shape[0]

    for i in range(N):
        mask = ann_np[i]
        mask_coords = np.argwhere(mask)  # shape: [K, 2]
        if mask_coords.size == 0:
            print(f"[!] 掩码 {i} 无有效像素，跳过。")
            continue

        # 构建像素坐标集合以加速查找
        mask_set = set(map(tuple, mask_coords))

        # 找出掩码中对应的点云索引
        matched_mask = np.array([tuple(pix) in mask_set for pix in point_pixels])
        matched_idx = np.where(matched_mask)[0]

        if matched_idx.size == 0:
            print(f"[!] 掩码 {i} 无对应点，跳过。")
            continue

        # 保存完整掩码对应点云
        masked_points = coords[matched_idx]
        save_path = os.path.join(output_dir, f"{prefix}_mask_{i}.ply")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(masked_points)
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"[✓] 掩码点云保存至: {save_path}")

        # ---------- 提取屋顶点云 ----------
        roof_points = []
        pix_to_idx = {}

        # 建立像素坐标到原始点索引的映射（只在 mask 区域内）
        for idx in matched_idx:
            pix = tuple(point_pixels[idx])
            if pix in mask_set:
                if pix not in pix_to_idx:
                    pix_to_idx[pix] = []
                pix_to_idx[pix].append(idx)

        # 对每个像素，找出 z 最大的点
        for pix, idx_list in pix_to_idx.items():
            z_vals = z[idx_list]
            max_z_idx = idx_list[np.argmax(z_vals)]
            roof_points.append(coords[max_z_idx])

        if roof_points:
            roof_points = np.array(roof_points)
            roof_save_path = os.path.join(output_dir, f"{prefix}_mask_{i}_roof.ply")
            roof_pcd = o3d.geometry.PointCloud()
            roof_pcd.points = o3d.utility.Vector3dVector(roof_points)
            o3d.io.write_point_cloud(roof_save_path, roof_pcd)
            print(f"[✓] 屋顶点云保存至: {roof_save_path}")
        else:
            print(f"[!] 掩码 {i} 无屋顶点，跳过。")
'''

def main(input_dir,output_dir):
    
    bev_dir='./bev_dir'
    # === 加载点云并生成 BEV 图像 ===
    las = laspy.read(input_dir)
    x, y, z = las.x, las.y, las.z
    scale =0.7

    x_norm = ((x - x.min()) / scale).astype(np.int32)
    y_norm = ((y - y.min()) / scale).astype(np.int32)
    img_w, img_h = x_norm.max() + 1, y_norm.max() + 1
    y_norm_flip = img_h - 1 - y_norm  # 图像坐标系中Y轴反向

    bev_img = np.zeros((img_h, img_w), dtype=np.uint8)
    z_norm = ((z - z.min()) / (z.max() - z.min()) * 255).astype(np.uint8)
    for i in range(len(x_norm)):
        xx, yy = x_norm[i], y_norm_flip[i]
        bev_img[yy, xx] = max(bev_img[yy, xx], z_norm[i])

    out_prefix = os.path.splitext(os.path.basename(input_dir))[0]
    os.makedirs(bev_dir, exist_ok=True)
    Image.fromarray(bev_img).save(os.path.join(bev_dir, f"{out_prefix}.png"))
    
    # === step 1 FastSAM 分割bev图像 ===
    model = FastSAM("./weights/FastSAM-x.pt")
    

    img_path = os.path.join(bev_dir, f"{out_prefix}.png")
    input_img = Image.open(img_path).convert("RGB")
    

    
    input = input_img
    
    everything_results = model(
        input,
        device='cpu',
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9    
    )

    bboxes = None
    points = None
    point_label = None
    prompt_process = FastSAMPrompt(input, everything_results, device='cpu')
    
    ann = prompt_process.everything_prompt()
    '''
    #没有建筑物保存原bev图
    if isinstance(ann, list) and len(ann) == 0:
        print(f"{os.path.basename(img_path)}: 没有建筑物")
        # 保存原始 BEV 图像到输出路径
        out_name = os.path.basename(img_path)
        out_path = os.path.join(args.output, out_name)
        input.save(out_path)
        #保存全0 BEV 图用于后续处理
        
        os.makedirs(output_dir, exist_ok=True)  # 若目录不存在则自动创建
        output_path = os.path.join(output_dir, f"{out_prefix}.png")
        bev_img = np.zeros((input.height, input.width), dtype=np.uint8)
        Image.fromarray(bev_img).save(output_path)
    '''
    N, H, W = ann.shape
    total_pixels = H * W
    filtered_masks = []
    
    #去除大面积掩码
    for i in range(N):
        mask = ann[i]
        area = (mask > 0).sum().item()
        ratio = area / total_pixels
        if ratio <= 0.2:
            filtered_masks.append(mask.unsqueeze(0))  # 保持维度

    if filtered_masks:
        ann = torch.cat(filtered_masks, dim=0)
    else:
        ann = torch.empty((0, H, W), device=ann.device)

    
    prompt_process.plot(
        annotations=ann,
        output_path= img_path,
        bboxes=bboxes,
        points=points,
        point_label=point_label,
        withContours=False,
        better_quality=False,
    )
    '''
    # 假设 bev_img 是一个 numpy 灰度图像 (H, W)
    bev_img = np.array(input.convert("L"))  # 确保是 numpy array
    bev_h, bev_w = bev_img.shape

    # Resize ann 到 BEV 尺寸
    ann_resized = F.interpolate(
        ann.unsqueeze(1),             # [N, 1, H_infer, W_infer]
        size=(bev_h, bev_w),          # [H_bev, W_bev]
        mode="nearest"
    ).squeeze(1)                      # [N, H, W]

    # 生成联合掩码
    mask_union = torch.any(ann_resized.bool(), dim=0)  # [H, W] bool tensor
    mask_union_np = mask_union.cpu().numpy()           # [H, W] numpy array

    # 应用掩码
    bev_masked = bev_img.copy()
    bev_masked[~mask_union_np] = 0   # 仅保留掩码区域
    bev_masked[bev_masked<20] = 0
    '''
    if len(ann) > 0:
        for i in range(len(ann)):
            save_path1 = os.path.join("./buildings_pc/buildings", f"{out_prefix}_mask_{i}.ply")
            save_path2 = os.path.join("./buildings_pc/roofs", f"{out_prefix}_mask_{i}.ply")
            mask_to_pointcloud(ann, i,input_dir, save_path1, save_path2)
    else:
        print("No object detected. Skipping point cloud saving.")
    
    '''



    if len(ann) > 0:

            save_path = os.path.join("/data2/wzt_home/FastSAM/buildings1", f"{out_prefix}_mask_{i}.ply")
            recover_pointcloud_from_ann(ann,matching_las_path, save_path)
    else:
        print("No object detected. Skipping point cloud saving.")
    '''


if __name__ == "__main__":
    input_dir='./guangxi_pc/LX0094.las'
    output_dir='./buildings_pc'
    main(input_dir,output_dir)