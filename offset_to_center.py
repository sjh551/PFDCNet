import numpy as np
import torch

def get_cam_center(cam):
    """
    计算给定热力图的显著区域中心位置。

    参数:
    cam (torch.Tensor): 热力图张量，形状为 (B, 1, H, W)

    返回:
    centers (torch.Tensor): 每个图像的显著区域中心坐标 (B, 2)，其中每行是 (center_x, center_y)
    """
    B, _, H, W = cam.shape  # 获取批量大小和热力图的尺寸

    # 创建一个坐标网格，指定 indexing="ij" 避免警告
    x_coords = torch.arange(W, device=cam.device)  # 在同一设备上创建坐标
    y_coords = torch.arange(H, device=cam.device)
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")  # 添加 indexing 参数
    
    # 初始化用于存储每个图像中心的张量
    centers = torch.zeros((B, 2), device=cam.device)  # (x, y)
    
    for b in range(B):
        if torch.any(cam[b, 0, :, :]>1):
            print(f'limitless numbers!')
        # 获取当前图像的热力图 (1, H, W)，并去除负值
        cam_image = torch.maximum(cam[b, 0, :, :], torch.tensor(0., device=cam.device))
        
        # 归一化热力图
        cam_image = cam_image / (1e-7 + torch.max(cam_image))
        
        # 计算加权坐标
        weighted_x = torch.sum(cam_image * x_grid)
        weighted_y = torch.sum(cam_image * y_grid)
        
        # 计算显著区域的中心坐标
        center_x = weighted_x / torch.sum(cam_image)

        center_y = weighted_y / torch.sum(cam_image)
        if torch.isnan(center_x):
            print(f'sum_x:{torch.sum(cam_image)}')
            print(f'cam:{cam}')
        # 存储当前图像的中心坐标
        centers[b] = torch.tensor([center_x, center_y], device=cam.device)
    
    return centers


def compute_loss(sample_coords,center_coords):
    """
    暂且针对1x1可变形卷积核
    offsets:(B,2,H,W)--->(B,18,H,W)
    centers:(B,2)
    计算L2损失,衡量每个采样坐标与中心坐标的偏移。
    参数:
    sample_coords (torch.Tensor): 形状为 (B, 2, H, W) 的采样坐标
    center_coords (torch.Tensor): 形状为 (B, 2) 的中心坐标
    返回:
    torch.Tensor: 损失值
    """
    _,C,H,W = sample_coords.shape
    num_pixles = int(C*H*W)
    len_coords = C//2
    # 将中心坐标扩展到与采样坐标相同的形状
    center_coords_expanded = center_coords.unsqueeze(2).unsqueeze(3)  # 形状变为 (B, 2, 1, 1)
    center_coords_expanded = center_coords_expanded.expand(-1, -1, H, W)  # 形状变为 (B, 2, H, W)

    # 提取 x 坐标和 y 坐标通道
    sample_coords_x = sample_coords[:, :len_coords, :, :]  # 取前 kernel_size 个通道 (x 坐标)
    sample_coords_y = sample_coords[:, len_coords:, :, :]  # 取后 kernel_size 个通道 (y 坐标)

    # 计算每个采样坐标与中心坐标的差值
    # 分别计算 x 和 y 坐标的差值
    diff_x = sample_coords_x - center_coords_expanded[:, 0:1, :, :]  # x 坐标的差值
    diff_y = sample_coords_y - center_coords_expanded[:, 1:2, :, :]  # y 坐标的差值

    # 计算 L2 损失 (欧几里得距离)，计算 x 和 y 坐标差值的平方和再开方
    loss = torch.sqrt(diff_x**2 + diff_y**2)  # 形状为 (B, 9, H, W)，计算每个像素点的 L2 距离
    
    # 对每个图像的所有像素点求和，得到每个样本的总损失
    loss = loss.sum(dim=(1,2,3))/(num_pixles)  # 形状为 (B,)
    # 对所有批次取平均
    total_loss = loss.mean()  # 标量，表示整个批次的平均损失

    return total_loss




if __name__=="__main__":
    # 假设 cam 是大小为 (2, 1, 28, 28) 的热力图
    # cam = torch.rand(2, 1, 28, 28)
    sample_coords = torch.zeros(4,2,28,28)
    sample_coords[:,:,1,1]=1
    cam = torch.zeros((1, 1, 28, 28))
    cam[:,:,0,0]=1
    # cam[:,:,27,27]=0.5
    # 计算每个图像的显著区域中心
    centers = get_cam_center(cam)
    loss = compute_loss(sample_coords,centers)
    print(loss)
    # 输出中心位置
    print(f"每个图像的显著区域中心位置:\n{centers}")



