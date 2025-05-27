import torch
# import numpy as np # 如果 t_values 使用 torch.linspace 并且 comb 返回合适的类型，则不是严格需要
from scipy.special import comb

def bezier_upsample(ctrl_pts_long, ctrl_pts_mid, ctrl_pts_short, pred_horizon=32):

    num_ctrl_pts_long = ctrl_pts_long.shape[1]
    num_ctrl_pts_mid = ctrl_pts_mid.shape[1]
    num_ctrl_pts_short = ctrl_pts_short.shape[1]

    # 从输入张量中获取批量大小 B、动作维度 act_dim、设备 device 和数据类型 dtype
    B, _, act_dim = ctrl_pts_long.shape
    device = ctrl_pts_long.device
    dtype = ctrl_pts_long.dtype

    # 嵌套的 bernstein_poly 函数, 已针对 torch 的兼容性进行修改
    def bernstein_poly(n, i, t):
        """计算伯恩斯坦基函数"""
        # comb(n, i) 返回一个 numpy float。将其转换为 torch 张量以进行一致的操作。
        # exact=False 确保返回浮点数，即使组合结果是整数
        coefficient = torch.tensor(comb(n, i, exact=False), device=t.device, dtype=t.dtype)
        # t 期望是一个 torch 张量
        return coefficient * (t ** i) * ((1 - t) ** (n - i))

    # 用于对单个尺度进行上采样的辅助函数
    def _upsample_single_scale(ctrl_pts, target_len):
        # ctrl_pts: (B, 当前尺度的控制点数量, act_dim)
        # target_len: 上采样后轨迹中期望的点数
        
        current_num_ctrl_pts = ctrl_pts.shape[1]
        
        # 如果没有控制点，或者目标长度为0，则返回相应形状的零张量
        if current_num_ctrl_pts == 0:
            return torch.zeros((B, target_len, act_dim), device=device, dtype=dtype)
        if target_len == 0:
             return torch.zeros((B, 0, act_dim), device=device, dtype=dtype)

        # 贝塞尔曲线的阶数 (m)，对应公式 P(t) = sum_{i=0 to m} B_{i,m}(t) * P_i
        # 如果控制点数量为 N, 则阶数 m = N-1
        m = current_num_ctrl_pts - 1

        # 生成用于对贝塞尔曲线进行采样的 t_values
        # t_values 形状: (target_len,)
        t_values = torch.linspace(0, 1, steps=target_len, device=device, dtype=dtype)

        # 计算所有 i 和所有 t 的伯恩斯坦基函数
        # bernstein_basis 将具有形状 (target_len, current_num_ctrl_pts)
        # 每一列 `j` 对应所有 t_values 的 B_{j,m}(t)
        # 每一行 `k` 对应所有控制点 `i` 的 B_{i,m}(t_k)
        
        bernstein_basis_list = []
        for i in range(current_num_ctrl_pts): # i 从 0 到 m
            # bernstein_poly(n, i, t) 其中 n 是阶数 m
            bern_poly_i_for_all_t = bernstein_poly(m, i, t_values) # 形状: (target_len,)
            bernstein_basis_list.append(bern_poly_i_for_all_t)
        
        # 如果 current_num_ctrl_pts > 0, bernstein_basis_list 不会为空
        bernstein_basis = torch.stack(bernstein_basis_list, dim=1) # 形状: (target_len, current_num_ctrl_pts)

        # 使用爱因斯坦求和约定执行贝塞尔曲线计算
        # ctrl_pts 形状: (B, current_num_ctrl_pts, act_dim)
        # bernstein_basis 形状: (target_len, current_num_ctrl_pts)
        #期望输出 upsampled_trajectory 形状: (B, target_len, act_dim)
        # 公式: C(t_k) = sum_{i=0 to m} ctrl_pts_i * bernstein_poly(m, i, t_k)
        # 'b' 代表批量, 'n' 代表 num_ctrl_pts, 'a' 代表 act_dim, 't' 代表 target_len
        upsampled_trajectory = torch.einsum('bna,tn->bta', ctrl_pts, bernstein_basis)
        
        return upsampled_trajectory

    # 1. 将每个尺度的控制点上采样到其目标长度
    # 长尺度
    target_len_long = pred_horizon
    upsampled_long = _upsample_single_scale(ctrl_pts_long, target_len_long)
    # upsampled_long 形状: (B, pred_horizon, act_dim)

    # 中尺度
    target_len_mid = pred_horizon // 2
    upsampled_mid = _upsample_single_scale(ctrl_pts_mid, target_len_mid)
    # upsampled_mid 形状: (B, pred_horizon // 2, act_dim)

    # 短尺度
    target_len_short = pred_horizon // 4
    upsampled_short = _upsample_single_scale(ctrl_pts_short, target_len_short)
    # upsampled_short 形状: (B, pred_horizon // 4, act_dim)

    # 2. 提取用于平均的片段
    # "取三个结果[:, :pred_horizion//4, :] 求平均"
    segment_len = pred_horizon // 4
    
    # 如果 segment_len 为 0 (例如 pred_horizon < 4),
    # 切片操作将正确处理并返回一个时间维度为0的张量。
    # 最后的平均操作也将产生一个时间维度为0的张量。

    avg_long_segment = upsampled_long[:, :segment_len, :]
    avg_mid_segment = upsampled_mid[:, :segment_len, :]
    # avg_short_segment 实际上是整个 upsampled_short，因为它的长度就是 segment_len
    avg_short_segment = upsampled_short[:, :segment_len, :] 

    # 3. 对提取的片段求平均
    final_output = (avg_long_segment * 0.8 + avg_mid_segment + avg_short_segment * 1.2) / 3.0
    
    return final_output


