import numpy as np
from scipy.optimize import least_squares
from scipy.special import comb

import numpy as np
from scipy.optimize import least_squares
from scipy.special import comb

class BezierFitter:
    def __init__(self, input_dim=14, num_control_points=5, horizon_length=32):
        """
        贝塞尔曲线拟合器
        :param input_dim: 输入数据维度 (默认14)
        :param num_control_points: 控制点数量 (至少2个，默认5)
        :param horizon_length: 输入数据长度 (至少2个点，默认32)
        """
        # 参数验证
        if num_control_points < 2:
            raise ValueError("至少需要2个控制点")
        if horizon_length < 2:
            raise ValueError("输入数据长度至少需要2个点")
        
        self.input_dim = input_dim
        self.num_control_points = num_control_points
        self.horizon_length = horizon_length
        self._n = num_control_points - 1  # 贝塞尔曲线阶数

    @staticmethod
    def bernstein_poly(n, i, t):
        """计算伯恩斯坦基函数"""
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    def bezier_curve(self, control_points, t_values):
        """生成贝塞尔曲线"""
        curve_points = np.zeros((self.input_dim, len(t_values)))
        
        for idx, t in enumerate(t_values):
            for k in range(self.num_control_points):
                basis = self.bernstein_poly(self._n, k, t)
                curve_points[:, idx] += control_points[:, k] * basis
        return curve_points

    def fit(self, input_points):
        """执行拟合"""
        # 输入验证
        if input_points.shape != (self.input_dim, self.horizon_length):
            raise ValueError(
                f"输入数据形状应为({self.input_dim}, {self.horizon_length}), "
                f"实际收到 {input_points.shape}"
            )

        # 生成参数空间
        t_space = np.linspace(0, 1, self.horizon_length)

        # 智能初始化控制点（等间距采样）
        init_controls = np.zeros((self.input_dim, self.num_control_points))
        sample_indices = np.linspace(0, self.horizon_length-1, 
                                   self.num_control_points, dtype=int)
        for i, idx in enumerate(sample_indices):
            init_controls[:, i] = input_points[:, idx]

        # 优化过程
        def error_func(x):
            controls = x.reshape(self.input_dim, self.num_control_points)
            generated = self.bezier_curve(controls, t_space)
            return (generated - input_points).flatten()

        result = least_squares(
            error_func,
            init_controls.flatten(),
            method='lm',
            max_nfev=2000  # 增加迭代次数提升精度
        )

        return result.x.reshape(self.input_dim, self.num_control_points)

    def predict(self, control_points, resolution=100):
        """使用控制点生成高分辨率曲线"""
        t_fine = np.linspace(0, 1, resolution)
        return self.bezier_curve(control_points, t_fine)