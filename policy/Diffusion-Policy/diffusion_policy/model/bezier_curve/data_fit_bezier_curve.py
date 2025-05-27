import numpy as np
from scipy.optimize import least_squares
from scipy.special import comb


class BezierFitter:
    def __init__(self, input_dim=14, num_control_points=5, horizon_length=64):
        """
        贝塞尔曲线拟合器
        :param input_dim: 输入数据维度 (默认14)
        :param num_control_points: 控制点数量 (至少2个，默认5)
        :param horizon_length: 输入数据长度 (至少2个点，默认64)
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
        # control_points形状: (input_dim, num_control_points) 或 (batch_size, input_dim, num_control_points)
        # 返回形状: (input_dim, len(t_values)) 或 (batch_size, input_dim, len(t_values))
        
        if control_points.ndim == 2:
            # 单个样本情况
            curve_points = np.zeros((self.input_dim, len(t_values)))
            for idx, t in enumerate(t_values):
                for k in range(self.num_control_points):
                    basis = self.bernstein_poly(self._n, k, t)
                    curve_points[:, idx] += control_points[:, k] * basis
            return curve_points
        else:
            # 批量处理情况
            batch_size = control_points.shape[0]
            curve_points = np.zeros((batch_size, self.input_dim, len(t_values)))
            for idx, t in enumerate(t_values):
                for k in range(self.num_control_points):
                    basis = self.bernstein_poly(self._n, k, t)
                    curve_points[:, :, idx] += control_points[:, :, k] * basis
            return curve_points

    def fit(self, input_points):
        """执行拟合
        :param input_points: 输入数据，形状为 (input_dim, horizon_length) 或 (batch_size, input_dim, horizon_length)
        :return: 控制点，形状为 (input_dim, num_control_points) 或 (batch_size, input_dim, num_control_points)
        """
        # 检查输入维度
        if input_points.ndim == 2:
            # 单个样本情况
            if input_points.shape != (self.input_dim, self.horizon_length):
                raise ValueError(
                    f"单个样本输入数据形状应为({self.input_dim}, {self.horizon_length}), "
                    f"实际收到 {input_points.shape}"
                )
            return self._fit_single(input_points)
        elif input_points.ndim == 3:
            # 批量处理情况
            batch_size = input_points.shape[0]
            if input_points.shape[1:] != (self.input_dim, self.horizon_length):
                raise ValueError(
                    f"批量输入数据形状应为(batch_size, {self.input_dim}, {self.horizon_length}), "
                    f"实际收到 {input_points.shape}"
                )
            
            # 为每个样本拟合控制点
            all_control_points = []
            for i in range(batch_size):
                control_points = self._fit_single(input_points[i])
                all_control_points.append(control_points)
            
            return np.stack(all_control_points, axis=0)
        else:
            raise ValueError("输入数据必须是2维或3维数组")

    def _fit_single(self, input_points):
        """拟合单个样本"""
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
        """使用控制点生成高分辨率曲线
        :param control_points: 控制点，形状为 (input_dim, num_control_points) 或 (batch_size, input_dim, num_control_points)
        :param resolution: 输出曲线的分辨率
        :return: 生成的曲线，形状为 (input_dim, resolution) 或 (batch_size, input_dim, resolution)
        """
        t_fine = np.linspace(0, 1, resolution)
        return self.bezier_curve(control_points, t_fine)