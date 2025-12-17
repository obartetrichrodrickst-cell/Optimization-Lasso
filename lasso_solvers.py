import numpy as np
import time


class LassoSolver:
    def __init__(self, A, b, lam):
        self.A = A
        self.b = b
        self.lam = lam
        self.n, self.p = A.shape

        # 预计算 Lipschitz 常数 (用于 ISTA/FISTA 步长)
        # L = max(eigenvalue(A^T A))
        self.L = np.linalg.norm(A, 2) ** 2
        self.step_size = 1 / self.L

    def soft_threshold(self, x, theta):
        """软阈值算子 S_theta(x)"""
        return np.sign(x) * np.maximum(np.abs(x) - theta, 0)

    def cost(self, x):
        """计算目标函数值: 1/2||Ax-b||^2 + lambda||x||_1"""
        return 0.5 * np.linalg.norm(self.A @ x - self.b) ** 2 + self.lam * np.linalg.norm(x, 1)

    def ista(self, max_iter=1000, tol=1e-5):
        """ISTA: Iterative Shrinkage-Thresholding Algorithm"""
        x = np.zeros(self.p)
        history = []
        times = []

        start_t = time.time()
        for k in range(max_iter):
            # 1. 梯度下降步
            grad = self.A.T @ (self.A @ x - self.b)
            x_temp = x - self.step_size * grad

            # 2. 邻近算子步 (Soft Thresholding)
            x = self.soft_threshold(x_temp, self.lam * self.step_size)

            # 记录
            history.append(self.cost(x))
            times.append(time.time() - start_t)

            # 简单的收敛判断
            if k > 0 and abs(history[-1] - history[-2]) < tol:
                break

        return x, history, times

    def fista(self, max_iter=1000, tol=1e-5):
        """FISTA: Fast ISTA with Nesterov Momentum"""
        x = np.zeros(self.p)
        y = x.copy()
        t = 1
        history = []
        times = []

        start_t = time.time()
        for k in range(max_iter):
            # 对 y 做梯度下降
            grad = self.A.T @ (self.A @ y - self.b)
            x_new = self.soft_threshold(y - self.step_size * grad, self.lam * self.step_size)

            # 动量更新 (Nesterov acceleration)
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            y = x_new + ((t - 1) / t_new) * (x_new - x)

            # 更新变量
            x = x_new
            t = t_new

            history.append(self.cost(x))
            times.append(time.time() - start_t)

            if k > 0 and abs(history[-1] - history[-2]) < tol:
                break

        return x, history, times

    def admm(self, rho=1.0, max_iter=1000, tol=1e-5):
        """ADMM: Alternating Direction Method of Multipliers"""
        x = np.zeros(self.p)
        z = np.zeros(self.p)
        u = np.zeros(self.p)
        history = []
        times = []

        # --- 预计算 ---
        # x-update 需要解线性方程 (A^T A + rho I)x = ...
        # 为了加速，提前计算这个矩阵的逆（或分解）
        # 注意：对于超高维 p，这里应该使用 Sherman-Morrison 公式优化
        start_t = time.time()
        K = self.A.T @ self.A + rho * np.eye(self.p)
        K_inv = np.linalg.inv(K)  # 预计算逆矩阵
        Atb = self.A.T @ self.b

        for k in range(max_iter):
            # 1. x-update: 最小化 1/2||Ax-b||^2 + (rho/2)||x - z + u||^2
            # 解: x = (A^T A + rho I)^(-1) (A^T b + rho(z - u))
            x = K_inv @ (Atb + rho * (z - u))

            # 2. z-update: 最小化 lambda||z||_1 + (rho/2)||x - z + u||^2
            # 解: SoftThreshold(x + u, lambda/rho)
            z = self.soft_threshold(x + u, self.lam / rho)

            # 3. u-update (对偶变量更新)
            u = u + x - z

            history.append(self.cost(x))
            times.append(time.time() - start_t)

            # ADMM 收敛判断通常看原始残差和对偶残差，这里简化为目标函数
            if k > 0 and abs(history[-1] - history[-2]) < tol:
                break

        return x, history, times