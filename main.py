import numpy as np
import matplotlib.pyplot as plt
from lasso_solvers import LassoSolver


# --- 数据生成工具 ---
def generate_data(n, p, sparsity=0.1, noise=0.1):
    np.random.seed(42)
    A = np.random.randn(n, p)
    # 标准化列，有利于算法收敛
    A = A / np.linalg.norm(A, axis=0)

    true_x = np.random.randn(p)
    # 随机置零，制造稀疏性
    mask = np.random.rand(p) > sparsity
    true_x[mask] = 0

    b = A @ true_x + noise * np.random.randn(n)
    return A, b, true_x


# --- 实验配置 ---
# 场景 1: n > p (良态，瘦高矩阵)
# 场景 2: n < p (欠定，高维胖矩阵，Lasso的主场)
# 场景 3: 大规模 (测试计算极限)
scenarios = [
    {"name": "Tall Matrix (n=500, p=200)", "n": 500, "p": 200, "lam": 0.1},
    {"name": "Fat Matrix (n=100, p=500)", "n": 100, "p": 500, "lam": 0.1},
    {"name": "Large Scale (n=1000, p=1000)", "n": 1000, "p": 1000, "lam": 0.5}
]

# --- 运行实验 ---
for scene in scenarios:
    print(f"Running scenario: {scene['name']}...")
    A, b, true_x = generate_data(scene['n'], scene['p'])
    solver = LassoSolver(A, b, lam=scene['lam'])

    # 运行三种算法
    # 注意：ADMM的rho参数可以调，这里设为1.0
    x_ista, h_ista, t_ista = solver.ista(max_iter=500)
    x_fista, h_fista, t_fista = solver.fista(max_iter=500)
    x_admm, h_admm, t_admm = solver.admm(rho=1.5, max_iter=500)

    # 计算最优值 f* (取三者中最小的最后那个值为近似 f*)
    f_star = min(h_ista[-1], h_fista[-1], h_admm[-1])

    # --- 绘图 ---
    plt.figure(figsize=(14, 5))

    # 图 1: Iteration vs Error (Log Scale)
    plt.subplot(1, 2, 1)
    plt.semilogy([h - f_star for h in h_ista], label='ISTA', linewidth=2)
    plt.semilogy([h - f_star for h in h_fista], label='FISTA', linewidth=2)
    plt.semilogy([h - f_star for h in h_admm], label='ADMM', linewidth=2)
    plt.title(f"Convergence by Iteration\n{scene['name']}")
    plt.xlabel('Iterations')
    plt.ylabel('log(f(x_k) - f*)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 图 2: Time vs Error (Log Scale)
    plt.subplot(1, 2, 2)
    plt.semilogy(t_ista, [h - f_star for h in h_ista], label='ISTA', linewidth=2)
    plt.semilogy(t_fista, [h - f_star for h in h_fista], label='FISTA', linewidth=2)
    plt.semilogy(t_admm, [h - f_star for h in h_admm], label='ADMM', linewidth=2)
    plt.title(f"Convergence by Time\n{scene['name']}")
    plt.xlabel('Time (seconds)')
    plt.ylabel('log(f(x_k) - f*)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"result_{scene['n']}_{scene['p']}.png")  # 保存图片用于报告
    plt.show()

print("All experiments done!")