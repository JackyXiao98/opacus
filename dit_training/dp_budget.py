
import math
# from opacus.accountants.utils import get_noise_multiplier

# 1. 参数设置
epsilon_target = 8.0
N = 30000
delta = 1 / (N * math.log(N))
batch_size = 56
epochs = 400
q = batch_size / N  # 采样率
print(f"迭代数：{N/batch_size*epochs}")

# # 2. 直接计算
# noise_multiplier = get_noise_multiplier(
#     target_epsilon=epsilon_target,
#     target_delta=delta,
#     sample_rate=q,
#     epochs=epochs  # 这里直接传入 epochs，内部会自动计算总步数
# )

# print(f"目标隐私预算：ε={epsilon_target}, δ={delta}")
# print(f"所需 noise_multiplier：{noise_multiplier:.4f}")


# import numpy as np
# import math
# from opacus.accountants import RDPAccountant

# # ---------------------- 1. 移除无效导入！----------------------
# # 删掉这行：from opacus.utils.uniform_sampler import sample_rate

# # ---------------------- 2. 确定已知参数（不变）----------------------
# epsilon_target = 8.0  # 目标 ε
# N = 70000             # 训练集总样本数
# delta = 1/(N * math.log(N)) # 手动指定的 δ（按 1/N 原则）
# max_grad_norm = 1.0   # 梯度裁剪阈值
# batch_size = 256      # 批次大小
# epochs = 1400          # 训练轮数
# num_training_steps = epochs * (N // batch_size)  # 总训练步数
# poisson_sampling = True  # 你的采样方式（保持不变）

# # ---------------------- 3. 手动计算采样率 q（关键修改！）----------------------
# # 采样率 q = 单批次样本数 / 总样本数（Poisson/均匀采样通用，无需依赖 Opacus）
# q = batch_size / N  # 直接手动计算，替代原来的 sample_rate 函数

# # ---------------------- 4. 后续逻辑完全不变！----------------------
# def compute_epsilon(noise_multiplier):
#     """给定 noise_multiplier，计算对应的 ε（基于 RDP 会计）"""
#     accountant = RDPAccountant()
#     # alphas = list(range(2, 101))  # RDP 阶数范围
#     accountant.step(
#         noise_multiplier=noise_multiplier,
#         sample_rate=q,
#     )
#     return accountant.get_epsilon(delta=delta)

# def find_noise_multiplier(epsilon_target, delta, max_iter=100, tol=1e-3):
#     """二分法查找最优 noise_multiplier（逻辑不变）"""
#     low = 0.1
#     high = 10.0

#     # 扩大 high 范围（确保覆盖满足条件的 noise_multiplier）
#     while compute_epsilon(high) > epsilon_target:
#         high *= 2

#     # 二分查找
#     for _ in range(max_iter):
#         mid = (low + high) / 2
#         current_epsilon = compute_epsilon(mid)
#         print(low, high, current_epsilon)

#         if current_epsilon < epsilon_target:
#             high = mid
#         else:
#             low = mid

#         if high - low < tol:
#             break

#     return high

# # 执行反推
# optimal_noise_multiplier = find_noise_multiplier(epsilon_target, delta)
# print(f"目标隐私预算：ε={epsilon_target}, δ={delta}")
# print(f"所需 noise_multiplier：{optimal_noise_multiplier:.4f}")
# print(f"验证：实际 ε={compute_epsilon(optimal_noise_multiplier):.4f}")




# import scipy
# import numpy as np
# import math

# def delta_Gaussian(eps, mu):
#    """Compute delta of Gaussian mechanism with shift mu or equivalently noise scale 1/mu"""
#    if mu==0:
#        return 0
#    return scipy.stats.norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * scipy.stats.norm.cdf(-eps / mu - mu / 2)

 
# def eps_Gaussian(delta, mu):
#    """Compute eps of Gaussian mechanism with shift mu or equivalently noise scale 1/mu"""
#    def f(x):
#        return delta_Gaussian(x, mu) - delta
#    return scipy.optimize.root_scalar(f, bracket=[0, 500], method='brentq').root


# def compute_epsilon(noise_multiplier, num_steps, delta):
#    return eps_Gaussian(delta, np.sqrt(num_steps) / noise_multiplier)

# N=70000 # ffhq
# delta= 1/(N*math.log(N))
# print("delta", delta)
# epoch=1400

# break_noise=0
# for eps in [2,4,8]:
#     for noise in np.arange(100,0, -0.01):
#         compute_epsilon(noise, epoch, delta)
#         if compute_epsilon(noise, epoch, delta)>eps:
#             break_noise=noise
#             break
#     print("threshold eps", eps, "break_noise", break_noise, f"eps {compute_epsilon(noise, epoch, delta):4f}")