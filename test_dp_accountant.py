import math
from opacus.accountants.utils import get_noise_multiplier

# 1. 参数设置
epsilon_target = 8.0
N = 70000
delta = 1 / (N * math.log(N))
batch_size = 256
epochs = 1400
q = batch_size / N  # 采样率

# 2. 直接计算
noise_multiplier = get_noise_multiplier(
    target_epsilon=epsilon_target,
    target_delta=delta,
    sample_rate=q,
    epochs=epochs  # 这里直接传入 epochs，内部会自动计算总步数
)

print(f"目标隐私预算：ε={epsilon_target}, δ={delta}")
print(f"所需 noise_multiplier：{noise_multiplier:.4f}")