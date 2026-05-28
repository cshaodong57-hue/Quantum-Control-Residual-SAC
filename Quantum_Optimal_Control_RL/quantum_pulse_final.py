"""
终极算力解放版：中性原子 CZ 门 DRL 控制框架 (大宗师残差底色 + 时间戳防覆盖)
包含：纯正物理叠加态考试、极其平滑的几何平均判卷、2.0能量极限
终极进化：残差强化学习(Residual RL) + 12控制点超高分辨率 + [256,256,256]深层脑容量
新增功能：无缝集成“对数奖励消融实验 (Ablation Study)”开关
"""

import os
import numpy as np
import pandas as pd
import qutip as qt
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

warnings.filterwarnings('ignore')

# ==========================================
# 1. 物理参数与硬件限制定义
# ==========================================
PI = np.pi
V_BLOCKADE = 50.0 * 2 * PI
OMEGA_MAX = 5.0 * 2 * PI
T_GATE = 1.2
N_STEPS = 60
DT = T_GATE / N_STEPS

GAMMA_1 = 0.05 * 2 * PI
GAMMA_2 = 0.10 * 2 * PI

P0, P1, Pr = qt.basis(3, 0), qt.basis(3, 1), qt.basis(3, 2)
sigma_1r = P1 * Pr.dag()
I3 = qt.qeye(3)

Pr_A = qt.tensor(Pr * Pr.dag(), I3)
Pr_B = qt.tensor(I3, Pr * Pr.dag())
sigma_1r_A = qt.tensor(sigma_1r, I3)
sigma_1r_B = qt.tensor(I3, sigma_1r)

c_ops = [
    np.sqrt(GAMMA_1) * sigma_1r_A, np.sqrt(GAMMA_1) * sigma_1r_B,
    np.sqrt(GAMMA_2) * Pr_A, np.sqrt(GAMMA_2) * Pr_B
]

# ==========================================
# 🌟 物理防作弊：相位锁定叠加态考题
# ==========================================
state_00 = qt.tensor(P0, P0)
state_01 = qt.tensor(P0, P1)
state_10 = qt.tensor(P1, P0)
state_11 = qt.tensor(P1, P1)

test_01 = (state_00 + state_01).unit()
test_10 = (state_00 + state_10).unit()
test_11 = (state_00 + state_11).unit()
psi_plus = (P0 + P1).unit()
state_plus_plus = qt.tensor(psi_plus, psi_plus)

target_01 = (state_00 - state_01).unit()
target_10 = (state_00 - state_10).unit()
target_11 = (state_00 - state_11).unit()
ideal_entangled = 0.5 * (qt.tensor(P0, P0) - qt.tensor(P0, P1) - qt.tensor(P1, P0) - qt.tensor(P1, P1))

test_states = [state_00, test_01, test_10, test_11, state_plus_plus]
target_states = [state_00, target_01, target_10, target_11, ideal_entangled]
target_projs = [target * target.dag() for target in target_states]


# ==========================================
# 2. 硬件滤波器与演化引擎
# ==========================================
def hardware_lowpass_filter(signal_array, cutoff_freq_mhz=100.0):
    nyquist = (1.0 / DT) / 2.0
    normal_cutoff = cutoff_freq_mhz / nyquist
    if normal_cutoff >= 1.0: return signal_array
    b, a = butter(2, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal_array)


def simulate_open_system(omega_seq, phase_seq, noise_detuning, use_dissipation=True):
    H0 = V_BLOCKADE * qt.tensor(Pr * Pr.dag(), Pr * Pr.dag()) + noise_detuning * (Pr_A + Pr_B)
    omega_complex = (omega_seq / 2.0) * np.exp(1j * phase_seq)

    H = [H0,
         [sigma_1r_A, omega_complex], [sigma_1r_A.dag(), np.conj(omega_complex)],
         [sigma_1r_B, omega_complex], [sigma_1r_B.dag(), np.conj(omega_complex)]]

    active_c_ops = c_ops if use_dissipation else []
    tlist = np.linspace(0, T_GATE, N_STEPS)
    fids = []

    for idx, initial_state in enumerate(test_states):
        rho_0 = initial_state * initial_state.dag()
        result = qt.mesolve(H, rho_0, tlist, active_c_ops)
        rho_t = result.states[-1]

        fid = np.real(qt.expect(target_projs[idx], rho_t))
        fids.append(fid)

    fids_array = np.clip(fids, 1e-10, 1.0)
    fid_prod = np.prod(fids_array)
    return float(fid_prod ** (1.0 / len(fids_array)))


# ==========================================
# 3. 高级强化学习环境 (✨ 融合消融实验分支)
# ==========================================
class AdvancedQuantumEnv(gym.Env):
    def __init__(self, reward_mode="log"):
        super().__init__()
        self.num_control_points = 12
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_control_points * 2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.curriculum_progress = 0.0
        self.best_fid = 0.0

        # 记录当前的奖励模式："log" (完整版) 或 "linear" (消融版)
        self.reward_mode = reward_mode

        # 💾 核心黑科技：加载大宗师的脉冲轨迹作为物理基准
        baseline_file = "Master_Pulse_Baseline.csv"

        if os.path.exists(baseline_file):
            print(f"🍁 成功捕获大宗师秘籍：{baseline_file}，启动残差学习模式！")
            df_baseline = pd.read_csv(baseline_file)
            # 提取并还原角频率
            self.master_omega = df_baseline['Amplitude(MHz)'].values * (2 * PI)
            self.master_phase = df_baseline['Phase(rad)'].values
        else:
            print(f"⚠️ 未找到大宗师秘籍({baseline_file})，退回从零探索模式！")
            self.master_omega = None
            self.master_phase = None

    def step(self, action):
        t_action = np.linspace(0, 1, self.num_control_points)
        t_sim = np.linspace(0, 1, N_STEPS)

        amp_action = action[:self.num_control_points].copy()
        phase_action = action[self.num_control_points:].copy()

        amp_action[0] = amp_action[-1] = 0.0
        phase_action[0] = phase_action[-1] = 0.0

        if self.master_omega is not None:
            # 🌟 残差模式：新模型的输出被限制为 10% 的微调力度
            amp_mod = CubicSpline(t_action, amp_action)(t_sim) * 0.1
            phase_mod = CubicSpline(t_action, phase_action)(t_sim) * (0.1 * PI)

            omega_raw = self.master_omega * (1.0 + amp_mod)
            phase_raw = self.master_phase + phase_mod
        else:
            # 旧版抛物线基准模式
            A = 4.0 * PI / T_GATE
            baseline_omega = A * (np.sin(PI * t_sim) ** 2)
            amp_mod = CubicSpline(t_action, amp_action)(t_sim) * 2.0
            omega_raw = baseline_omega * (1.0 + amp_mod)
            phase_raw = CubicSpline(t_action, phase_action)(t_sim) * PI

        omega_seq = np.clip(hardware_lowpass_filter(omega_raw), 0, OMEGA_MAX)
        phase_seq = np.clip(hardware_lowpass_filter(phase_raw), -PI, PI)

        noise_level = 0.15 * 2 * PI * self.curriculum_progress
        detuning_noise = np.random.normal(0, noise_level)
        use_dissipation = self.curriculum_progress > 0.5

        fidelity = simulate_open_system(omega_seq, phase_seq, detuning_noise, use_dissipation)

        # ==========================================
        # 🎯 奖励计算 (包含对数/线性分支与平滑度惩罚)
        # ==========================================
        amplitude_diff = np.sum(np.diff(amp_action) ** 2)
        phase_diff = np.sum(np.diff(phase_action) ** 2)

        if self.reward_mode == "log":
            # 完整版：对数惩罚奖励，提供指数级梯度
            base_reward = -np.log10(1.0001 - fidelity)
        elif self.reward_mode == "linear":
            # 消融版：普通线性奖励，放大 10 倍以匹配 SAC 默认网络梯度尺度
            base_reward = fidelity * 10.0
        else:
            raise ValueError("Unknown reward_mode.")

        penalty = 0.01 * (amplitude_diff + phase_diff)  # 抖动惩罚
        reward = base_reward - penalty

        if fidelity > self.best_fid: self.best_fid = fidelity

        return np.array([fidelity], dtype=np.float32), reward, True, False, {'fidelity': fidelity, 'omega': omega_seq,
                                                                             'phase': phase_seq}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([0.0], dtype=np.float32), {}


# ==========================================
# 4. 课程学习动态调度器
# ==========================================
class CurriculumCallback(BaseCallback):
    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.locals['total_timesteps']
        for env_idx in range(self.training_env.num_envs):
            self.training_env.env_method("set_curriculum", progress, indices=env_idx)
        self.logger.record("curriculum/progress", progress)
        return True


def set_curriculum_wrapper(env, progress):
    env.curriculum_progress = progress


AdvancedQuantumEnv.set_curriculum = set_curriculum_wrapper

# ==========================================
# 5. 主程序与结果导出
# ==========================================
if __name__ == "__main__":
    # ==========================================
    # 🎛️ 实验模式切换开关 (极其重要)
    # False -> 运行你的终极完整版大宗师模型 (30万步，对数奖励)
    # True  -> 运行对数奖励的消融实验 (5万步，线性奖励)
    # ==========================================
    RUN_ABLATION_STUDY = True

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs("./models/", exist_ok=True)
    tensorboard_log_dir = "./quantum_tensorboard/"

    # 动态设定训练参数
    if RUN_ABLATION_STUDY:
        print(f"=== ⚠️ 启动消融实验：移除对数奖励 (w/o Log-Reward) (代号: Ablation_{run_id}) ===")
        TOTAL_STEPS = 50000
        current_reward_mode = "linear"
        log_name_prefix = f"SAC_Linear_Ablation_{run_id}"
        save_prefix = f'ckpt_ablation_{run_id}'
        final_model_name = f"SAC_Ablation_Linear_{run_id}"
    else:
        print(f"=== 🚀 启动终局算力引擎：完整大宗师模型 (代号: {run_id}) ===")
        TOTAL_STEPS = 300000
        current_reward_mode = "log"
        log_name_prefix = f"SAC_Master_{run_id}"
        save_prefix = f'ckpt_master_{run_id}'
        final_model_name = f"SAC_Master_{run_id}"

    # 实例化带参数的环境
    env = DummyVecEnv([lambda: AdvancedQuantumEnv(reward_mode=current_reward_mode) for _ in range(4)])

    # 🎲 强行更换随机种子，彻底摆脱非凸陷阱厄运
    model = SAC("MlpPolicy", env, verbose=0, learning_rate=3e-4, batch_size=256, ent_coef='auto',
                policy_kwargs=dict(net_arch=[256, 256, 256]),
                tensorboard_log=tensorboard_log_dir, seed=42)

    # 定期存档防丢失
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // env.num_envs, 1),
        save_path='./models/',
        name_prefix=save_prefix
    )

    all_callbacks = CallbackList([CurriculumCallback(), checkpoint_callback])

    print(f"🔥 开始 {TOTAL_STEPS} 步强化学习特训 (模式: {current_reward_mode})...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=all_callbacks, progress_bar=True, tb_log_name=log_name_prefix)

    # 终极封印
    model.save(final_model_name)
    print(f"🎉 训练完成，模型已成功封印至本地: {final_model_name}.zip")

    print("\n=== 开始提取环境真实耗散下的最高保真度物理脉冲 ===")
    test_env = AdvancedQuantumEnv(reward_mode=current_reward_mode)
    test_env.curriculum_progress = 1.0  # 测试时全额开启耗散和噪声
    obs, _ = test_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = test_env.step(action)

    print(f"🎯 最终策略单次验证保真度: {info['fidelity']:.4f}")

    df = pd.DataFrame({
        'Time(us)': np.linspace(0, T_GATE, N_STEPS),
        'Amplitude(MHz)': info['omega'] / (2 * PI),
        'Phase(rad)': info['phase']
    })

    csv_name = f"AWG_Pulse_{'Ablation' if RUN_ABLATION_STUDY else 'Master'}_{run_id}.csv"
    df.to_csv(csv_name, index=False)
    print(f"✅ 物理脉冲数据已导出: {csv_name}")

    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    plt.plot(df['Time(us)'], df['Amplitude(MHz)'], color='blue', linewidth=2.5, label='AWG Amplitude')
    plt.ylabel('Amplitude (MHz)')
    title_str = 'Ultimate Gate (Linear Ablation)' if RUN_ABLATION_STUDY else 'Ultimate Gate (Residual Version)'
    plt.title(f'{title_str} [{run_id}]')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.plot(df['Time(us)'], df['Phase(rad)'], color='red', linewidth=2.5, label='Phase Control')
    plt.xlabel('Time (us)')
    plt.ylabel('Phase (rad)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    pic_name = f"Ultimate_Pulse_{'Ablation' if RUN_ABLATION_STUDY else 'Master'}_{run_id}.png"
    plt.savefig(pic_name, dpi=300)
    print(f"✅ 脉冲图像已保存: {pic_name}")

    plt.show()