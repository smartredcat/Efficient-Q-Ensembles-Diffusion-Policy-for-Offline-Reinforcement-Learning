import os
import torch
import numpy as np
from torch.distributions import Normal
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from toy_experiments.toy_helpers import Data_Sampler

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--ill", action='store_true')
parser.add_argument("--seed", default=2022, type=int)
parser.add_argument("--exp", default='exp_1', type=str)
parser.add_argument("--x", default=0., type=float)
parser.add_argument("--y", default=0., type=float)
parser.add_argument("--eta", default=2.5, type=float)
parser.add_argument('--device', default=0, type=int)
parser.add_argument("--dir", default='whole_grad', type=str)
parser.add_argument("--r_fun", default='no', type=str)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument("--mode", default='whole_grad', type=str)
args = parser.parse_args()

# 设置随机种子和设备
r_fun_std = 0.25
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
eta = args.eta
seed = args.seed
lr = args.lr
hidden_dim = args.hidden_dim

# 数据生成函数
def generate_data(num, device='cpu'):
    each_num = int(num / 4)
    pos = 0.8
    std = 0.05
    left_up_conor = Normal(torch.tensor([-pos, pos]), torch.tensor([std, std]))
    left_bottom_conor = Normal(torch.tensor([-pos, -pos]), torch.tensor([std, std]))
    right_up_conor = Normal(torch.tensor([pos, pos]), torch.tensor([std, std]))
    right_bottom_conor = Normal(torch.tensor([pos, -pos]), torch.tensor([std, std]))

    left_up_samples = left_up_conor.sample((each_num,)).clip(-1.0, 1.0)
    left_bottom_samples = left_bottom_conor.sample((each_num,)).clip(-1.0, 1.0)
    right_up_samples = right_up_conor.sample((each_num,)).clip(-1.0, 1.0)
    right_bottom_samples = right_bottom_conor.sample((each_num,)).clip(-1.0, 1.0)

    data = torch.cat([left_up_samples, left_bottom_samples, right_up_samples, right_bottom_samples], dim=0)

    action = data
    state = torch.zeros_like(action)

    r_left_up = 3.0 + 0.5 * torch.randn((each_num, 1))
    r_left_bottom = 0.5 * torch.randn((each_num, 1))
    r_right_up = 1.5 + 0.5 * torch.randn((each_num, 1))
    r_right_bottom = 5.0 + 0.5 * torch.randn((each_num, 1))
    reward = torch.cat([r_left_up, r_left_bottom, r_right_up, r_right_bottom], dim=0)

    return Data_Sampler(state, action, reward, device)

# 设置随机种子
torch.manual_seed(seed)
np.random.seed(seed)

# 数据生成
num_data = int(10000)
data_sampler = generate_data(num_data, device)

# 环境参数
state_dim = 2
action_dim = 2
max_action = 1.0

discount = 0.99
tau = 0.005
model_type = 'MLP'

T = 50
beta_schedule = 'vp'
num_epochs = 1000
batch_size = 100
iterations = int(num_data / batch_size)

# 创建图像保存目录
img_dir = f'toy_imgs/{args.dir}'
os.makedirs(img_dir, exist_ok=True)

# 绘制图像
fig, axs = plt.subplots(1, 5, figsize=(5.5 * 5, 5))
axis_lim = 1.1

# 第一个图：散点图
pos = 0.8
std = 0.05
left_up_conor = Normal(torch.tensor([-pos, pos]), torch.tensor([std, std])).sample((200,)).clip(-1.0, 1.0).numpy()
left_bottom_conor = Normal(torch.tensor([-pos, -pos]), torch.tensor([std, std])).sample((200,)).clip(-1.0, 1.0).numpy()
right_up_conor = Normal(torch.tensor([pos, pos]), torch.tensor([std, std])).sample((200,)).clip(-1.0, 1.0).numpy()
right_bottom_conor = Normal(torch.tensor([pos, -pos]), torch.tensor([std, std])).sample((200,)).clip(-1.0, 1.0).numpy()

axs[0].scatter(left_up_conor[:, 0], left_up_conor[:, 1], label=r"$r \sim N (3.0, 0.5)$", color='red')
axs[0].scatter(left_bottom_conor[:, 0], left_bottom_conor[:, 1], label=r"$r \sim N (0.0, 0.5)$", color='blue')
axs[0].scatter(right_up_conor[:, 0], right_up_conor[:, 1], label=r"$r \sim N (1.5, 0.5)$", color='green')
axs[0].scatter(right_bottom_conor[:, 0], right_bottom_conor[:, 1], label=r"$r \sim N (5.0, 0.5)$", color='purple')
axs[0].set_xlim(-axis_lim, axis_lim)
axs[0].set_ylim(-axis_lim, axis_lim)
axs[0].set_xlabel('x', fontsize=20)
axs[0].set_ylabel('y', fontsize=20)
axs[0].set_title('Add Reward', fontsize=25)
axs[0].legend(loc='best', fontsize=15, title_fontsize=15)

# 第二个图：核密度估计曲线
sns.kdeplot(x=left_up_conor[:, 0], y=left_up_conor[:, 1], ax=axs[1], cmap="Reds", fill=True)
sns.kdeplot(x=left_bottom_conor[:, 0], y=left_bottom_conor[:, 1], ax=axs[1], cmap="Blues", fill=True)
sns.kdeplot(x=right_up_conor[:, 0], y=right_up_conor[:, 1], ax=axs[1], cmap="Greens", fill=True)
sns.kdeplot(x=right_bottom_conor[:, 0], y=right_bottom_conor[:, 1], ax=axs[1], cmap="Purples", fill=True)

# 手动添加图例
axs[1].plot([], [], color='red', label=r"$r \sim N (3.0, 0.5)$")
axs[1].plot([], [], color='blue', label=r"$r \sim N (0.0, 0.5)$")
axs[1].plot([], [], color='green', label=r"$r \sim N (1.5, 0.5)$")
axs[1].plot([], [], color='purple', label=r"$r \sim N (5.0, 0.5)$")
axs[1].legend(loc='best', fontsize=15, title_fontsize=15)

axs[1].set_xlim(-axis_lim, axis_lim)
axs[1].set_ylim(-axis_lim, axis_lim)
axs[1].set_xlabel('x', fontsize=20)
axs[1].set_ylabel('y', fontsize=20)
axs[1].set_title('True Distribution', fontsize=25)

# 初始化扩散模型
from toy_experiments.ql_diffusion import QL_Diffusion

agent = QL_Diffusion(state_dim=state_dim,
                     action_dim=action_dim,
                     max_action=max_action,
                     device=device,
                     discount=discount,
                     tau=tau,
                     eta=eta,
                     beta_schedule=beta_schedule,
                     n_timesteps=T,
                     model_type=model_type,
                     hidden_dim=hidden_dim,
                     lr=lr,
                     r_fun=None,
                     mode=args.mode)

# 训练扩散模型
for i in range(1, num_epochs + 1):
    b_loss, q_loss = agent.train(data_sampler, iterations=iterations, batch_size=batch_size)
    if i % 100 == 0:
        print(f'QL-Diffusion Epoch: {i} B_loss {b_loss} Q_loss {q_loss}')

# 生成扩散模型的样本
num_eval = 1000
new_state = torch.zeros((num_eval, 2), device=device)
new_action = agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()

# 绘制扩散模型生成的核密度估计曲线
sns.kdeplot(x=new_action[:, 0], y=new_action[:, 1], ax=axs[2], cmap="Oranges", fill=True, label='Diffusion-QL')

axs[2].set_xlim(-axis_lim, axis_lim)
axs[2].set_ylim(-axis_lim, axis_lim)
axs[2].set_xlabel('x', fontsize=20)
axs[2].set_ylabel('y', fontsize=20)
axs[2].set_title('Diffusion-QL', fontsize=25)

# 手动添加图例
axs[2].plot([], [], color='orange', label='Diffusion-QL')
axs[2].legend(loc='best', fontsize=15, title_fontsize=15)

# 保存图像
file_name = f'ql_all_T{T}_eta{eta}_r_fun{args.r_fun}_lr{lr}_hd{hidden_dim}_mode_{args.mode}_sd{args.seed}.pdf'
fig.tight_layout()
fig.savefig(os.path.join(img_dir, file_name))
plt.show()