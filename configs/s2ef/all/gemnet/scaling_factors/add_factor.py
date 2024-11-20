"""
增加新的缩放因子
"""
import torch

# 步骤 1: 加载现有的模型状态字典
checkpoint_path =  "/home/liud/Documents/code/ocp/configs/s2ef/all/gemnet/scaling_factors/gemnet-oc.pt"  # 替换为你的 .pt 文件路径
model_state_dict = torch.load(checkpoint_path)

# 打印当前缩放因子
print("当前缩放因子:")
for name, param in model_state_dict.items():
    print(f"{name}: {param}")

# 步骤 2: 添加新的缩放因子
model_state_dict['int_blocks.0.atom_interaction_input_graph.scale_rbf_sum'] = torch.nn.Parameter(torch.tensor(0.5019, dtype=torch.float32))
model_state_dict['int_blocks.1.atom_interaction_input_graph.scale_rbf_sum'] = torch.nn.Parameter(torch.tensor(0.4642, dtype=torch.float32))
model_state_dict['int_blocks.2.atom_interaction_input_graph.scale_rbf_sum'] = torch.nn.Parameter(torch.tensor(0.4351, dtype=torch.float32))
model_state_dict['int_blocks.3.atom_interaction_input_graph.scale_rbf_sum'] = torch.nn.Parameter(torch.tensor(0.4076, dtype=torch.float32))

# 禁用梯度追踪
for key in model_state_dict:
    if "scale_rbf_sum" in key:
        model_state_dict[key].requires_grad = False

# 打印更新后的缩放因子
print("更新后的缩放因子:")
for name, param in model_state_dict.items():
    print(f"{name}: {param}")

# 步骤 3: 保存更新后的状态字典
new_checkpoint_path = '/home/liud/Documents/code/ocp/configs/s2ef/all/gemnet/scaling_factors/gemnet-oc_1.pt'  # 替换为新的 .pt 文件路径
torch.save(model_state_dict, new_checkpoint_path)

print(f"已将更新后的缩放因子保存到 {new_checkpoint_path}")