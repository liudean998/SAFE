"""
增加新的缩放因子
"""
import torch

# 步骤 1: 加载现有的模型状态字典
checkpoint_path =  "/home/liud/Documents/code/ocp/configs/s2ef/all/gemnet/scaling_factors/gemnet-oc-is2rv.pt"  # 替换为你的 .pt 文件路径
model_state_dict = torch.load(checkpoint_path)

# 打印当前缩放因子
# print("当前缩放因子:")
# for name, param in model_state_dict.items():
#     print(f"{name}: {param}")

# # 步骤 2: 添加新的缩放因子
model_state_dict['int_blocks.3.trip_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(8.6465, dtype=torch.float32))
model_state_dict['int_blocks.3.trip_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(4.6171, dtype=torch.float32))
model_state_dict['int_blocks.3.quad_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(9.0613, dtype=torch.float32))
model_state_dict['int_blocks.3.quad_interaction.scale_cbf'] = torch.nn.Parameter(torch.tensor(30.9794, dtype=torch.float32))
model_state_dict['int_blocks.3.quad_interaction.scale_sbf_sum'] = torch.nn.Parameter(torch.tensor(8.7431, dtype=torch.float32))
model_state_dict['int_blocks.3.atom_edge_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(8.6083, dtype=torch.float32))
model_state_dict['int_blocks.3.atom_edge_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(5.6123, dtype=torch.float32))
model_state_dict['int_blocks.3.edge_atom_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(9.0417, dtype=torch.float32))
model_state_dict['int_blocks.3.edge_atom_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(0.5292, dtype=torch.float32))
model_state_dict['int_blocks.3.atom_interaction.scale_rbf_sum'] = torch.nn.Parameter(torch.tensor(0.5019, dtype=torch.float32))
model_state_dict['int_blocks.3.atom_update.scale_sum'] = torch.nn.Parameter(torch.tensor(0.7639, dtype=torch.float32))
#
model_state_dict['int_blocks.4.trip_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(8.6022, dtype=torch.float32))
model_state_dict['int_blocks.4.trip_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(4.7884, dtype=torch.float32))
model_state_dict['int_blocks.4.quad_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(9.0025, dtype=torch.float32))
model_state_dict['int_blocks.4.quad_interaction.scale_cbf'] = torch.nn.Parameter(torch.tensor(31.1560, dtype=torch.float32))
model_state_dict['int_blocks.4.quad_interaction.scale_sbf_sum'] = torch.nn.Parameter(torch.tensor(8.6648, dtype=torch.float32))
model_state_dict['int_blocks.4.atom_edge_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(8.7595, dtype=torch.float32))
model_state_dict['int_blocks.4.atom_edge_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(5.8084, dtype=torch.float32))
model_state_dict['int_blocks.4.edge_atom_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(9.0961, dtype=torch.float32))
model_state_dict['int_blocks.4.edge_atom_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(0.5163, dtype=torch.float32))
model_state_dict['int_blocks.4.atom_interaction.scale_rbf_sum'] = torch.nn.Parameter(torch.tensor(0.4642, dtype=torch.float32))
model_state_dict['int_blocks.4.atom_update.scale_sum'] = torch.nn.Parameter(torch.tensor(0.6910, dtype=torch.float32))
#
model_state_dict['int_blocks.5.trip_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(8.4562, dtype=torch.float32))
model_state_dict['int_blocks.5.trip_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(4.5697, dtype=torch.float32))
model_state_dict['int_blocks.5.quad_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(9.2125, dtype=torch.float32))
model_state_dict['int_blocks.5.quad_interaction.scale_cbf'] = torch.nn.Parameter(torch.tensor(31.9272, dtype=torch.float32))
model_state_dict['int_blocks.5.quad_interaction.scale_sbf_sum'] = torch.nn.Parameter(torch.tensor(8.1914, dtype=torch.float32))
model_state_dict['int_blocks.5.atom_edge_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(8.7566, dtype=torch.float32))
model_state_dict['int_blocks.5.atom_edge_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(5.6597, dtype=torch.float32))
model_state_dict['int_blocks.5.edge_atom_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(9.4116, dtype=torch.float32))
model_state_dict['int_blocks.5.edge_atom_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(0.4903, dtype=torch.float32))
model_state_dict['int_blocks.5.atom_interaction.scale_rbf_sum'] = torch.nn.Parameter(torch.tensor(0.4351, dtype=torch.float32))
model_state_dict['int_blocks.5.atom_update.scale_sum'] = torch.nn.Parameter(torch.tensor(0.6962, dtype=torch.float32))

model_state_dict['int_blocks.6.trip_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(8.6465, dtype=torch.float32))
model_state_dict['int_blocks.6.trip_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(4.6171, dtype=torch.float32))
model_state_dict['int_blocks.6.quad_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(9.0613, dtype=torch.float32))
model_state_dict['int_blocks.6.quad_interaction.scale_cbf'] = torch.nn.Parameter(torch.tensor(30.9794, dtype=torch.float32))
model_state_dict['int_blocks.6.quad_interaction.scale_sbf_sum'] = torch.nn.Parameter(torch.tensor(8.7431, dtype=torch.float32))
model_state_dict['int_blocks.6.atom_edge_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(8.6083, dtype=torch.float32))
model_state_dict['int_blocks.6.atom_edge_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(5.6123, dtype=torch.float32))
model_state_dict['int_blocks.6.edge_atom_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(9.0417, dtype=torch.float32))
model_state_dict['int_blocks.6.edge_atom_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(0.5292, dtype=torch.float32))
model_state_dict['int_blocks.6.atom_interaction.scale_rbf_sum'] = torch.nn.Parameter(torch.tensor(0.5019, dtype=torch.float32))
model_state_dict['int_blocks.6.atom_update.scale_sum'] = torch.nn.Parameter(torch.tensor(0.7639, dtype=torch.float32))
#
model_state_dict['int_blocks.7.trip_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(8.6022, dtype=torch.float32))
model_state_dict['int_blocks.7.trip_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(4.7884, dtype=torch.float32))
model_state_dict['int_blocks.7.quad_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(9.0025, dtype=torch.float32))
model_state_dict['int_blocks.7.quad_interaction.scale_cbf'] = torch.nn.Parameter(torch.tensor(31.1560, dtype=torch.float32))
model_state_dict['int_blocks.7.quad_interaction.scale_sbf_sum'] = torch.nn.Parameter(torch.tensor(8.6648, dtype=torch.float32))
model_state_dict['int_blocks.7.atom_edge_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(8.7595, dtype=torch.float32))
model_state_dict['int_blocks.7.atom_edge_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(5.8084, dtype=torch.float32))
model_state_dict['int_blocks.7.edge_atom_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(9.0961, dtype=torch.float32))
model_state_dict['int_blocks.7.edge_atom_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(0.5163, dtype=torch.float32))
model_state_dict['int_blocks.7.atom_interaction.scale_rbf_sum'] = torch.nn.Parameter(torch.tensor(0.4642, dtype=torch.float32))
model_state_dict['int_blocks.7.atom_update.scale_sum'] = torch.nn.Parameter(torch.tensor(0.6910, dtype=torch.float32))
#
model_state_dict['int_blocks.8.trip_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(8.4562, dtype=torch.float32))
model_state_dict['int_blocks.8.trip_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(4.5697, dtype=torch.float32))
model_state_dict['int_blocks.8.quad_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(9.2125, dtype=torch.float32))
model_state_dict['int_blocks.8.quad_interaction.scale_cbf'] = torch.nn.Parameter(torch.tensor(31.9272, dtype=torch.float32))
model_state_dict['int_blocks.8.quad_interaction.scale_sbf_sum'] = torch.nn.Parameter(torch.tensor(8.1914, dtype=torch.float32))
model_state_dict['int_blocks.8.atom_edge_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(8.7566, dtype=torch.float32))
model_state_dict['int_blocks.8.atom_edge_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(5.6597, dtype=torch.float32))
model_state_dict['int_blocks.8.edge_atom_interaction.scale_rbf'] = torch.nn.Parameter(torch.tensor(9.4116, dtype=torch.float32))
model_state_dict['int_blocks.8.edge_atom_interaction.scale_cbf_sum'] = torch.nn.Parameter(torch.tensor(0.4903, dtype=torch.float32))
model_state_dict['int_blocks.8.atom_interaction.scale_rbf_sum'] = torch.nn.Parameter(torch.tensor(0.4351, dtype=torch.float32))
model_state_dict['int_blocks.8.atom_update.scale_sum'] = torch.nn.Parameter(torch.tensor(0.6962, dtype=torch.float32))

model_state_dict['out_blocks.0.scale_sum'] = torch.nn.Parameter(torch.tensor(0.8043))
model_state_dict['out_blocks.0.scale_rbf_F'] = torch.nn.Parameter(torch.tensor(8.6099))
model_state_dict['out_blocks.1.scale_sum'] = torch.nn.Parameter(torch.tensor(0.7047))
model_state_dict['out_blocks.1.scale_rbf_F'] = torch.nn.Parameter(torch.tensor(8.0398))
model_state_dict['out_blocks.2.scale_sum'] = torch.nn.Parameter(torch.tensor(0.6786))
model_state_dict['out_blocks.2.scale_rbf_F'] = torch.nn.Parameter(torch.tensor(8.3941))
model_state_dict['out_blocks.3.scale_sum'] = torch.nn.Parameter(torch.tensor(0.6700))
model_state_dict['out_blocks.3.scale_rbf_F'] = torch.nn.Parameter(torch.tensor(8.3941))

model_state_dict['out_blocks.4.scale_sum'] = torch.nn.Parameter(torch.tensor(0.8043))
model_state_dict['out_blocks.4.scale_rbf_F'] = torch.nn.Parameter(torch.tensor(8.6099))
model_state_dict['out_blocks.5.scale_sum'] = torch.nn.Parameter(torch.tensor(0.7047))
model_state_dict['out_blocks.5.scale_rbf_F'] = torch.nn.Parameter(torch.tensor(8.0398))
model_state_dict['out_blocks.6.scale_sum'] = torch.nn.Parameter(torch.tensor(0.6786))
model_state_dict['out_blocks.6.scale_rbf_F'] = torch.nn.Parameter(torch.tensor(8.3941))
model_state_dict['out_blocks.7.scale_sum'] = torch.nn.Parameter(torch.tensor(0.6700))
model_state_dict['out_blocks.7.scale_rbf_F'] = torch.nn.Parameter(torch.tensor(8.3941))

model_state_dict['out_blocks.8.scale_sum'] = torch.nn.Parameter(torch.tensor(0.8043))
model_state_dict['out_blocks.8.scale_rbf_F'] = torch.nn.Parameter(torch.tensor(8.6099))
model_state_dict['out_blocks.9.scale_sum'] = torch.nn.Parameter(torch.tensor(0.7047))
model_state_dict['out_blocks.9.scale_rbf_F'] = torch.nn.Parameter(torch.tensor(8.0398))
model_state_dict['out_blocks.10.scale_sum'] = torch.nn.Parameter(torch.tensor(0.6786))
model_state_dict['out_blocks.10.scale_rbf_F'] = torch.nn.Parameter(torch.tensor(8.3941))
model_state_dict['out_blocks.11.scale_sum'] = torch.nn.Parameter(torch.tensor(0.6700))
model_state_dict['out_blocks.11.scale_rbf_F'] = torch.nn.Parameter(torch.tensor(8.3941))
#
# # model_state_dict['out_blocks.0.scale_rbf_V'] = torch.nn.Parameter(torch.tensor(4.6099, dtype=torch.float32))
# # model_state_dict['out_blocks.1.scale_rbf_V'] = torch.nn.Parameter(torch.tensor(4.0398, dtype=torch.float32))
# # model_state_dict['out_blocks.2.scale_rbf_V'] = torch.nn.Parameter(torch.tensor(4.3941, dtype=torch.float32))
# # model_state_dict['out_blocks.3.scale_rbf_V'] = torch.nn.Parameter(torch.tensor(4.7591, dtype=torch.float32))
# # model_state_dict['out_blocks.4.scale_rbf_V'] = torch.nn.Parameter(torch.tensor(4.8912, dtype=torch.float32))
#
# # 禁用梯度追踪
for key in model_state_dict:
    # if "scale_rbf_sum" in key:
    model_state_dict[key].requires_grad = False
#
# # 打印更新后的缩放因子
# print("更新后的缩放因子:")
# for name, param in model_state_dict.items():
#     print(f"{name}: {param}")
#
# # 步骤 3: 保存更新后的状态字典
new_checkpoint_path = '/home/liud/Documents/code/ocp/configs/s2ef/all/gemnet/scaling_factors/gemnet-oc-is2rv2.pt'  # 替换为新的 .pt 文件路径
torch.save(model_state_dict, new_checkpoint_path)
#
print(f"已将更新后的缩放因子保存到 {new_checkpoint_path}")