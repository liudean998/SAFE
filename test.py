# import torch
# import time
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# def compute_f_rmsd_torch(F_pred, F_ref):
#     """
#     使用 PyTorch 计算力的均方根偏差（F-RMSD）
#
#     参数:
#     - F_pred: 形状为 (N, 3) 的 PyTorch 张量，预测力
#     - F_ref: 形状为 (N, 3) 的 PyTorch 张量，参考力（真实力）
#
#     返回:
#     - F-RMSD 值
#     """
#     diff = F_pred - F_ref
#     rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=1)))
#     return rmsd.item()
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# F_pred = torch.randn(200000, 3, device=device)  # 100万原子的预测力
# F_ref = torch.randn(200000, 3, device=device)   # 100万原子的参考力
#
# a = time.time()
# f_rmsd = compute_f_rmsd_torch(F_pred, F_ref)
# print(f"F-RMSD: {f_rmsd:.6f}")
# b = time.time()
# print(b - a)
#
# import numpy as np
# def compute_rmsd(positions1, positions2):
#     diff = positions1 - positions2
#     return np.sqrt(np.mean(np.sum(diff**2, axis=1)))
#
# positions1 = np.random.randn(200000, 3)
# positions2 = np.random.randn(200000, 3)
# c = time.time()
# p_rmsd = compute_rmsd(positions1, positions2)
# print(f"RMSD: {p_rmsd:}")
# print(time.time() - c)


# import sqlite3
#
# conn = sqlite3.connect('/home/wuyinkai/yxy/gem/result/gemnet_all.db')
# cursor = conn.cursor()
#
# # 查看有哪些表
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# print(cursor.fetchall())
#
# # 查看前几条数据
# cursor.execute("SELECT * FROM systems LIMIT 5")
# for row in cursor.fetchall():
#     print(row)
#
# conn.close()


from ase.db import connect

# 连接 ASE 数据库
db = connect('/home/wuyinkai/yxy/gem/result/gemnet_all.db')

print(f"数据库中共有 {len(db)} 条记录")
i = 1
while True:
    try:
        row = db.get(i)  # 逐行尝试
        print(f"id={row.id}, formula={row.formula}")
        i += 1
    except IndexError:
        break  # 到末尾
