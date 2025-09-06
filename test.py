import torch
import lmdb
import pickle
import inspect
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from ocpmodels.models.gemnet_oc.gemnet_oc_is2rse import GemNetOCRSE

checkpoint_path = '/home/wuyinkai/liud/ocp/checkpoints/2025-05/2025-05-30-19-07-44/best_checkpoint.pt'
test_dataset_path = '/data2/liud/dataset/Cu_self/IS2RE/od_C2H2O2/tags/test/data.lmdb'

check =torch.load(checkpoint_path, map_location=torch.device("cpu"))
model = GemNetOCRSE(**check['config']['model_attributes'], num_atoms=1, bond_feat_dim=1, num_targets=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = nn.DataParallel(model)
model.load_state_dict(check['state_dict'])
model.eval()

# 输入数据进行预测
class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with self.env.begin(write=False) as txn:
            # LMDB 里有多少条数据
            self.length = txn.stat()["entries"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(str(idx).encode("ascii"))
            if byteflow is None:
                raise IndexError(f"Index {idx} not found in LMDB.")
            data = pickle.loads(byteflow)
        return data


dataset = LMDBDataset(test_dataset_path)

dataloader = DataLoader(dataset, batch_size=2, num_workers=2)

for batch in dataloader:
    batch = batch.to(device)
    out_e, out_v, out_p, out_f, main_graph  = model(batch)
    print(out_e)
    break