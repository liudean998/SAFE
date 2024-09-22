# import argparse
# import glob
# import os
# import pickle
# import ase.io
# import lmdb
# from ocpmodels.preprocessing import AtomsToGraphs
#
#
# def write_images_to_lmdb(mp_arg, start_idx):
#     a2g, db_path, extxyz_files, energy_corrections, args = mp_arg
#     db = lmdb.open(
#         db_path,
#         map_size=1099511627776 * 2,
#         subdir=False,
#         meminit=False,
#         map_async=True,
#     )
#     txn = db.begin(write=True)
#     idx = start_idx
#     for file_path in extxyz_files:
#         xyz_idx = os.path.splitext(os.path.basename(file_path))[0]
#         sid = int(xyz_idx.split("random")[1])
#         frame = ase.io.read(file_path, "-1")  # 只选择最后一个点
#         data_object = a2g.convert(frame)
#         data_object.sid = sid
#         data_object.fid = 0  # 只有一个点，fid设为0
#         if args.ref_energy:
#             ref_energy = energy_corrections.get(xyz_idx, 0)
#             data_object.y -= ref_energy
#         txn.put(
#             f"{idx}".encode("ascii"),
#             pickle.dumps(data_object, protocol=-1),
#         )
#         idx += 1
#     txn.commit()
#     db.sync()
#     db.close()
#     return idx
#
#
# def main(args: argparse.Namespace):
#     a2g = AtomsToGraphs(
#         max_neigh=50,
#         radius=6,
#         r_energy=not args.test_data,
#         r_forces=not args.test_data,
#         r_fixed=True,
#         r_distances=False,
#         r_edges=args.get_edges,
#     )
#
#     os.makedirs(os.path.join(args.out_path), exist_ok=True)
#     db_path = os.path.join(args.out_path, "data.lmdb")
#     idx = 0
#
#     for dir_path in glob.glob(os.path.join(args.data_path, "*")):
#         train_dir = os.path.join(dir_path, "test_all")
#         change_txt = glob.glob(os.path.join(dir_path, "*_change.txt"))[0]
#
#         energy_corrections = {}
#         with open(change_txt, 'r') as file:
#             for line in file:
#                 name, energy = line.strip().split(',')
#                 energy_corrections[name] = float(energy)
#
#         extxyz_files = glob.glob(os.path.join(train_dir, "*.extxyz"))
#         mp_arg = (a2g, db_path, extxyz_files, energy_corrections, args)
#         idx = write_images_to_lmdb(mp_arg, idx)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data-path", help="Path to dir containing train folders and change.txt files")
#     parser.add_argument("--out-path", help="Directory to save extracted features. Will create if doesn't exist")
#     parser.add_argument("--get-edges", action="store_true",
#                         help="Store edge indices in LMDB, ~10x storage requirement.")
#     parser.add_argument("--num-workers", type=int, default=1, help="No. of processors to use")
#     parser.add_argument("--ref-energy", action="store_true", help="Subtract reference energies from data")
#     parser.add_argument("--test-data", action="store_true", help="Is data being processed test data?")
#     args = parser.parse_args()
#     main(args)


import argparse
import glob
import os
import pickle
import ase.io
import lmdb
from ocpmodels.preprocessing import AtomsToGraphs


def write_images_to_lmdb(mp_arg, start_idx):
    a2g, db_path, extxyz_files, energy_corrections, args = mp_arg
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    txn = db.begin(write=True)
    idx = start_idx
    for file_path in extxyz_files:
        xyz_idx = os.path.splitext(os.path.basename(file_path))[0]
        sid = int(xyz_idx.split("random")[1])
        traj_frames = ase.io.read(file_path, ":")[-1:]
        for fid, frame in enumerate(traj_frames, start=len(traj_frames) - 1):
            data_object = a2g.convert(frame)
            data_object.sid = sid
            data_object.fid = fid
            if args.ref_energy:
                ref_energy = energy_corrections.get(xyz_idx, 0)
                data_object.y -= ref_energy
            txn.put(
                f"{idx}".encode("ascii"),
                pickle.dumps(data_object, protocol=-1),
            )
            idx += 1
    txn.commit()
    db.sync()
    db.close()
    return idx


def main(args: argparse.Namespace):
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=not args.test_data,
        r_forces=not args.test_data,
        r_fixed=True,
        r_distances=False,
        r_edges=args.get_edges,
    )

    os.makedirs(os.path.join(args.out_path), exist_ok=True)
    db_path = os.path.join(args.out_path, "data.lmdb")
    idx = 0

    for dir_path in glob.glob(os.path.join(args.data_path, "*")):
        train_dir = os.path.join(dir_path, "all")
        change_txt = glob.glob(os.path.join(dir_path, "*_change.txt"))[0]
        # change_txt = "/media/zjy/ST2G/data/H/H_change.txt"

        energy_corrections = {}
        with open(change_txt, 'r') as file:
            for line in file:
                name, energy = line.strip().split(',')
                energy_corrections[name] = float(energy)

        extxyz_files = glob.glob(os.path.join(train_dir, "*.extxyz"))
        mp_arg = (a2g, db_path, extxyz_files, energy_corrections, args)
        idx = write_images_to_lmdb(mp_arg, idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", help="Path to dir containing train folders and change.txt files")
    parser.add_argument("--out-path", help="Directory to save extracted features. Will create if doesn't exist")
    parser.add_argument("--get-edges", action="store_true",
                        help="Store edge indices in LMDB, ~10x storage requirement.")
    parser.add_argument("--num-workers", type=int, default=1, help="No. of processors to use")
    parser.add_argument("--ref-energy", action="store_true", help="Subtract reference energies from data")
    parser.add_argument("--test-data", action="store_true", help="Is data being processed test data?")
    args = parser.parse_args()
    main(args)