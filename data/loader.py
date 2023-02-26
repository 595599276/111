from torch.utils.data import DataLoader

from data.trajectories import TrajectoryDataset, seq_collate1, seq_collate2, ActevDataset


def data_loader(args, path):
    if args.dataset_name  == 'actev':
        dset = ActevDataset(
            args,
            data_type = path)
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers,
            collate_fn=seq_collate1,
            pin_memory=True)
    else:
        dset = TrajectoryDataset(
            path,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=args.skip,
            delim=args.delim)
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers,
            collate_fn=seq_collate2,
            pin_memory=True)
    return dset, loader
