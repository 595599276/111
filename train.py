import argparse
import logging
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
from data.loader import data_loader
from models import TrajectoryGenerator, TrajectoryGenerator_cmi
from utils import (
    displacement_error,
    final_displacement_error,
    get_dset_path,
    int_tuple,
    l2_loss,
    l2_loss_cmi,
    relative_to_abs,
)


__all__ = ["activity2id", "object2id",
           "initialize", "read_data"]

activity2id = {
    "BG": 0,  # background
    "activity_walking": 1,
    "activity_standing": 2,
    "activity_carrying": 3,
    "activity_gesturing": 4,
    "Closing": 5,
    "Opening": 6,
    "Interacts": 7,
    "Exiting": 8,
    "Entering": 9,
    "Talking": 10,
    "Transport_HeavyCarry": 11,
    "Unloading": 12,
    "Pull": 13,
    "Loading": 14,
    "Open_Trunk": 15,
    "Closing_Trunk": 16,
    "Riding": 17,
    "specialized_texting_phone": 18,
    "Person_Person_Interaction": 19,
    "specialized_talking_phone": 20,
    "activity_running": 21,
    "PickUp": 22,
    "specialized_using_tool": 23,
    "SetDown": 24,
    "activity_crouching": 25,
    "activity_sitting": 26,
    "Object_Transfer": 27,
    "Push": 28,
    "PickUp_Person_Vehicle": 29,
}

object2id = {
    "Person": 0,
    "Vehicle": 1,
    "Parking_Meter": 2,
    "Construction_Barrier": 3,
    "Door": 4,
    "Push_Pulled_Object": 5,
    "Construction_Vehicle": 6,
    "Prop": 7,
    "Bike": 8,
    "Dumpster": 9,
}


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def process_args(args):
  """Process arguments.

  Model will be in outbasepath/modelname/runId/save

  Args:
    args: arguments.

  Returns:
    Edited arguments.
  """

  def mkdir(path):
    if not os.path.exists(path):
      os.makedirs(path)

  if args.activation_func == "relu":
    args.activation_func = nn.ReLU
  elif args.activation_func == "tanh":
    args.activation_func = nn.Tanh
  elif args.activation_func == "lrelu":
    args.activation_func = nn.LeakyReLU
  else:
    print("unrecognied activation function, using relu...")
    args.activation_func = nn.ReLU

  args.seq_len = 20

  args.outpath = os.path.join(
      args.outbasepath, args.modelname, str(args.runId).zfill(2))
  mkdir(args.outpath)

  args.save_dir = os.path.join(args.outpath, "save")
  mkdir(args.save_dir)
  args.save_dir_model = os.path.join(args.save_dir, "save")
  args.save_dir_best = os.path.join(args.outpath, "best")
  mkdir(args.save_dir_best)
  args.save_dir_best_model = os.path.join(args.save_dir_best, "save-best")

  args.write_self_sum = True
  args.self_summary_path = os.path.join(args.outpath, "train_sum.txt")

  args.record_val_perf = True
  args.val_perf_path = os.path.join(args.outpath, "val_perf.p")

  # assert os.path.exists(args.frame_path)
  # args.resnet_num_block = [3,4,23,3] # resnet 101
  assert os.path.exists(args.person_feat_path)

  args.object2id = object2id
  args.num_box_class = len(args.object2id)

  # categories of traj
  if args.is_actev:
    args.virat_mov_actids = [
        activity2id["activity_walking"],
        activity2id["activity_running"],
        activity2id["Riding"],
    ]
    args.traj_cats = [
        ["static", 0],
        ["mov", 1],
    ]
    args.scenes = ["0000", "0002", "0400", "0401", "0500"]

  args.num_act = len(activity2id.keys())  # include the BG class

  # has to be 2,4 to match the scene CNN strides
  args.scene_grid_strides = (2, 4)
  args.scene_grids = []
  for stride in args.scene_grid_strides:
    h, w = args.scene_h, args.scene_w
    this_h, this_w = round(h*1.0/stride), round(w*1.0/stride)
    this_h, this_w = int(this_h), int(this_w)
    args.scene_grids.append((this_h, this_w))

  if args.load_best:
    args.load = True
  if args.load_from is not None:
    args.load = True

  # if test, has to load
  if not args.is_train:
    args.load = True
    args.num_epochs = 1
    args.keep_prob = 1.0

  args.activity2id = activity2id
  return args


def initialize(load, load_best, args, engine):
  """Initialize graph with given model weights.

  Args:
    load: boolean, whether to load model weights
    load_best: whether to load from best model path
    args: arguments
    saver:

  Returns:
    None
  """

  if load:
    print("restoring model...")

    load_from = None
    if args.load_from is not None:
      load_from = args.load_from
    else:
      if load_best:
        load_from = args.save_dir_best
      else:
        load_from = args.save_dir
    # load_from = args.load_from
    saver = Saver(1)
    saver.restore(engine, load_from)


parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

parser.add_argument("--dataset_name", default="zara2", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_epochs", default=400, type=int)

parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")

parser.add_argument(
    "--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size"
)
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma"
)
parser.add_argument(
    "--hidden-units",
    type=str,
    default="16",
    help="Hidden units in each hidden layer, splitted with comma",
)
parser.add_argument(
    "--graph_network_out_dims",
    type=int,
    default=32,
    help="dims of every node after through GAT module",
)
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)


parser.add_argument(
    "--lr",
    default=1e-3,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)

parser.add_argument("--best_k", default=20, type=int)
parser.add_argument("--print_every", default=10, type=int)
parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="0", type=str)

parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)



parser.add_argument("--prepropath",default = 'actev_preprocess', type=str)
parser.add_argument("--outbasepath", default = 'next-models/actev_single_model', type=str,
                    help="full path will be outbasepath/modelname/runId")
parser.add_argument("--modelname", default = 'model', type=str)
parser.add_argument("--runId", type=int, default=0,
                    help="used for run the same model multiple times")

# ---- gpu stuff. Now only one gpu is used
parser.add_argument("--gpuid", default=0, type=int)

parser.add_argument("--load", action="store_true",
                    default=False, help="whether to load existing model")
parser.add_argument("--load_best", action="store_true",
                    default=False, help="whether to load the best model")
# use for pre-trained model
parser.add_argument("--load_from", type=str, default=None)

# ------------- experiment settings
parser.add_argument("--is_actev", action="store_true",
                    help="is actev/virat dataset, has activity info")

# ------------------- basic model parameters
parser.add_argument("--emb_size", type=int, default=128)
parser.add_argument("--enc_hidden_size", type=int,
                    default=256, help="hidden size for rnn")
parser.add_argument("--dec_hidden_size", type=int,
                    default=256, help="hidden size for rnn")
parser.add_argument("--activation_func", type=str,
                    default="tanh", help="relu/lrelu/tanh")

# ---- multi decoder
parser.add_argument("--multi_decoder", action="store_true")

# ----------- add person appearance features
parser.add_argument("--person_feat_path",default = 'next-data/actev_personboxfeat', type=str)
parser.add_argument("--person_feat_dim", type=int, default=256)
parser.add_argument("--person_h", type=int, default=9,
                    help="roi align resize to feature size")
parser.add_argument("--person_w", type=int, default=5,
                    help="roi align resize to feature size")

# ---------------- other boxes
parser.add_argument("--random_other", action="store_true",
                    help="randomize top k other boxes")
parser.add_argument("--max_other", type=int, default=15,
                    help="maximum number of other box")
parser.add_argument("--box_emb_size", type=int, default=64)

# ---------- person pose features
parser.add_argument("--add_kp", action="store_true")
parser.add_argument("--kp_size", default=17, type=int)

# --------- scene features
parser.add_argument("--scene_conv_kernel", default=3, type=int)
parser.add_argument("--scene_h", default=36, type=int)
parser.add_argument("--scene_w", default=64, type=int)
parser.add_argument("--scene_class", default=11, type=int)
parser.add_argument("--scene_conv_dim", default=64, type=int)
parser.add_argument("--pool_scale_idx", default=0, type=int)

#  --------- activity
parser.add_argument("--add_activity", action="store_true")

#  --------- loss weight
parser.add_argument("--act_loss_weight", default=1.0, type=float)
parser.add_argument("--grid_loss_weight", default=0.1, type=float)
parser.add_argument("--traj_class_loss_weight", default=1.0, type=float)

# ---------------------------- training hparam
parser.add_argument("--save_period", type=int, default=300,
                    help="num steps to save model and eval")
# drop out rate
parser.add_argument("--keep_prob", default=0.7, type=float,
                    help="1.0 - drop out rate")
# l2 weight decay rate
parser.add_argument("--wd", default=0.0001, type=float,
                    help="l2 weight decay loss")
parser.add_argument("--clip_gradient_norm", default=10, type=float,
                    help="gradient clipping")
parser.add_argument("--optimizer", default="adadelta",
                    help="momentum|adadelta|adam")
parser.add_argument("--learning_rate_decay", default=0.95,
                    type=float, help="learning rate decay")
parser.add_argument("--num_epoch_per_decay", default=2.0,
                    type=float, help="how epoch after which lr decay")
parser.add_argument("--init_lr", default=0.2, type=float,
                    help="Start learning rate")
parser.add_argument("--emb_lr", type=float, default=1.0,
                    help="learning scaling factor for emb variables")

parser.add_argument("--preload_features", action="store_true")
parser.add_argument("--embed_traj_label", action="store_true")





best_ade_2 = 100
best_ade_3 = 100
best_ade_4 = 100


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    logging.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, 'train')
    logging.info("Initializing val dataset")
    _, val_loader = data_loader(args, 'val')

    writer = SummaryWriter()

    n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
    )
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    model = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        alpha=args.alpha,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
    )
    model_cmi = TrajectoryGenerator_cmi(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        alpha=args.alpha,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
    )
    model.cuda()
    model_cmi.cuda()
    optimizer = optim.Adam(
        [
            {"params": model.traj_lstm_model.parameters(), "lr": 1e-2},
            {"params": model.traj_hidden2pos.parameters()},
            
            {"params": model_cmi.traj_lstm_model.parameters(), "lr": 1e-2},
            {"params": model_cmi.traj_hidden2pos.parameters()},
            {"params": model_cmi.gatencoder.parameters(), "lr": 3e-2},
            {"params": model_cmi.graph_lstm_model.parameters(), "lr": 1e-2},
            {"params": model_cmi.traj_gat_hidden2pos.parameters()},
            {"params": model_cmi.pred_lstm_model.parameters()},
            {"params": model_cmi.pred_hidden2pos.parameters()},
        ],
        lr=args.lr,
    )
    '''
    optimizer = optim.Adam(
        [
            {"params": model.traj_lstm_model.parameters(), "lr": 1e-2},
            {"params": model.traj_hidden2pos.parameters()},
            {"params": model.gatencoder.parameters(), "lr": 3e-2},
            {"params": model.graph_lstm_model.parameters(), "lr": 1e-2},
            {"params": model.traj_gat_hidden2pos.parameters()},
            {"params": model.pred_lstm_model.parameters()},
            {"params": model.pred_hidden2pos.parameters()},
        ],
        lr=args.lr,
    )
    '''
    global best_ade_2
    global best_ade_3
    global best_ade_4
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    training_step = 1
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        if epoch < 150:
            training_step = 1
        elif epoch < 200:
            training_step = 2
        elif epoch < 250:
            training_step = 3
        else:
            if epoch == 250:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 5e-3
            training_step = 4
        
        train(args, model, model_cmi, train_loader, optimizer, epoch, training_step, writer)
        validate(args, model, model_cmi, val_loader, epoch, writer, training_step)
        if training_step == 2:
            ade = validate(args, model, model_cmi, val_loader, epoch, writer, training_step)
            is_best = ade < best_ade_2
            best_ade_2 = min(ade, best_ade_2)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_ade_2": best_ade_2,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                f"./checkpoint/checkpoint{epoch}.pth.tar",
            )
        if training_step == 3:
            ade = validate(args, model, model_cmi, val_loader, epoch, writer, training_step)
            is_best = ade < best_ade_3
            best_ade_3 = min(ade, best_ade_3)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_ade_3": best_ade_3,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                f"./checkpoint/checkpoint{epoch}.pth.tar",
            )
        if training_step == 4:
            ade = validate(args, model, model_cmi, val_loader, epoch, writer, training_step)
            is_best = ade < best_ade_4
            best_ade_4 = min(ade, best_ade_4)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_ade_4": best_ade_4,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                f"./checkpoint/checkpoint{epoch}.pth.tar",
            )
    writer.close()


def train(args, model, model_cmi, train_loader, optimizer, epoch, training_step, writer):
    losses = utils.AverageMeter("Loss", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )
    model.train()
    model_cmi.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_rel_gt,
            obs_scene,
            pred_scene,
            loss_mask,
            seq_start_end,
        ) = batch
        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = []
        loss_mask = loss_mask[:, args.obs_len :]
        if training_step == 1:
            model_input = obs_traj_rel
            pred_traj_fake_rel = model(
                model_input, obs_traj, seq_start_end, 1, training_step
            )
            l2_loss_rel.append(
                l2_loss(pred_traj_fake_rel, model_input, loss_mask, mode="raw")
            )
            
            l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
            l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
            for start, end in seq_start_end.data:
                _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
                _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
                _l2_loss_rel = torch.min(_l2_loss_rel) / (
                    (pred_traj_fake_rel.shape[0]) * (end - start)
                )
                l2_loss_sum_rel += _l2_loss_rel

            loss += l2_loss_sum_rel
            losses.update(loss.item(), obs_traj.shape[1])
            loss.backward()
            optimizer.step()
            if batch_idx % args.print_every == 0:
                progress.display(batch_idx)
        elif training_step == 2:
            model_input = obs_traj_rel
            traj_lstm_hidden_states = model(
                model_input, obs_traj, seq_start_end, 1, training_step
            )
            pred_traj_fake_rel = model_cmi(
                traj_lstm_hidden_states, model_input, obs_traj, seq_start_end, obs_scene, 1, training_step
            )
            l2_loss_rel.append(
                l2_loss(pred_traj_fake_rel, model_input, loss_mask, mode="raw")
            )
            
            l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
            l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
            for start, end in seq_start_end.data:
                _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
                _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
                _l2_loss_rel = torch.min(_l2_loss_rel) / (
                    (pred_traj_fake_rel.shape[0]) * (end - start)
                )
                l2_loss_sum_rel += _l2_loss_rel

            loss += l2_loss_sum_rel
            losses.update(loss.item(), obs_traj.shape[1])
            loss.backward()
            optimizer.step()
            if batch_idx % args.print_every == 0:
                progress.display(batch_idx)
        elif training_step == 3:
            model_input = obs_traj_rel
            traj_lstm_hidden_states = model(
                model_input, obs_traj, seq_start_end, 1, training_step
            )
            pred_traj_fake_rel, pred_traj_fake_rel_cmi, n_l = model_cmi(
                traj_lstm_hidden_states, model_input, obs_traj, seq_start_end, obs_scene, 1, training_step
            )
            l2_loss_rel.append(
                l2_loss_cmi(pred_traj_fake_rel, pred_traj_fake_rel_cmi, n_l, model_input, loss_mask, mode="raw")
            )
            l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
            l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
            for start, end in seq_start_end.data:
                _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
                _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
                _l2_loss_rel = torch.min(_l2_loss_rel) / (
                    (pred_traj_fake_rel.shape[0]) * (end - start)
                )
                l2_loss_sum_rel += _l2_loss_rel

            loss += l2_loss_sum_rel
            losses.update(loss.item(), obs_traj.shape[1])
            loss.backward()
            optimizer.step()
            if batch_idx % args.print_every == 0:
                progress.display(batch_idx)
        elif training_step == 4:
            model_input = torch.cat([obs_traj_rel,pred_traj_rel_gt],dim = 0)
            for _ in range(args.best_k):
                traj_lstm_hidden_states = model(
                    model_input, obs_traj, seq_start_end, 1, training_step
                )
                pred_traj_fake_rel= model_cmi(
                    traj_lstm_hidden_states, model_input, obs_traj, seq_start_end, obs_scene, 1, training_step
                )
                l2_loss_rel.append(
                    l2_loss(pred_traj_fake_rel, model_input[-args.pred_len :], loss_mask, mode="raw")
                )
            l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
            l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
            for start, end in seq_start_end.data:
                _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
                _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
                _l2_loss_rel = torch.min(_l2_loss_rel) / (
                    (pred_traj_fake_rel.shape[0]) * (end - start)
                )
                l2_loss_sum_rel += _l2_loss_rel

            loss += l2_loss_sum_rel
            losses.update(loss.item(), obs_traj.shape[1])
            loss.backward()
            optimizer.step()
            if batch_idx % args.print_every == 0:
                progress.display(batch_idx)
               
            
            

    writer.add_scalar("train_loss", losses.avg, epoch)


def validate(args, model, model_cmi, val_loader, epoch, writer, training_step):
    ade = utils.AverageMeter("ADE", ":.6f")
    fde = utils.AverageMeter("FDE", ":.6f")
    progress = utils.ProgressMeter(len(val_loader), [ade, fde], prefix="Test: ")

    model.eval()
    model_cmi.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_rel_gt,
                obs_scene,
                pred_scene,
                loss_mask,
                seq_start_end,
            ) = batch
            if training_step == 2:
                loss_mask = loss_mask[:, :args.obs_len ]
                traj_lstm_hidden_states = model(
                    obs_traj_rel, obs_traj, seq_start_end, 1, training_step
                )
                pred_traj_fake_rel = model_cmi(
                    traj_lstm_hidden_states, obs_traj_rel, obs_traj, seq_start_end, obs_scene, 1, training_step
                )
                pred_traj_fake_rel_predpart = pred_traj_fake_rel[ : args.obs_len]
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, obs_traj[0])
                ade_, fde_ = cal_ade_fde(obs_traj, pred_traj_fake)
                ade_ = ade_ / (obs_traj.shape[1] * args.obs_len)

                fde_ = fde_ / (obs_traj.shape[1])
                ade.update(ade_, obs_traj.shape[1])
                fde.update(fde_, obs_traj.shape[1])
            if training_step == 3:
                loss_mask = loss_mask[:, :args.obs_len ]
                model_input = obs_traj_rel
                traj_lstm_hidden_states = model(
                    model_input, obs_traj, seq_start_end, 1, training_step
                )
                pred_traj_fake_rel, pred_traj_fake_rel_cmi, n_l = model_cmi(
                    traj_lstm_hidden_states, model_input, obs_traj, seq_start_end, obs_scene, 1, training_step
                )
                pred_traj_fake_rel_predpart = pred_traj_fake_rel_cmi[ : args.obs_len]
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, obs_traj[0])
                ade_, fde_ = cal_ade_fde(obs_traj, pred_traj_fake)
                ade_ = ade_ / (obs_traj.shape[1] * args.obs_len)

                fde_ = fde_ / (obs_traj.shape[1])
                ade.update(ade_, obs_traj.shape[1])
                fde.update(fde_, obs_traj.shape[1])
            if training_step == 4:
                loss_mask = loss_mask[:, :args.obs_len ]
                model_input = obs_traj_rel
                traj_lstm_hidden_states = model(
                    model_input, obs_traj, seq_start_end, 1, training_step
                )
                pred_traj_fake_rel = model_cmi(
                    traj_lstm_hidden_states, model_input, obs_traj, seq_start_end, obs_scene, 1, training_step
                )
                pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len :]
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, obs_traj[-1])
                ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
                ade_ = ade_ / (obs_traj.shape[1] * args.pred_len)

                fde_ = fde_ / (obs_traj.shape[1])
                ade.update(ade_, obs_traj.shape[1])
                fde.update(fde_, obs_traj.shape[1])
            if i % args.print_every == 0:
                progress.display(i)
        logging.info(
            "epoch:{epoch} * ADE  {ade.avg:.3f} FDE  {fde.avg:.3f}".format(epoch=epoch, ade=ade, fde=fde)
        )
        writer.add_scalar("val_ade", ade.avg, epoch)
    return ade.avg
    '''
    ade = utils.AverageMeter("ADE", ":.6f")
    fde = utils.AverageMeter("FDE", ":.6f")
    progress = utils.ProgressMeter(len(val_loader), [ade, fde], prefix="Test: ")

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch
            loss_mask = loss_mask[:, args.obs_len :]
            pred_traj_fake_rel = model(obs_traj_rel, obs_traj, seq_start_end)

            pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len :]
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, obs_traj[-1])
            ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
            ade_ = ade_ / (obs_traj.shape[1] * args.pred_len)

            fde_ = fde_ / (obs_traj.shape[1])
            ade.update(ade_, obs_traj.shape[1])
            fde.update(fde_, obs_traj.shape[1])

            if i % args.print_every == 0:
                progress.display(i)

        logging.info(
            " * ADE  {ade.avg:.3f} FDE  {fde.avg:.3f}".format(ade=ade, fde=fde)
        )
        writer.add_scalar("val_ade", ade.avg, epoch)
    return ade.avg
    '''


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return ade, fde


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, filename)
        logging.info("-------------- lower ade ----------------")
        shutil.copyfile(filename, "model_best.pth.tar")


if __name__ == "__main__":
    args = parser.parse_args()
    args.is_train = True
    args.is_test = False
    args.save_output = None
    args = process_args(args)
    utils.set_logger(os.path.join(args.log_dir, "train.log"))
    checkpoint_dir = "./checkpoint"
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)
    main(args)
