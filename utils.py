import os
import logging
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)

    return os.path.join(_dir, "datasets", dset_name, dset_type)


def int_tuple(s):
    return tuple(int(i) for i in s.split(","))


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode="raw"):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    # equation below , the first part do noing, can be delete

    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == "raw":
        return loss.sum(dim=2).sum(dim=1)
        

def l2_loss_cmi(pred_traj, pred_traj_cmi, n_l, pred_traj_gt, loss_mask, random=0, mode="average"):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    
    seq_len, batch, _ = pred_traj.size()
    # equation below , the first part do noing, can be delete

    loss_p = (pred_traj_gt.permute(1, 0, 2) - pred_traj_cmi.permute(1, 0, 2)) ** 2
    index = 0
    gt= []
    for i in n_l:
        for j in range(i):
            gt.append(pred_traj_gt[:,index].unsqueeze(1).repeat(1,i+2,1))
            index += 1
    gt = torch.cat(gt,dim =1)
    loss_cmi = (gt - pred_traj) ** 2
    l = []
    index = 0
    for i in n_l:
        loss = loss_cmi[:,index:index+ i*(i+2)]
        loss = loss.view(seq_len,i,i+2,2)
        loss1 = loss[:,:,0]
        loss2 = loss[:,:,1:]
        loss2 = torch.sum(loss2,dim=2)/(i+1)
        loss = (loss1+loss2)/2
        l.append(loss)
        index += i*(i+2)
    l = torch.cat(l,dim = 1).permute(1,0,2)
    l = (l + loss_p)/2
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == "raw":
        return l.sum(dim=2).sum(dim=1)
    
    
    
    
    seq_len, batch, _ = pred_traj.size()
    # equation below , the first part do noing, can be delete
    
    loss_cmi = (pred_traj_gt.permute(1, 0, 2) - pred_traj_cmi.permute(1, 0, 2)) ** 2
    pred_traj_gt_l = []
    index = 0
    for i in n_l:
        for j in range(i):
            times = i+2
            temp = pred_traj_gt[:, index].unsqueeze(1).repeat(1, times, 1)
            index += 1
            pred_traj_gt_l.append(temp)
    pred_traj_gt_l = torch.cat(pred_traj_gt_l,dim =1)
    loss_1 = (pred_traj_gt_l.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2
    li = []
    index = 0
    for i in n_l:
        temp = loss_1[index:index + i*(i+2)]
        temp = temp.view(i+2 ,i , -1, 2).permute(1,0,2,3)
        l =torch.sum(temp,dim=1)/(i+2)
        li.append(l)
        index += i*(i+2)
    
    loss_1 = torch.cat(li,dim=0)
    loss = (loss + loss_1)/2
    
    
    
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == "raw":
        return loss.sum(dim=2).sum(dim=1)


def l2_loss_cmi_3(pred_traj, pred_traj_cmi, n_l, model_input, loss_mask, random=0, mode="average"):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    # equation below , the first part do noing, can be delete
    print(seq_len, batch)
    print(pred_traj.shape, pred_trah_cmi.shape,model_input.shape)
    index = 0
    for i in n_l:
        print(i)
        index += i
    print(index)

    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == "raw":
        return loss.sum(dim=2).sum(dim=1)


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode="sum"):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory. [12, person_num, 2]
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """

    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)

    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode="sum"):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """

    loss = pred_pos_gt - pred_pos
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == "raw":
        return loss
    else:
        return torch.sum(loss)
