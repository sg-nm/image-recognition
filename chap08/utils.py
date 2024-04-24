import os
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

# print for each gpu
def setup_for_distributed(rank):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        builtin_print('[RANK:{}]'.format(rank), *args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print('Multiple GPUs training')
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = True
    else:
        print('single GPU training')
        args.distributed = False
        cudnn.benchmark = True
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.world_size = 1
        args.rank = 0
        return

    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    cudnn.benchmark = True
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, 
                                         init_method=args.dist_url, 
                                         world_size=args.world_size, 
                                         rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank)



class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
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