import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import random
import argparse
import sys, os
import logging
from dataset import Kinetics
from model import GlobalLocal

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--work_dir', type=str, default='work_dirs/')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enable = False

setup_seed(2021)
args = parse_args()

world_size = int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=world_size)

if args.local_rank == 0:
    os.system("mkdir -p %s/checkpoints"%args.work_dir)
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(args.work_dir + '/log.txt')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

dataset = Kinetics(folder='data/kinetics', video_list='data/kinetics/list.txt', audio_zip_name='data/kinetics/kinetics_audio.zip')
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, sampler=sampler)

model = GlobalLocal()
model.cuda(args.local_rank)
model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

opt = torch.optim.Adam(model.parameters(), lr=args.lr)
batch_size = args.batch_size * world_size
stepsize = len(dataset) // batch_size +1

for e in range(args.epochs):
    model.train()
    sampler.set_epoch(e)

    for b, (img, audio) in enumerate(loader):

        img = img.cuda(args.local_rank)
        audio = audio.cuda(args.local_rank)

        opt.zero_grad()
        loss_global, loss_local = model(img, audio)
        loss = loss_global + loss_local
        loss.backward()
        opt.step()

        if args.local_rank == 0 and b % args.print_freq == 0:
            logger.info('Epoch %d/%d Batch %d/%d Loss Global %f Loss Local %f Loss %f'%(e, args.epochs, b, stepsize, loss_global.item(), loss_local.item(), loss.item()))

    if args.local_rank == 0:
        torch.save(model.state_dict(), '%s/checkpoints/epoch_%d.pth'%(args.work_dir, e))
