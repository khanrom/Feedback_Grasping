#########################
# In this script we train PVGG on ImageNet
# We use the pretrained model and only train feedback connections.
#########################
# %%
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import wandb

from torch.utils.data import DataLoader, Subset
from pipelines.dataloader import JacquardDataset

import torch.nn as nn
import torch.optim as optim
import numpy as np
from ranger import Ranger

import os
import toml
import time


################################################
#       Global configs
################################################

class Args():

    def __init__(self, config_file):

        config = toml.load(config_file)
        for k, v in config.items():
            setattr(self, k, v)

    def print_params(self):
        for x in vars(self):
            print("{:<20}: {}".format(x, getattr(args, x)))


args = Args('train_grasping_feedback_weights_config.toml')
args.print_params()

# %%


# Setup the training
if args.RANDOM_SEED:
    np.random.seed(args.RANDOM_SEED)
    torch.manual_seed(args.RANDOM_SEED)

#if not os.path.exists(args.SAVE_DIR):
#    print(f'Creating a new dir : {args.SAVE_DIR}')
#    os.mkdir(args.SAVE_DIR)

device = torch.device(args.DEVICE)

################################################
#          Net , optimizers
################################################
## Update the following to change the network architecture
from modules.vgg16_baseline_longer import VGG16Baseline
from grasping_pvgg16_longer_bb import PVGG16SeparateHP as PVGG16

model = VGG16Baseline()
state_dict = torch.load(args.PRETRAINED_MODEL)
model.load_state_dict(state_dict)
pnet = PVGG16(model, build_graph=True, random_init=False)
pnet.to(device)

NUMBER_OF_PCODERS = pnet.number_of_pcoders

loss_function = nn.MSELoss()

if args.OPTIM_NAME == 'SGD':
    optimizer = optim.SGD(
        [{'params': getattr(pnet, f"pcoder{x + 1}").pmodule.parameters()} for x in range(NUMBER_OF_PCODERS)],
        lr=args.LR,
        weight_decay=args.WEIGHT_DECAY
        )
if args.OPTIM_NAME == 'Adam':
    optimizer = optim.Adam(
        [{'params': getattr(pnet, f"pcoder{x + 1}").pmodule.parameters()} for x in range(NUMBER_OF_PCODERS)],
        lr=args.LR,
        weight_decay=args.WEIGHT_DECAY
        )
# if args.OPTIM_NAME=='Ranger':
#     # optmizer
#     optimizer = Ranger(model.parameters(), lr=args.LR, weight_decay=args.WEIGHT_DECAY)

if args.SCHEDULER == 'cosine_annealing':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3)

################################################
#       Dataset and train-test helpers
################################################
transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

data_root = args.DATA_DIR
print(f"Data root: {data_root}")
batch_size = args.BATCHSIZE
num_workers = args.NUM_WORKERS
train_val_split = args.TRAIN_VAL_SPLIT
dataset = JacquardDataset(data_root, data_type="COMB")
N = len(dataset)
rand_ind = torch.randperm(N)
train_ind = rand_ind[:int(N * train_val_split)]
val_ind = rand_ind[int(N * train_val_split):]
train_set = Subset(dataset, train_ind)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_set = Subset(dataset, val_ind)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# initialize wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="Grasping_Feedback_Connectivity",
    # track hyperparameters and run metadata
    config={
        "LR": args.LR,
        "BATCHSIZE": args.BATCHSIZE,
        "NUM_EPOCHS": args.NUM_EPOCHS,
        "WEIGHT_DECAY": args.WEIGHT_DECAY,
        "FEEDBACK_CONNECTIVITY": 'conn_TetraLoop_lonbase'
    }
)


def train_pcoders(net, epoch, train_loader, verbose=True):
    ''' A training epoch '''

    net.train()

    tstart = time.time()

    for batch_index, (images, _, _) in enumerate(train_loader):
        net.reset()
        images = images.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        for i in range(NUMBER_OF_PCODERS):
            if i == 0:
                a = loss_function(net.pcoder1.prd, images)
                loss = a
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_curr = getattr(net, f"pcoder{i + 1}")
                a = loss_function(pcoder_curr.prd, pcoder_pre.rep)
                loss += a
        
            # log metrics to wandb
            wandb.log({f"MSE Train/PCoder{i + 1}": a.item()})

        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.BATCHSIZE + len(images),
            total_samples=len(train_loader.dataset)
        ))
        print('Time taken:', time.time() - tstart)
        
        # log metrics to wandb
        wandb.log({"MSE Train/Sum": loss.item()})


def test_pcoders(net, epoch, test_loader, verbose=True):
    ''' A testing epoch '''

    net.eval()

    tstart = time.time()
    final_loss = [0 for i in range(NUMBER_OF_PCODERS)]
    for batch_index, (images, _, _) in enumerate(test_loader):
        net.reset()
        images = images.to(device)
        with torch.no_grad():
            outputs = net(images)
        for i in range(NUMBER_OF_PCODERS):
            if i == 0:
                final_loss[i] += loss_function(net.pcoder1.prd, images).item()
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_curr = getattr(net, f"pcoder{i + 1}")
                final_loss[i] += loss_function(pcoder_curr.prd, pcoder_pre.rep).item()

    loss_sum = 0
    for i in range(NUMBER_OF_PCODERS):
        final_loss[i] /= len(test_loader)
        loss_sum += final_loss[i]

        # log metrics to wandb
        wandb.log({f"MSE Test/PCoder{i + 1}": final_loss[i]})
    # log metrics to wandb
    wandb.log({"MSE Test/Sum": loss_sum})
    print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        loss_sum,
        optimizer.param_groups[0]['lr'],
        epoch=epoch,
        trained_samples=batch_index * args.BATCHSIZE + len(images),
        total_samples=len(test_loader.dataset)
    ))
    print('Time taken:', time.time() - tstart)


################################################
#        Load checkpoints if given...
################################################

if args.RESUME_TRAINING:

    assert len(args.RESUME_CKPTS) == NUMBER_OF_PCODERS;
    'the number os ckpts provided is not equal to the number of pcoders'

    print('-' * 30)
    print(f'Loading checkpoint from {args.RESUME_CKPTS}')
    print('-' * 30)

    for x in range(NUMBER_OF_PCODERS):
        checkpoint = torch.load(args.RESUME_CKPTS[x])
        args.START_EPOCH = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        getattr(pnet, f"pcoder{x + 1}").pmodule.load_state_dict(
            {k[len('pmodule.'):]: v for k, v in checkpoint['pcoderweights'].items()})

    print('Checkpoint loaded...')

else:
    print("Training from scratch...")



################################################
#              Train loops
################################################
for epoch in range(args.START_EPOCH, args.NUM_EPOCHS):
    train_pcoders(pnet, epoch, train_loader)

    test_pcoders(pnet, epoch, val_loader)

    #for pcod_idx in range(NUMBER_OF_PCODERS):
    #    torch.save({
    #        'pcoderweights': getattr(pnet, f"pcoder{pcod_idx + 1}").state_dict(),
    #        'optimizer': optimizer.state_dict(),
    #        'epoch': epoch,
    #    }, f'{args.TASK_NAME}/pnet_pretrained_pc{pcod_idx + 1}_{epoch:03d}.pth')

if not os.path.exists(args.TASK_NAME): 
    os.makedirs(args.TASK_NAME)
    
for pcod_idx in range(NUMBER_OF_PCODERS):                                                            
        torch.save({                                                                                     
            'pcoderweights': getattr(pnet, f"pcoder{pcod_idx + 1}").state_dict(),                        
            'optimizer': optimizer.state_dict(),                                                                     'epoch': args.NUM_EPOCHS,         
        }, f'{args.TASK_NAME}/pnet_pretrained_pc{pcod_idx + 1}_{args.NUM_EPOCHS:03d}.pth')     
    
# finish the wandb run
wandb.finish()
