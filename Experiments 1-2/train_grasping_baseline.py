import yaml
import json
from types import SimpleNamespace
from pipelines.dataloader import JacquardDataset
from pipelines.utils import show_tensor
import torch
from torch.utils.data import DataLoader, Subset
from modules.vgg16_baseline import VGG16Baseline
from tqdm import tqdm
from modules.loss_functions import simple_MSE_loss
from torch.utils.tensorboard import SummaryWriter  # Import the SummaryWriter class

with open('config.yaml', 'r') as file:
    p = yaml.safe_load(file)
    params = json.loads(json.dumps(p), object_hook=lambda d: SimpleNamespace(**d))


def train():
    # hyperparameters
    device = params.Meta_data.DEVICE
    epoch = params.Training_data_baseline.EPOCH
    batch_size = params.Training_data_baseline.BATCH_SIZE
    lr = params.Training_data_baseline.LR
    weight_decay = params.Training_data_baseline.WEIGHT_DECAY
    train_val_split = params.Training_data_baseline.TRAIN_VAL_SPLIT

    # dataset
    dataset = JacquardDataset(params.Meta_data.DATA_DIR, data_type="COMB")
    N = len(dataset)
    rand_ind = torch.randperm(N)
    train_ind = rand_ind[:int(N*train_val_split)]
    val_ind = rand_ind[int(N*train_val_split):]
    train_set = Subset(dataset, train_ind)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_set = Subset(dataset, val_ind)
    test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # model
    model = VGG16Baseline()
    optmizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = simple_MSE_loss
    model.to(device)

    train_losses = []
    val_losses = []

    # Create the SummaryWriter instance
    writer = SummaryWriter()

    for e in tqdm(range(epoch)):
        for img, mask, label in train_loader:
            img=img.to(device)
            mask=mask.to(device)
            label=label.to(device)
            grasp_pred = model(img)
            loss = loss_fn(grasp_pred, label)
            loss.backward()
            optmizer.step()
            optmizer.zero_grad()
            train_losses.append(loss.item())
            
        # Log the training loss to TensorBoard
        writer.add_scalar('Loss/train', train_losses[-1], e)
        
        for img, mask, label in test_loader:
            img=img.to(device)
            mask=mask.to(device)
            label=label.to(device)
            grasp_pred = model(img)
            loss = loss_fn(grasp_pred, label)
            val_losses.append(loss.item())

        # Log the validation loss to TensorBoard
        writer.add_scalar('Loss/validation', val_losses[-1], e)
        
        print("Epoch: ", e, "Train Loss: ", train_losses[-1], "Val Loss: ", val_losses[-1])

    # Close the SummaryWriter
    writer.close()
    
    return model


if __name__ == "__main__":
    model_name="vgg16_baseline.pth"
    model = train()
    torch.save(model.state_dict(), model_name)
