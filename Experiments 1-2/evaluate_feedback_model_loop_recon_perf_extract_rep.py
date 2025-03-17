import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from pipelines.dataloader import JacquardDataset
from modules.vgg16_baseline_longer import VGG16Baseline
from grasping_pvgg16_longer_bb import PVGG16SeparateHP as PVGG16
import toml
from modules.loss_functions import simple_MSE_loss
import numpy as np
import os

# Load config file
class Args():

    def __init__(self,config_file):

        config = toml.load(config_file)
        for k,v in config.items():
            setattr(self,k,v)
    
    def print_params(self):
        for x in vars(self):
            print ("{:<20}: {}".format(x, getattr(args, x)))


args = Args('train_grasping_feedback_weights_config.toml')
args.print_params()


def load_model(backbone_ckpt, pcoder1_ckpt, pcoder2_ckpt, pcoder3_ckpt, pcoder4_ckpt, device):
    model = VGG16Baseline()
    pnet = PVGG16(model, build_graph=False, random_init=False)

    # Load the weights for the backbone
    state_dict = torch.load(backbone_ckpt)
    pnet.backbone.load_state_dict(state_dict)

    # Load the weights for pcoder1
    state_dict = torch.load(pcoder1_ckpt)
    pnet.pcoder1.load_state_dict(state_dict['pcoderweights'])

    # Load the weights for pcoder2
    state_dict = torch.load(pcoder2_ckpt)
    pnet.pcoder2.load_state_dict(state_dict['pcoderweights'])

    # Load the weights for pcoder3
    state_dict = torch.load(pcoder3_ckpt)
    pnet.pcoder3.load_state_dict(state_dict['pcoderweights'])

    # Load the weights for pcoder4                                                                                                                                                                                            
    state_dict = torch.load(pcoder4_ckpt)
    pnet.pcoder4.load_state_dict(state_dict['pcoderweights'])

    pnet.to(device)  # Move the entire model to the specified device
    
    return pnet



def add_gaussian_noise(data, mean=0., std_dev=1.):
    noise = torch.normal(mean, std_dev, size=data.shape)
    data_noisy = data + noise
    return data_noisy


def evaluate():
    # Load the trained model
    device = torch.device(args.DEVICE)
    checkpoint_dir= args.TASK_NAME
    model = load_model(args.PRETRAINED_MODEL, checkpoint_dir+args.PCODER1_CKPT, checkpoint_dir+args.PCODER2_CKPT, checkpoint_dir+args.PCODER3_CKPT, checkpoint_dir+args.PCODER4_CKPT, device)
    print(model)
    # Load the test data
    data_root  = args.TEST_DATA_DIR
    print(f"Data root: {data_root}")
    batch_size = args.BATCHSIZE
    max_time_step = args.MAX_TIME_STEP  # Add the maximum number of time steps as an argument

    # Load the test dataset
    dataset = JacquardDataset(args.TEST_DATA_DIR, data_type="COMB")
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    loss_fn = simple_MSE_loss
    total_losses = np.zeros(max_time_step)  # Keep track of losses at each time step

    # Evaluate the model on the test data
    
    with torch.no_grad():
        for gauss_noise_level in [0]:#, 0.25, 0.5, 0.75, 1.0]:

            pred_diffs = {}
            
            for i, (img,mask,label) in enumerate(test_loader):
                if i==335:

                # Adding varying levels of Gaussian noise
                    img = add_gaussian_noise(img, std_dev=gauss_noise_level)

                    if isinstance(img, torch.Tensor):
                        img = img.cpu().numpy()
#                    elif not isinstance(img, np.ndarray):
#                    raise TypeError(f"Expected img to be a torch.Tensor or np.ndarray, but got {type(img)}")

                    
                    np.save('pred_diffs_dir/imgs/img_'+str(i)+'noise_level_'+str(gauss_noise_level)+'.npy', img)

                    img = torch.from_numpy(img)
                    img = img.to(device)
            
                    label = label.to(device)

                    # Add time step loop
                    pred_save_dir = f'pred_diffs_dir/img{i}_nl{gauss_noise_level}/'                                                              
                    os.makedirs(pred_save_dir, exist_ok=True)
                    for t in range(max_time_step):  
                        grasp_pred = model(img if t==0 else None)

                        # Convert to NumPy array                                                                                           
                        grasp_pred_array = grasp_pred.cpu().numpy()                                                                           
                        # Save the NumPy array to a .npy file
                        np.save(os.path.join(pred_save_dir,f'img{i}_nl{gauss_noise_level}_ts{t}.npy'), grasp_pred_array)
                    
#                    loss = loss_fn(grasp_pred, label)
#                    total_losses[t] += loss.item()  # Add loss to the correct time step

               
#            # Calculate average loss at each time step
#            average_losses = total_losses / len(test_loader)
#            #for t in range(max_time_step):
#            #    print(f'Average test loss for noise level {gauss_noise_level} at time step {t}: {average_losses[t]:.4f}')
#            print(f'Average test loss at each timestep, for noise level {gauss_noise_level}: {average_losses}')
    
if __name__ == "__main__":
    evaluate()
