# Predictive Coding for Grasp Planning
This repo contains open-source code related to the paper **Khan, R., Zhong, H., Das, S., Cai, J., Niemeier, M. (2025) Predictive Coding Explains Asymmetric Connectivity in the Brain: A Neural Network Study. bioRxiv, 2025.02.27.640572**.  

Link to the preprint: https://www.biorxiv.org/content/10.1101/2025.02.27.640572v2

## Abstract
Seminal frameworks of predictive coding propose a hierarchy of generative modules, each attempting to infer the neural representation of the module one level below; the predictions are carried by top-down feedback projections, while the predictive error is propagated by reciprocal forward pathways. Such symmetric feedback connections support visual processing of noisy stimuli in computational models. However, neurophysiological studies have yielded evidence of asymmetric cortical feedback connections. We investigated the contribution of neural feedback during sensorimotor processes, in particular visual processing during grasp planning, by utilizing convolutional neural network models that had been augmented with predictive feedback and were trained to compute grasp positions for real-world objects. After establishing an ameliorative effect of symmetric feedback on grasp detection performance when evaluated on noisy stimuli, we characterized the performance effects of asymmetric feedback, similar to that observed in the cortex. Specifically, we tested model variants extended with short-, medium- and long-range feedback connections (i) originating at the same source layer or (ii) terminating at the same target layer. We found that the performance-enhancing effect of predictive coding under adverse conditions was optimal for medium-range asymmetric feedback. Moreover, this effect was most prominent when medium-range feedback originated at a level of representational abstraction that was proximal to the input layer, in contrast to more distal layers. To conclude, our simulations show that introducing biologically realistic asymmetric predictive feedback improves model robustness to noisy visual stimuli in a neural network model optimized for grasp detection.

## Installation
1. Clone the Repository:
```
git clone https://github.com/khanrom/Feedback_Grasping.git
cd Feedback_Grasping
```
2. Setup Python Environment: Ensure you have Python 3.x installed (the code was tested with Python 3 and PyTorch on an NVIDIA GPU). Install the required packages:
   ```
   pip install torch torchvision numpy toml wandb
   ```
## Dataset

## Training

## Evaluation

## Citation
If you use this code or otherwise build on this research, please cite the paper:
```
@article{Khan2025FeedbackGrasping,
  title   = {Feedback-driven regrasping improves multi-fingered robotic grasping in the real world},
  author  = {Khan, Romesa and Zhong, Hongsheng and Das, Shuvam and Cai, Jack and Niemeier, Matthias},
  journal = {bioRxiv},
  volume  = {2025.02.27.640572},  # bioRxiv ID
  year    = {2025},
  doi     = {10.1101/2025.02.27.640572},
  note    = {Preprint}
}
```







