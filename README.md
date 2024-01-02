# SSL Sandbox: Modular SSL Models

This library was created to generate and train SSL models from multiple landmark methods, yet with highly customizeable architectures and training methods. 
Replicable models include the following:
- SimSiam (Chen and He, 2020: https://arxiv.org/abs/2011.10566)
- BYOL (Grill et al., 2020: https://arxiv.org/abs/2006.07733)
- SimCLR (Chen et al., 2020: https://arxiv.org/abs/2002.05709)
- MoCoV3 (Chen et al., 2021: https://arxiv.org/abs/2104.02057)
- Barlow Twins (Zbontar et al., 2021: https://arxiv.org/abs/2103.03230)
- VICReg (Bardes et al., 2021: https://arxiv.org/abs/2105.04906)
- MEC (Liu et al., 2022: https://arxiv.org/abs/2210.11464)
- DINO (Caron et al., 2021: https://arxiv.org/abs/2104.14294)

This library is extended with multiple training techniques including the following:
- Adversarial SSL pretraining (Jiang et al., 2020: https://arxiv.org/abs/2010.13337)
- Adversarial supervised finetuning (Goodfellow et al., 2014: https://arxiv.org/abs/1412.6572 and Madry et al., 2017: https://arxiv.org/abs/1706.06083)
- SSL poison generation (He et al., 2023: https://arxiv.org/abs/2202.11202)

Models and algorithms are highly customizeable. The customizeable settings include the following:
- Online Encoder - Any torchvision ResNet and most TIMM ViTs
- Online Projector - Multiple options, mirroring the projectors used in the above SSL methods, with customizeable hidden layer and output layer sizes
- Target Encoder - Mimics online encoder, optionally with EMA update
- Target Projector - Mimics online projector, optionally with EMA update
- Asymmetric Predictor - MLP from BYOL, SimSiam, and MoCoV3 papers
  - Optionally applied as an 'optimal predictor' following DirectSet (Wang et al., 2022: https://arxiv.org/abs/2110.04947)
- Loss Functions - Multiple options, mirroring the losses used in the above SSL methods, with customizeable parameters
- Optional stop-gradient on target network, optional symmetrized loss, optional LARS, optional cosine LR decay, optional linear warmup
- Optional multiview (i.e., more than 2) augmentations

This code processes models and data on CUDA - CPU-only training is not available.
Training on 1 GPU is permissible with or without DistributedDataParallel. 
Training on multiple GPUs is permissible with DistributedDataParallel

### Citation

If you use this repo, please cite the originating Github repo: 
https://github.com/JStyborski/SSL_Sandbox

### Preparation

Required libraries
- PyTorch (>v2.1.0): Installs along with torchvision and torchaudio from https://pytorch.org/get-started/locally/
- torchmetrics: Recommend installing along with PyTorch-Lightning, as in "conda install -c conda-forge pytorch-lightning"
- timm: Recommend installing along with HuggingFace, as in "conda install -c fastai timm"
Some auxiliary libraries (used by portions of the code that are currently unused or commented out) include matplotlib, torchattacks, and probably some others.

### SSL Poison Generation

File: SSL_Poison_DDP.py

Generate SSL poisons following the "Contrastive Poisoning" (CP) method (He et al., 2023: https://arxiv.org/abs/2202.11202).
This method is similar to the "Unlearnable Examples" method (Huang et al., 2021: https://arxiv.org/abs/2101.04898).
Note that unlike SSL-based adversarial examples, CP backpropagates through augmentations to the original images.
The original CP code was implemented only for CIFAR, where all input images are small and of identical sizes.
The adapted method in this code is applicable for any image library, even those with varying image sizes, though with drawbacks to code efficiency.
The implemented method works as follows:
- Create a 'shadow' directory of perturbation tensors that correspond to the source image directory
- At each batch load step, load the source image and corresponding perturbation tensors
- Combine source image and perturbation, then augment with multiple views
- Push augmented samples through the model and backpropagate
- Train the model or perturbations to minimize loss
- After training, synthesize final 'poisoned' images by saving combined source image + perturbation

Example command line inputs:
Unlearnable example poison generation for CIFAR100 on SimCLR with ResNet18 encoder:
python SSL_Poison_DDP.py --trainRoot ../Datasets/CIFAR100/train --deltaRoot ../Datasets/Poisoned_CIFAR/CP_100/deltas --poisonRoot ../Datasets/Poisoned_CIFAR/CP_100/train 
--ptPrefix CP --prdHidDim 0 --winceBeta 1.0 --applySG False

Unlearnable example poison generation with DDP for a 20% subset of ImageNet100 on BYOL with ViT_Small encoder:
CUDA_VISIBLE_DEVICES=0,1 python SSL_Poison_DDP.py --useDDP True --workers 8 --trainRoot ../Datasets/ImageNet100_20pct/train
--deltaRoot ../Datasets/Poisoned_ImageNet/CP_100/deltas --poisonRoot ../Datasets/Poisoned_ImageNet/CP_100/train --ptPrefix CP --encArch vit_small --momEncBeta 0.999

### SSL Pretraining

File: SSL_Pretrain_DDP.py

Unfinished description

### Model Finetuning / Linear Probe

File: SSL_Finetune_DDP.py

Unfinished description

### Trained Model Evaluation

File: SSL_Evaluate.py

Unfinished description

### Development Notes

In adapting the above methods, the focus was on customizeability. Not all features of all methods are exactly replicable. Caveats include the following:
- Contrastive loss is implemented via an explicit split between positive and negative samples following the method in BYOL (Grill et al., 2020: https://arxiv.org/abs/2006.07733).
  - A beta coefficient is used to weight the contrastive terms: beta=0.0 ignores contrastive terms (as in SimSiam and BYOL), beta=1.0 fully considers contrastive terms (as in SimCLR and MoCoV3).
- The projector architectures included in the code have the correct layers types, but it is up to the user to set the desired hidden, bottleneck, and output sizes.
- Multiview is not the same as MultiCrop (Caron et al., 2021: https://arxiv.org/abs/2006.09882) as multiview does not support multiple image resolutions (yet).
- With Multiview (>2 views) and symmetrized loss, loss functions calculate loss between all permutations of online/target outputs for different views and average the results
  - This may be different from official implementations of MultiCrop.
- 
