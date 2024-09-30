<div align="center">
  <h1>Towards Low-Energy Adaptive Personalization for Resource-Constrained Devices</h1>
</div>


<p align="center">
  <strong><a href="https://arxiv.org/abs/2403.08040">[Paper]</a></strong>
  <strong><a href="https://netsys.doc.ic.ac.uk/">[Team]</a></strong>
</p>

Accepted to *[Accepted to The 9th ACM/IEEE Symposium on Edge Computing (SEC 2024)]([https://euromlsys.eu/](https://acm-ieee-sec.org/2024/)).*


## Abstract 
Microcontroller Units (MCUs) are ideal platforms for edge applications due to their low cost and energy consumption, and are widely used in various applications, including personalized machine learning tasks, where customized models can enhance the task adaptation. However, existing approaches for local on-device personalization mostly support simple ML architectures or require complex local pre-training/training, leading to high energy consumption and negating the low-energy advantage of MCUs. In this paper, we introduce MicroT, an efficient and low-energy MCU personalization approach. $MicroT$ includes a robust, general, but tiny feature extractor, developed through self-supervised knowledge distillation, which trains a task-specific head to enable independent on-device personalization with minimal energy and computational requirements. MicroT implements an MCU-optimized early-exit inference mechanism called stage-decision to further reduce energy costs. This mechanism allows for user-configurable exit criteria (stage-decision ratio) to adaptively balance energy cost with model performance. We evaluated $MicroT$ using two models, three datasets, and two MCU boards. MicroT outperforms traditional transfer learning (TTL) and two SOTA approaches by 2.12 - 11.60% across two models and three datasets. Targeting widely used energy-aware edge devices, MicroT's on-device training requires no additional complex operations, halving the energy cost compared to SOTA approaches by up to 2.28X while keeping SRAM usage below 1MB. During local inference, MicroT reduces energy cost by 14.17% compared to TTL across two boards and two datasets, highlighting its suitability for long-term use on energy-aware resource-constrained MCUs.

<div align="center">
<img src="./figure/overview.png" width="320"> 
</div>

<div align="center">
  <h5>System Overview</h5>
</div>

## 1. Requirements
To get started and download all dependencies, run:

```
pip install -r requirements.txt 
```
We use the Oxford-iiit Pet (pet) [1], The caltech-ucsd birds-200-2011 (bird) [2], and LifeCLEF 2017 (plant) [3] datasets. Please refer to the reference and download them.


## 2. Motivation Experiments

Fine-tuning accuracy results on noised blocks. The bset block-based accuracy is highlighted.

<div align="center">
<img src="./figure/motivation_result.png" width="400"> 
</div>

The code is in `./motivation_exp`.

(1) Train the original model, shown as `./motivation_exp/train_origin_resnet.py`. We also release the model utilised in our paper, please refer to `./motivation_exp/resnet26_model.pth`.

(2) Add noise and finetune, shown as `./motivation_exp/add_noise_resnet.py`.

## 2. Main Experiments

### 2.1 Cifar10-C

The experimental results for the Cifar10-C dataset.

<div align="center">
<img src="./figure/cifar10c_result.png" width="400"> 
</div>

The original model here is same with the original model in Motivation Experiments (trained on Cifar10).

To run, use `python ./exp_cifar10c.py` with appropriate model and parameters (see `exp_cifar10c.py` 113-123 for defaults).

### 2.2 Living17

The experimental results for the Living17 dataset.

<div align="center">
<img src="./figure/Living17_result.png" width="400"> 
</div>

The code is in `./main_exp/Living17`.

(1) Train the original model, shown as `./main_exp/Living17/Living17_ResNet26_origin.py`. We also release the model for Living17 dataset utilised in our paper, please refer to `./main_exp/Living17/ResNet26_origin_Living17.pth`.

(2) Add noise and finetune, shown as `./main_exp/Living17/Living17_ResNet26_finetune.py`.

### 2.3 Cifar-Flip

The experimental results for the Cifar-Flip dataset.

<div align="center">
<img src="./figure/cifarflip_result.png" width="400"> 
</div>

The original model here is same with the original model in Motivation Experiments (trained on Cifar10).

To run, use `python ./exp_cifar10flip.py` with the appropriate model and parameters (see `exp_cifar10flip.py` 101-111 for defaults).

### 2.4 System Cost

The time and energy costs of block-based and full model fine-tuning. The Energy-SavingRate is calculated by comparing the current energy cost to the energy cost of full model fine-tuning.

<div align="center">
<img src="./figure/systemcost.png" width="700"> 
</div>

For the System Cost, please refer to the paper for further details.


## Citation

If you found our work useful please consider citing it:

```bibtex
@misc{huang2024lowenergyondevicepersonalizationmcus,
      title={Low-Energy On-Device Personalization for MCUs}, 
      author={Yushan Huang and Ranya Aloufi and Xavier Cadet and Yuchen Zhao and Payam Barnaghi and Hamed Haddadi},
      year={2024},
      eprint={2403.08040},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.08040}, 
}
```

## Acknowledgments

Our paper and code partially reference Robustbench [1], Breeds [2], Surgical Fine-Tuning [3], and MEMO [4]. We would like to express our gratitude for their open-sourcing of the codebase, which served as the foundation for our work.


## References

[1] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. The oxford-iiit pet dataset.

[2] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The caltech-ucsd birds-200-2011 dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.

[3] Herve Goeau, Pierre Bonnet, and Alexis Joly. Plant identification based on noisy web data: the amazing performance of deep learning (lifeclef 2017). CEUR Workshop Proceedings, 2017.

[4] Zhang, M., Levine, S. and Finn, C., 2022. Memo: Test time robustness via adaptation and augmentation. Advances in neural information processing systems, 35, pp.38629-38642.
