# HNU in AnDi Challenge

## Zihan Huang

*School of Physics and Electronics, Hunan University, Changsha, China*

Summary of our methods for task 1 (inference of the anomalous diffusion exponent) in AnDi Challenge.

### Data Generation
The file `generate_trajectory.py` is utilized to generate training data based on the module [`andi_datasets`](https://github.com/AnDiChallenge/ANDI_datasets):

`python generate_trajectory.py --l 200 --N 2000000`

where `--l` denotes the length of trajectory and `--N` denotes the number of trajectories.

The length of a trajectory provided by `andi-datasets` ranges from 10 to 999. To handle such a long span, we generated 1D fixed-length trajectory data at 43 specific lengths as the training dataset, as shown in the following table. The number of trajectories for each specific length is also given. Each trajectory is normalized so that its position's average and standard deviation are 0 and 1 respectively. The total size of training dataset is about 330 GB.

| Length | Number | Length | Number | Length | Number | Length | Number |
|  :-:  | :-:   |  :-:  | :-:   | :-: | :-:  | :-:  | :-:  |
| 10    |5000000 |
| 15  | 5000000 |

### System Environment
* OS: Ubuntu 16.04
* GPU: RTX 2080Ti
* Python==3.7.4
* PyTorch==1.6.0
* cuda==10.2
* cuDNN==7.6.5
