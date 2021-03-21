# Anomalous HNU in AnDi Challenge

Summary of our LSTM-based methods for task 1 (**inference of the anomalous diffusion exponent**) in AnDi Challenge.

>**Note: ** The report of an improved version of our methods based on WaveNet is in progress.

## [Zihan Huang](http://grjl.hnu.edu.cn/p/2020162)

*School of Physics and Electronics, Hunan University, Changsha, China*

### 1. Data Generation
The file `generate_trajectory.py` is utilized to generate training data based on the module [`andi_datasets`](https://github.com/AnDiChallenge/ANDI_datasets):
```
python generate_trajectory.py --l 200 --N 2000000
```
where `--l` denotes the length of trajectory and `--N` denotes the number of trajectories.

The length of a trajectory provided by `andi-datasets` ranges from 10 to 999. To handle such a long span, we generated 1D fixed-length trajectory data at 43 specific lengths as the training dataset, as shown in the following table. The number of trajectories for each specific length is also given. Each trajectory is normalized so that its position's average and standard deviation are 0 and 1 respectively. The total size of training dataset is about 330 GB.

| Length | Number | Length | Number | Length | Number | Length | Number |
|  :-:  | :-:   |  :-:  | :-:   | :-: | :-:  | :-:  | :-:  |
| 10    |5000000 | 105  |2000000 |275 |1500000 |400 |1000000 |
| 15  | 5000000 | 110  |2000000 |300  |1500000 |425 |1000000 |
| 20  | 5000000 | 115  |2000000 |325  |1500000 |450 |1000000 |
| 25  | 5000000 | 120  |2000000 |350  |1500000 |475 |1000000 |
| 30  | 5000000 | 125  |2000000 |375  |1500000 |500 |1000000 |
| 40  | 5000000 | 150  |2000000 |||550 |1000000 |
| 45  | 5000000 | 175  |2000000 |||600 |1000000 |
| 50  | 5000000 | 200  |2000000 |||650 |1000000 |
| 55  | 5000000 | 225  |2000000 |||700 |1000000 |
| 60  | 5000000 | 250  |2000000 |||750 |1000000 |
| 70  | 5000000 |||||800 |1000000 |
| 80  | 5000000 |||||850 |1000000 |
| 90  | 5000000 |||||900 |1000000 |
| 100  | 5000000 |||||950 |1000000 |

### 2. Model Training
We use an **LSTM**-based RNN model for this task. For 1D trajectories, the input dimension is set as 1. The number of features in the hidden state is 64, and the number of recurrent LSTM layers is 3. A fully-connected layer with a dropout rate *p* = 0.2 and one output node is added at the end of model. For this regression task, no activation function is applied on the output layer. The PyTorch version of our model is as follow:

```python
class AnDiModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, state = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:,-1,:])
        return out

model = AnDiModel(1, 64, 3, 1)
```

Models for each specific length are trained separately. 80% of training data is used for training, while the other is used for validation. We implement the LSTM-based model by PyTorch 1.6.0 on RTX 2080Ti. The model is trained by back propagation method with a batch size 512, where loss function is the mean squared error (MSE). The optimizer is Adam with a learning rate *l* = 0.001. The learning rate is changed as *l* -> *l*/5 if the valid loss doesn't decrease after 2 epochs. When the number of such changes exceeds 1, the training process is early stopped to save time and avoid overfitting. The best model for a specific length is determined by the mean absolute error (MAE) of validation set. We choose the epoch with the lowest valid MAE as the best model.

The codes for training are in the file `train_lstm.py`, where the usage is:
```
python train_lstm.py --l 300 --f 0
```
`--l` denotes the specific length and `--f` denotes the fold ranging from 0 to 4. An example of the training process is shown as follow:

```
Epoch: 0	LR: 0.001000	Valid Loss: 0.0624	Valid MAE: 0.1907
Epoch: 1	LR: 0.001000	Valid Loss: 0.0655	Valid MAE: 0.1979
Epoch: 2	LR: 0.001000	Valid Loss: 0.0553	Valid MAE: 0.1751
Epoch: 3	LR: 0.001000	Valid Loss: 0.0506	Valid MAE: 0.1675
Epoch: 4	LR: 0.001000	Valid Loss: 0.0482	Valid MAE: 0.1566
Epoch: 5	LR: 0.001000	Valid Loss: 0.0456	Valid MAE: 0.1534
Epoch: 6	LR: 0.001000	Valid Loss: 0.0462	Valid MAE: 0.1559
Epoch: 7	LR: 0.001000	Valid Loss: 0.0447	Valid MAE: 0.1524
Epoch: 8	LR: 0.001000	Valid Loss: 0.0443	Valid MAE: 0.1495
Epoch: 9	LR: 0.001000	Valid Loss: 0.0447	Valid MAE: 0.1521
Epoch: 10	LR: 0.001000	Valid Loss: 0.0437	Valid MAE: 0.1486
Epoch: 11	LR: 0.001000	Valid Loss: 0.0442	Valid MAE: 0.1481
Epoch: 12	LR: 0.001000	Valid Loss: 0.0435	Valid MAE: 0.1481
Epoch: 13	LR: 0.001000	Valid Loss: 0.0451	Valid MAE: 0.1501
Epoch: 14	LR: 0.001000	Valid Loss: 0.0433	Valid MAE: 0.1482
Epoch: 15	LR: 0.001000	Valid Loss: 0.0433	Valid MAE: 0.1486
Epoch: 16	LR: 0.001000	Valid Loss: 0.0436	Valid MAE: 0.1489
Epoch: 17	LR: 0.001000	Valid Loss: 0.0435	Valid MAE: 0.1477
Epoch: 18	LR: 0.000200	Valid Loss: 0.0426	Valid MAE: 0.1447
Epoch: 19	LR: 0.000200	Valid Loss: 0.0425	Valid MAE: 0.1461
Epoch: 20	LR: 0.000200	Valid Loss: 0.0423	Valid MAE: 0.1453
Epoch: 21	LR: 0.000200	Valid Loss: 0.0423	Valid MAE: 0.1458
Epoch: 22	LR: 0.000200	Valid Loss: 0.0423	Valid MAE: 0.1448
```

### 3. Inference
We firstly normalize the challenge data for task 1 as the same with training data. After that, the predicted anomalous exponent of a trajectory is determined by the following rule:
* If the original length of this trajectory belongs to 43 specific lengths, the trajectory data remains the same. The model output is exactly the predicted exponent of this trajectory.
* Otherwise, a new length of this trajectory is set as **the closest smaller specific length** in the table. For instance, the new length of a trajectory with a length 49 should be 45. The trajectory data is subsequently transformed into 2 sequences. For clarity, we set the trajectory data as [x_1, x_2, ..., x_T] where T is the original length. We denote T_n as the new length. Note that T_n < T, the two sequences are [x_1, x_2, ..., x_{T_n}] and [x_{T-T_n+1}, x_{T-T_n+2}, ..., x_T] respectively. Such two sequences are both used for inference, with model outputs alpha_1 and alpha_2. The predicted exponent a of this trajectory is given by alpha=(alpha_1+alpha_2)/2.

### 4. K-Fold & Post-Processing
To improve the model performance, K-Fold cross validation is utilized where *K* = 5 in this work. However, due to the time limit of this competition, we only calculate 3 folds. Therefore, totally 43 * 3 = 129 models are used for our final submission.

On the other hand, we generate an external validation dataset containing 100000 1D trajectories where the lengths are uniformly distributed. We find that multiply the model outputs by 1.011 leads to a relatively lowest MAE on this validation dataset. Therefore, the predicted results for challenge data are also multiplied by 1.011. Finally, since the exponent is in [0.05,2], the final results are clipped to ensure reasonable predictions.

The codes for inference and post-processing are in the file `inference-1d.py`. Before running the code, make two folders `data` and `output`, then put the challenge data 'task1.txt' into folder 'data'. The result will be generated in folder `output`.

### 5. 2D & 3D Task
The methods for predicting anomalous exponents of 2D and 3D trajectories are both based on models for 1D trajectories. We separate the dimensions of trajectories and treat the data of each dimension as 1D trajectories. Using the same method for 1D trajectories, we can get predicted exponents alpha_x, alpha_y, and alpha_z for *x*, *y*, and *z* dimensions respectively. The final results are alpha_2D = (alpha_x+alpha_y)/2 for 2D trajectories, and alpha_3D = (alpha_x+alpha_y+alpha_z)/3 for 3D trajectories.

### 6. System Environment
* OS: Ubuntu 16.04
* GPU: RTX 2080Ti
* Python==3.7.4
* PyTorch==1.6.0
* cuda==10.2
* cuDNN==7.6.5

### 7. References
[1] G. Muñoz-Gil, G. Volpe, M. A. Garcia-March, R. Metzler, M. Lewenstein, and C. Manzo. [AnDi: The Anomalous Diffusion Challenge](https://arxiv.org/abs/2003.12036). arXiv: 2003.12036.

[2] S. Bo, F. Schmidt, R. Eichhorn, and G. Volpe. [Measurement of Anomalous Diffusion Using Recurrent Neural Networks](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.010102). Phys. Rev. E 100, 010102(R) (2019).
