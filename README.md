# Anomalous HNU in AnDi Challenge

Summary of our LSTM-based methods for task 1 (**inference of the anomalous diffusion exponent**) in AnDi Challenge.

## Zihan Huang

*School of Physics and Electronics, Hunan University, Changsha, China*

### Data Generation
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

### Model Training
We use a **LSTM**-based RNN model for this task. For 1D trajectories, the input dimension is set as 1. The number of features in the hidden state is 64, and the number of recurrent LSTM layers is 3. A fully-connected layer with a dropout rate *p* = 0.2 and one output node is added at the end of model. For this regression task, no activation function is applied on the output layer. The pytorch version of our model is as follow:

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
```

Models for each specific length are trained separately. 80% of training data is used for training, while the other is used for validation. We implement the LSTM-based model by PyTorch 1.6.0 on RTX 2080Ti. The model is trained by back propagation method with a batch size 512, where loss function is the mean squared error (MSE). The optimizer is Adam with a learning rate *l* = 0.001. The learning rate is changed as *l* -> *l*/5 if the valid loss doesn't decrease after 2 epochs. When the number of such changes exceeds 1, the training process is early stopped to save time and avoid overfitting. The best model for a specific length is determined by the mean absolute error (MAE) of validation set. We choose the epoch with the lowest valid MAE as the best model.

The codes for training are in the file `train_lstm.py`, where the usage is:
```
python train_lstm.py --l 100 --f 0
```
`--l` denotes the specific length and `--f` denotes the fold ranging from 0 to 4.

### Inference
We firstly normalize the challenge data for task 1 as the same with training data. After that, the predicted anomalous exponent of a trajectory is determined by the following rule:
* If the original length of this trajectory belongs to 43 specific lengths, the trajectory data remains the same. The model output is exactly the predicted exponent of this trajectory.
* Otherwise, a new length of this trajectory is set as the closest smaller specific length in the table. For instance, the new length of a trajectory with a length 49 should be 45. The trajectory data is subsequently transformed into 2 sequences. For clarity, we set the trajectory data as [x1, x2, ..., xT] where T is the original length. We denote Tn as the new length. Note that Tn < T, the two sequences are [x1, x2, ..., xTn] and [x_{T-Tn+1}, x_{T-Tn+2}, ..., xT] respectively. Such two sequences are both used for inference, with model outputs a1 and a2. The predicted exponent a of this trajectory is given by a=(a1+a2)/2.

### System Environment
* OS: Ubuntu 16.04
* GPU: RTX 2080Ti
* Python==3.7.4
* PyTorch==1.6.0
* cuda==10.2
* cuDNN==7.6.5
