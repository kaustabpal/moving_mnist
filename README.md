# Pytorch Lightning Template Repo

```
mkdir data
cd data
wget https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
export DATA_RAW=/Users/kaustabpal/work/moving_mnist/data/mnist_test_seq.npy
```
# Conv-LSTM
Conv-LSTM with peep and without peep connection has been implemented.

## Many to One
### With Binary Cross Entropy Loss
**Prediction**

![Many to One prediction](img/conv_lstm_m2o_pred.jpg)

**Loss curve**
<img src="img/conv_lstm_val_loss.png" alt="Many to One loss curve" width="75"/>
<!-- ![Many to One loss curve](img/conv_lstm_val_loss.png) -->


