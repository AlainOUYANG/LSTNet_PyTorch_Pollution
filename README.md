# LSTNet simple implementation in PyTorch

A simple implementation of LSTNet and an out-of-box way about:

- Modeling
- Forecasting
- Evaluating (RMSE and MAPE)

Note:

1. Inspired by [PatientEz/LSTNet_Keras](https://github.com/PatientEz/LSTNet_Keras). Used the same [`pollution` dataset](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data).
2. The original implementation of [laiguokun/LSTNet](https://github.com/laiguokun/LSTNet) used a normalization over the whole dataset, which is not standard. I normalized the training and test set separately.

Requirements:

```
matplotlib~=3.4.2
scikit-learn~=0.24.2
visdom~=0.1.8.9
numpy~=1.20.3
pytorch~=1.9.1
```

References:

- [PatientEz/LSTNet_Keras](https://github.com/PatientEz/LSTNet_Keras)
- [laiguokun/LSTNet](https://github.com/laiguokun/LSTNet)