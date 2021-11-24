import datetime
from utils import *
import torch
import torch.nn as nn
from LSTNet import LSTNet
from sklearn import metrics
import matplotlib.pyplot as plt
from visdom import Visdom


args = Args()
dataset_train = PollutionDataset(window=args.window, horizon=args.horizon, skip=args.skip, train=True)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True, num_workers=0)

dataset_test = PollutionDataset(window=args.window, horizon=args.horizon, skip=args.skip, train=False)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=64, shuffle=True, num_workers=0)

viz = Visdom(port=8097)
viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss', legend=['loss']))

total_samples_train = len(dataset_train)
n_iterations = total_samples_train // 64
print(total_samples_train, n_iterations)

device = torch.device('cuda:0' if args.cuda else 'cpu')
print(f'device: {device}')

# model
model = LSTNet(args, input_dim=7, output_dim=1)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

flog = open(args.log, "a")
now = str(datetime.datetime.now())
print(now)
flog.write(now + "\n")
flog.flush()

# training loop
num_epochs = 2
for epoch in range(num_epochs):
    for i, (inputs_test, targets_test) in enumerate(dataloader_train):
        targets_test = targets_test.to(device)

        # forward
        y_pred = model(inputs_test).to(device)
        loss = criterion(y_pred, targets_test)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        viz.line([loss.item()], [i+(epoch*n_iterations)], win='train_loss', update='append')

        if (i+1) % 10 == 0:
            s = f'Epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, loss = {loss.item():.4f}'
            print(s)
            flog.write(s + "\n")
            flog.flush()

# test
with torch.no_grad():
    scaler_low = dataset_test.scaler[0, 0]
    scaler_high = dataset_test.scaler[0, 1]
    scaler_delta = scaler_high - scaler_low
    rmse = 0.0
    mape = 0.0


    def get_mape(y_pred, y_true):
        # Delete the zeros
        y_pred = np.delete(y_pred, np.where(y_true == 0))
        y_true = np.delete(y_true, np.where(y_true == 0))
        return metrics.mean_absolute_percentage_error(y_true, y_pred)

    for i, (inputs_test, targets_test) in enumerate(dataloader_test):
        targets_test = targets_test.to(device)

        y_pred_test = model(inputs_test).to(device)

        targets_test_real = (targets_test * scaler_delta + scaler_low).numpy()
        y_pred_test_real = (y_pred_test * scaler_delta + scaler_low).numpy()

        rmse += np.sqrt(metrics.mean_squared_error(targets_test_real, y_pred_test_real))
        mape += get_mape(y_pred_test_real, targets_test_real)

    s = f'RMSE {rmse/i:.4f}, MAPE {mape/i:.4f}'
    print(s)
    flog.write(s + "\n")
    flog.flush()
    flog.close()

    x_plot = dataset_test.x[-50:, :, :]
    y_pred_plot = model(x_plot).to(device)
    y_pred_plot_real = (y_pred_plot * scaler_delta + scaler_low).detach().numpy()
    targets_plot = dataset_test.y[-50:, :]
    targets_plot_real = (targets_plot * scaler_delta + scaler_low).numpy()
    plt.plot(y_pred_plot_real)
    plt.plot(targets_plot_real)
    plt.legend(['pred', 'target'])
    plt.show()
