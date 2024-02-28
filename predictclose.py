import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import time
import math
from sklearn.metrics import mean_squared_error

price = pd.read_csv('stock_data.csv')[['open','high','low','close','vol','amount']]

# 进行不同的数据缩放，将数据缩放到0和1之间
scaler=dict()
for c in price.columns:
    scaler[c] = MinMaxScaler(feature_range=(0, 1))
    price[c] = scaler[c].fit_transform(price[c].values.reshape(-1, 1))

# lookback表示观察的跨度
def split_data(stock, lookback):
	# 将stock转化为ndarray类型
    data_raw = stock
    data = []
    for index in range(len(data_raw) - lookback +1):
        data.append(data_raw[index: index + lookback])
    data = np.array(data)
    # 进行训练集、测试集划分
    train_set_size = 525 - lookback +1

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, 3]
    x_test = data[train_set_size:, :-1, :]
    y_test = data[train_set_size:, -1, 3]
    return [x_train, y_train, x_test, y_test]

lookback = 2
x_train, y_train, x_test, y_test = split_data(price, lookback)

x_train = torch.from_numpy(x_train).type(torch.Tensor).cuda()
x_test = torch.from_numpy(x_test).type(torch.Tensor).cuda()
y_train=y_train[:,np.newaxis]
y_test=y_test[:,]
# 真实的数据
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor).cuda()
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor).cuda()



input_dim = 6
hidden_dim = 128
num_layers = 3
output_dim = 1
num_epochs = 3000
learningrate = 0.001
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:,-1,:])
        return out

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).cuda()
criterion = torch.nn.MSELoss().cuda()
optimiser = torch.optim.Adam(model.parameters(), lr=learningrate)
start_time = time.time()
lstm = []

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1.0
hist=[]
for t in range(num_epochs):
    model.train()
    y_train_pred = model(x_train).cuda()
    loss = criterion(y_train_pred, y_train_lstm)
    if (t+1)%100==0:
        print("Epoch ", t+1, "MSE: ", loss.item())
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    if (t+1)%100==0:
        model.eval()
        y_test_pred = model(x_test)
        y_test_pred = scaler['close'].inverse_transform(y_test_pred.cpu().detach().numpy().reshape(-1, 1))
        y_test = scaler['close'].inverse_transform(y_test_lstm.cpu().detach().numpy().reshape(-1, 1))
        mse=mean_squared_error(y_test, y_test_pred)
        hist.append(mse)
        if  mse < best_loss:
            best_loss = mse
            best_model_wts = copy.deepcopy(model.state_dict())    

training_time = time.time() - start_time
print("Training time: {}".format(training_time))
import seaborn as sns
sns.set_style("darkgrid")

fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.subplot(1, 1, 1)
ax = sns.lineplot(data=hist, color='royalblue')
ax.set_xlabel("100Epoch", size = 14)
ax.set_ylabel("MSELoss", size = 14)
ax.set_title("TestLoss", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)
plt.show()

model.load_state_dict(best_model_wts)
## 保存模型
torch.save(model, str(round(best_loss,5))+'model.pth')

## 读取模型
##model = torch.load('model_name.pth')

# make predictions
model.eval()
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler['close'].inverse_transform(y_train_pred.cpu().detach().numpy().reshape(-1, 1))
y_train = scaler['close'].inverse_transform(y_train_lstm.cpu().detach().numpy().reshape(-1, 1))
y_test_pred = scaler['close'].inverse_transform(y_test_pred.cpu().detach().numpy().reshape(-1, 1))
y_test = scaler['close'].inverse_transform(y_test_lstm.cpu().detach().numpy().reshape(-1, 1))

# calculate root mean squared error
trainScore = mean_squared_error(y_train, y_train_pred)
print('Train Score: %.2f MSE' % (trainScore))
testScore = mean_squared_error(y_test, y_test_pred)
print('Test Score: %.2f MSE' % (testScore))
lstm.append(trainScore)
lstm.append(testScore)
lstm.append(training_time)
# shift train predictions for plotting
trainPredictPlot = np.empty_like(price['close'])[:,np.newaxis]
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback-1:len(y_train_pred)+lookback-1, :] = y_train_pred

# shift test predictions for plotting
testPredictPlot = np.empty_like(price['close'])[:,np.newaxis]
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred)+lookback-1:len(price), :] = y_test_pred

original = scaler['close'].inverse_transform(price['close'].values.reshape(-1,1))

predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
predictions = np.append(predictions, original, axis=1)
result = pd.DataFrame(predictions)
import plotly.express as px
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                    mode='lines',
                    name='Train prediction')))
fig.add_trace(go.Scatter(x=result.index, y=result[1],
                    mode='lines',
                    name='Test prediction'))
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                    mode='lines',
                    name='Actual Value')))
fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Close',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

)



annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Results (LSTM)',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()


