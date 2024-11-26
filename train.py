import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


# 读取Excel表格
excel_file = 'C:/Users/Administrator.DESKTOP-FSP7GI4/Desktop/xpp+colorimetric+patches.xlsx'
df = pd.read_excel(excel_file)
# 第一组输入输出
input_columns = ['LCST (oC)', 'Adhesion (kPa)', 'Stability (days)']
target_columns =['X1', 'X3', 'X4', 'X5', 'X6 (Glycerin) g', 'X8 Length (mm)']

# 第二组输入输出
# input_columns = ['Elongation (%)', 'Stress（Mpa）']
# target_columns =['X1', 'X3', 'X4', 'X5', 'X6 (Glycerin) g', 'X10 Length (mm)']

# 创建输入特征 X 和目标变量 y
df.fillna(0, inplace=True)
X = df[input_columns].values
y = df[target_columns].values

X = X.astype('float64')
y = torch.from_numpy(y).float()
# # 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler parameters during training
scaler_params = {'mean_': scaler.mean_, 'scale_': scaler.scale_}
torch.save(scaler_params, './scaler_params_1.pth')

# 划分数据集为训练集和测试集，保持类别分布的均匀性
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 转换为PyTorch的Tensor
# X_train = torch.FloatTensor(X_train)
# y_train = torch.FloatTensor(y_train)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)



#2构建模型
class SimpleModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

input_size = X_train.shape[1]
output_size = len(target_columns)
model = SimpleModel(input_size, output_size)

#3.定义损失函数和优化器
criterion =nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4.训练模型
epochs =10000

for epoch in range(epochs):
    optimizer.zero_grad()

    # 将 NumPy 数组转换为 PyTorch Tensor
    X_tensor = torch.from_numpy(X_train)
    X_tensor = X_tensor.float()
    # 模型前向传播
    outputs = model(X_tensor)

    # 计算损失
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    loss.backward()
    optimizer.step()


    if epoch % 100 == 0:

        # 在测试集上评估模型
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)

        # 测试代码
        new_data = torch.FloatTensor([[32,2.063,2]])  # 替换为新数据的实际值
        scaled_new_data = torch.FloatTensor(scaler.transform(new_data.numpy()))
        prediction = model(scaled_new_data)

        # 将每个元素转换为字符串并使用 ', ' 连接
        rounded_prediction = ', '.join([f'{item:.2f}' for item in prediction.flatten().detach().numpy()])
        # print(f'Prediction: [{rounded_prediction}]')

        new_data_2 = torch.FloatTensor([[42,4.403,40]])
        scaled_new_data_2 = torch.FloatTensor(scaler.transform(new_data_2.numpy()))
        prediction_2 = model(scaled_new_data_2)

        # 将每个元素转换为字符串并使用 ', ' 连接
        rounded_prediction_2 = ', '.join([f'{item:.2f}' for item in prediction_2.flatten().detach().numpy()])
        # print(f'Prediction: [{rounded_prediction_2}]')

        print(f'Epoch[{epoch}/{epochs}], Train Loss: {loss.item():.4f},Test Loss: {test_loss.item():.4f},'
              f'Prediction: [{rounded_prediction}],Prediction: [{rounded_prediction_2}]')
        if test_loss < 10:
            # 定义保存模型的目录
            save_dir = 'saved_model_1'

            # 创建目录（如果不存在）
            os.makedirs(save_dir, exist_ok=True)

            # 保存模型参数到指定目录
            model.eval()
            torch.save(model.state_dict(), save_dir + '/' + 'saved_model' + '_%d.pth' % (epoch/100))

model.eval()
torch.save(model.state_dict(), save_dir + '/' + 'saved_model' + '_%d.pth' % (epochs/100))


# 测试代码
new_data = torch.FloatTensor([[32,2.063,2]])  # 替换为新数据的实际值
scaled_new_data = torch.FloatTensor(scaler.transform(new_data.numpy()))
prediction = model(scaled_new_data)

# 将每个元素转换为字符串并使用 ', ' 连接
rounded_prediction = ', '.join([f'{item:.2f}' for item in prediction.flatten().detach().numpy()])
print(f'Prediction: [{rounded_prediction}]')

new_data_2 = torch.FloatTensor([[42,4.403,40]])
scaled_new_data_2 = torch.FloatTensor(scaler.transform(new_data_2.numpy()))
prediction_2 = model(scaled_new_data_2)

# 将每个元素转换为字符串并使用 ', ' 连接
rounded_prediction_2 = ', '.join([f'{item:.2f}' for item in prediction_2.flatten().detach().numpy()])
print(f'Prediction: [{rounded_prediction_2}]')