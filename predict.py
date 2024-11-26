import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

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


# Load the trained model
save_dir = 'saved_model'
model = SimpleModel(input_size=3, output_size=6)  # Initialize the model with the correct input and output size
model.load_state_dict(torch.load('./saved_model_1/saved_model_77.pth'))  # 模型参数
model.eval()

# 输入数据
new_data = torch.FloatTensor([[32, 2.063, 2]])

# scaler parameters
scaler_params = torch.load('./scaler_params_1.pth')
scaler = StandardScaler()

# Set scaler parameters based on training data
scaler.mean_ = scaler_params['mean_']
scaler.scale_ = scaler_params['scale_']

# Scale new data using the loaded scaler
scaled_new_data = scaler.transform(new_data.numpy())

# Convert the scaled data to a PyTorch Tensor
scaled_new_data_tensor = torch.FloatTensor(scaled_new_data)

# Make predictions
prediction = model(scaled_new_data_tensor)

# 将每个元素转换为字符串并使用 ', ' 连接
rounded_prediction = ', '.join([f'{item:.2f}' for item in prediction.flatten().detach().numpy()])
print(f'Prediction: [{rounded_prediction}]')
