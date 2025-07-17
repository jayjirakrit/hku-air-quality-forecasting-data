import numpy as np
import pickle

import torch
import torch.nn as nn

from images_to_patches import images_to_patches


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        return nn.functional.relu(out)

class CNN(nn.Module):
    def __init__(self, in_channels, num_residual_units=4):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(64)
        
        layers = []
        for i in range(num_residual_units):
            layers.append(ResidualUnit(in_channels=64, out_channels=64))
            # if i < num_residual_units - 1:
            #     layers.append(nn.Conv2d(64, 64, kernel_size=1))
                
        self.residual_blocks = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = nn.functional.relu(self.initial_bn(self.initial_conv(x)))
        out = self.residual_blocks(out)
        out = self.adaptive_pool(out) # Shape: (batch_size, 64, 1, 1)
        out = torch.flatten(out, 1) # Shape: (batch_size, 64)
        return out
    
class CNNLSTM(nn.Module):
    def __init__(self, in_channels, seq_length, lstm_hidden_size=256, num_lstm_layers=2, pred_len=1):
        super().__init__()
        self.seq_length = seq_length
        self.cnn = CNN(in_channels=in_channels, num_residual_units=1)
        self.lstm = nn.LSTM(
            input_size=64, # Matches the output size of AirRes
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden_size, pred_len)

    def forward(self, x):
        batch_size, _, channels, height, width = x.shape
        x = x.view(batch_size * self.seq_length, channels, height, width)
        cnn_out = self.cnn(x) # Shape: (batch_size * seq_length, 64)
        lstm_in = cnn_out.view(batch_size, self.seq_length, -1) # Shape: (batch_size, seq_length, 64)
        lstm_out, _ = self.lstm(lstm_in) # Shape: (batch_size, seq_length, lstm_hidden_size)
        last_time_step_out = lstm_out[:, -1, :]
        prediction = self.fc(last_time_step_out)
        
        return prediction


def transform_with_channel_scalers(X, scalers):
    T, C, S, H, W = X.shape
    Xs = np.zeros_like(X, dtype=np.float32)
    for c, sc in enumerate(scalers):
        vals = X[:, c, :, :, :].reshape(-1, 1)
        Xs[:, c, :, :, :] = sc.transform(vals).reshape(T, S, H, W)
    return Xs

def ar_to_aqhi(ar):
    thresholds = np.array([1.87, 3.73, 5.60, 7.46, 9.33, 11.20, 12.81, 14.94, 17.08, 19.21])
    bin_index = np.sum(ar[..., np.newaxis] > thresholds, axis=-1) + 1
    out = np.where(bin_index <= 10, bin_index.astype(object), 10) # AQHI larger than 10 is still given 10

    return out

features = [
    "so2",
    "no",
    "no2",
    "rsp",
    "o3",
    "fsp",
    "aqi",
    "humidity",
    "max_temp",
    "min_temp",
    "pressure",
    "wind_direction",
    "wind_speed",
    "max_wind_speed",
    "season",
    "is_weekend",
]

# NOTE: Replace this with the actual image from past 48 hours
images = np.load('./data/past48h_tensor.npy')
X = images_to_patches(images, "./data/stations_epd_idx.csv", patch_size=15) # (48, 16, 17, 15, 15)

with open('./data/x_scalers.pkl', 'rb') as f:
    X_scalers = pickle.load(f)

X_s = transform_with_channel_scalers(X, X_scalers)
X_s = X_s.transpose(2, 0, 1, 3, 4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./data/dummy.pth"
cnn_lstm_model = torch.load(model_path, weights_only=False).to(device)
cnn_lstm_model.eval()
with torch.no_grad():
    added_health_risk = cnn_lstm_model(torch.tensor(X_s).to(device)).cpu().numpy()

aqhi = ar_to_aqhi(added_health_risk) # (Station, Prediction)
print(aqhi.shape)
print(aqhi)