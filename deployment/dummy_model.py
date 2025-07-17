import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import json

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
        out = self.adaptive_pool(out)  # Shape: (batch_size, 64, 1, 1)
        out = torch.flatten(out, 1)  # Shape: (batch_size, 64)
        return out


class CNNLSTM(nn.Module):
    def __init__(self, in_channels, seq_length, lstm_hidden_size=256, num_lstm_layers=2, pred_len=1):
        super().__init__()
        self.seq_length = seq_length
        self.cnn = CNN(in_channels=in_channels, num_residual_units=1)
        self.lstm = nn.LSTM(
            input_size=64,  # Matches the output size of AirRes
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden_size, pred_len)

    def forward(self, x):
        batch_size, _, channels, height, width = x.shape
        x = x.view(batch_size * self.seq_length, channels, height, width)
        cnn_out = self.cnn(x)  # Shape: (batch_size * seq_length, 64)
        lstm_in = cnn_out.view(batch_size, self.seq_length, -1)  # Shape: (batch_size, seq_length, 64)
        lstm_out, _ = self.lstm(lstm_in)  # Shape: (batch_size, seq_length, lstm_hidden_size)
        last_time_step_out = lstm_out[:, -1, :]
        prediction = self.fc(last_time_step_out)

        return prediction


# Thresholds for converting added health risk to AQHI bands
AQHI_THRESHOLDS = np.array([1.87, 3.73, 5.60, 7.46, 9.33, 11.20, 12.81, 14.94, 17.08, 19.21])


def load_images(path: str) -> np.ndarray:
    return np.load(path)


def load_scalers(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_station_names(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    return df["station"].tolist()


def transform_with_channel_scalers(X: np.ndarray, scalers) -> np.ndarray:
    T, C, S, H, W = X.shape
    Xs = np.zeros_like(X, dtype=np.float32)
    for c, sc in enumerate(scalers):
        vals = X[:, c].reshape(-1, 1)
        Xs[:, c] = sc.transform(vals).reshape(T, S, H, W)
    return Xs


def ar_to_aqhi(ar: np.ndarray) -> np.ndarray:
    # Convert %AR to AQHI index (1-10)
    bin_index = np.sum(ar[..., np.newaxis] > AQHI_THRESHOLDS, axis=-1) + 1
    return np.where(bin_index <= 10, bin_index, 10)


def prepare_input(images_path: str, scalers_path: str, stations_csv: str, patch_size: int) -> np.ndarray:
    images = load_images(images_path)
    patches = images_to_patches(images, stations_csv, patch_size)
    scalers = load_scalers(scalers_path)
    scaled = transform_with_channel_scalers(patches, scalers)
    # Reorder to (batch=stations, seq, channels, H, W)
    return scaled.transpose(2, 0, 1, 3, 4)


def load_model(path: str, device) -> nn.Module:
    model = torch.load(path, weights_only=False).to(device)
    model.eval()
    return model


def predict(model: nn.Module, X_s: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        inp = torch.tensor(X_s).to(device)
        ar = model(inp).cpu().numpy()
    return ar


def format_output(aqhi: np.ndarray, station_names: list[str], start_hour: int = 1) -> list[dict]:
    hours = aqhi.shape[1]
    times = [f"{h:02d}:00" for h in range(start_hour, start_hour + hours)]
    output = []
    for si, station in enumerate(station_names[: aqhi.shape[0]]):
        for hi, t in enumerate(times):
            output.append({"date": None, "time": t, "station": station, "aqi": int(aqhi[si, hi]), "pm2_5": None})
    return output


def main():
    images_path = "./data/past48h_tensor.npy"
    scalers_path = "./data/x_scalers.pkl"
    stations_csv = "./data/stations_epd_idx.csv"
    patch_size = 15
    model_path = "./data/dummy.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_s = prepare_input(images_path, scalers_path, stations_csv, patch_size)
    model = load_model(model_path, device)
    ar = predict(model, X_s, device)
    aqhi = ar_to_aqhi(ar)
    station_names = load_station_names(stations_csv)
    output = format_output(aqhi, station_names)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
