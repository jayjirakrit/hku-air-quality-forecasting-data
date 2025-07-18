import os
import pandas as pd


def process_pressure():
    data_name = "pressure"
    path = "./data/raw/" + data_name
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(path + "/" + f))
    df = pd.concat(dfs, ignore_index=True)
    df = df.infer_objects()
    df["Datetime"] = pd.to_datetime(df["Date time"], format="%Y%m%d%H%M")
    df = df.drop(columns=["Date time"])
    df["Datetime"] = df["Datetime"].dt.round("h")
    # Grouping for mean aggregation (non-wind case)
    df = df.groupby(["Datetime", "Automatic Weather Station"], as_index=False).mean()
    # Interpolation across time per station
    df = (
        df.groupby("Automatic Weather Station", as_index=False)
        .apply(lambda group: group.sort_values("Datetime").interpolate(method="linear", limit_direction="both"))
        .reset_index(drop=True)
    )
    # Deduplicate again in case of interpolation artifacts
    df = df.groupby(["Datetime", "Automatic Weather Station"], as_index=False).mean()
    df.to_csv(f"./data/{data_name}.csv", index=False)


def process_wind():
    data_name = "wind"
    path = "./data/raw/" + data_name
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(os.path.join(path, f)))
    df = pd.concat(dfs, ignore_index=True)
    df = df.infer_objects()
    # Clean up speed column
    df["10-Minute Mean Speed(km/hour)"] = df["10-Minute Mean Speed(km/hour)"].astype(str)
    df["10-Minute Mean Speed(km/hour)"] = df["10-Minute Mean Speed(km/hour)"].str.replace(r"N/A", "", regex=True)
    # Ensure correct types
    df = df.astype(
        dtype={
            "Date time": str,
            "Automatic Weather Station": str,
            "10-Minute Mean Wind Direction(Compass points)": str,
            "10-Minute Mean Speed(km/hour)": float,
            "10-Minute Maximum Gust(km/hour)": float,
        }
    )
    # Parse and round datetime
    df["Datetime"] = pd.to_datetime(df["Date time"], format="%Y%m%d%H%M")
    df["Datetime"] = df["Datetime"].dt.round("h")
    # Group by rounded Datetime and Station to avoid duplicates
    df = df.groupby(["Datetime", "Automatic Weather Station"], as_index=False).agg(
        {
            "10-Minute Mean Wind Direction(Compass points)": lambda x: x.mode().iloc[0] if not x.mode().empty else "",
            "10-Minute Mean Speed(km/hour)": "mean",
            "10-Minute Maximum Gust(km/hour)": "mean",
        }
    )
    # Interpolation per station
    df = df.sort_values(["Automatic Weather Station", "Datetime"])
    df = (
        df.groupby("Automatic Weather Station", as_index=False)
        .apply(lambda g: g.set_index("Datetime").interpolate(method="time", limit_direction="both").reset_index())
        .reset_index(drop=True)
    )
    # Save output
    df.to_csv(f"./data/{data_name}.csv", index=False)


def process_humidity():
    data_name = "humidity"
    path = "./data/raw/" + data_name
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(os.path.join(path, f)))
    df = pd.concat(dfs, ignore_index=True)
    df = df.infer_objects()
    df["Datetime"] = pd.to_datetime(df["Date time"], format="%Y%m%d%H%M")
    df["Datetime"] = df["Datetime"].dt.round("h")
    df["Relative Humidity(percent)"] = pd.to_numeric(df["Relative Humidity(percent)"], errors="coerce")
    df = df.sort_values(["Automatic Weather Station", "Datetime"])

    df = (
        df.groupby("Automatic Weather Station", as_index=False)
        .apply(lambda g: g.set_index("Datetime").interpolate(method="time", limit_direction="both").reset_index())
        .reset_index(drop=True)
    )
    df = df.groupby(["Datetime", "Automatic Weather Station"], as_index=False).mean()

    df.drop(columns=["Date time"], inplace=True)
    df.rename(columns={"Datetime": "Hour", "Automatic Weather Station": "Station"}, inplace=True)
    df.to_csv(f"./data/{data_name}.csv", index=False)


def process_temperature():
    data_name = "temperature"
    path = "./data/raw/" + data_name
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(os.path.join(path, f)))
    df = pd.concat(dfs, ignore_index=True)
    df = df.infer_objects()
    df["Datetime"] = pd.to_datetime(df["Date time"], format="%Y%m%d%H%M")
    df["Datetime"] = df["Datetime"].dt.round("h")

    df["Maximum Air Temperature Since Midnight(degree Celsius)"] = pd.to_numeric(
        df["Maximum Air Temperature Since Midnight(degree Celsius)"], errors="coerce"
    )
    df["Minimum Air Temperature Since Midnight(degree Celsius)"] = pd.to_numeric(
        df["Minimum Air Temperature Since Midnight(degree Celsius)"], errors="coerce"
    )
    df = df.sort_values(["Automatic Weather Station", "Datetime"])
    df = (
        df.groupby("Automatic Weather Station", as_index=False)
        .apply(lambda g: g.set_index("Datetime").interpolate(method="time", limit_direction="both").reset_index())
        .reset_index(drop=True)
    )
    df = df.groupby(["Datetime", "Automatic Weather Station"], as_index=False).mean()
    df.rename(
        columns={
            "Datetime": "Hour",
            "Automatic Weather Station": "Station",
            "Maximum Air Temperature Since Midnight(degree Celsius)": "Average Max Temp",
            "Minimum Air Temperature Since Midnight(degree Celsius)": "Average Min Temp",
        },
        inplace=True,
    )

    df.drop(columns=["Date time"], inplace=True)
    df.to_csv(f"./data/{data_name}.csv", index=False)


def process_all():
    process_humidity()
    process_pressure()
    process_temperature()
    process_wind()


if __name__ == "__main__":
    process_all()
