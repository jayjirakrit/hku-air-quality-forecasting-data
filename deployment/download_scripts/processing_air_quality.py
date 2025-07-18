import pandas as pd


def process_env():
    df = pd.read_csv("./data/raw/air_quality_env.csv")
    df["report_datetime"] = (
        pd.to_datetime(
            df["datetime"], format="%a, %d %b %Y %H:%M:%S %z", utc=True  # e.g. "Mon, 14 Jul 2025 00:00:00 +0800"
        )
        .dt.tz_localize(None)
        .dt.round("h")
    )
    df = df.drop(columns=["datetime"])
    df = df.groupby(["report_datetime", "station"], as_index=False).mean()
    df = (
        df.groupby("station", as_index=False)
        .apply(
            lambda g: g.set_index("report_datetime").interpolate(method="time", limit_direction="both").reset_index()
        )
        .reset_index(drop=True)
    )
    df = df.groupby(["report_datetime", "station"], as_index=False).mean()
    df = df.rename(
        columns={"NO2": "no2", "O3": "o3", "SO2": "so2", "CO": "co", "PM10": "rsp", "PM2.5": "fsp", "NO": "no"}
    )
    df["no"] = df["no2"]
    df = df.drop(columns=["co"])
    df.to_csv(f"./data/air_quality_env.csv", index=False)


def process_idx():
    df = pd.read_csv("./data/raw/air_quality_idx.csv")
    df["report_datetime"] = pd.to_datetime(df["DateTime"], utc=True)
    df["report_datetime"] = df["report_datetime"].dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None)
    df["report_datetime"] = df["report_datetime"].dt.round("h")
    df = df.drop(columns=["DateTime"])
    df = df.groupby(["report_datetime", "StationName"], as_index=False).mean()
    df = (
        df.groupby("StationName", as_index=False)
        .apply(lambda g: g.sort_values("report_datetime").interpolate(method="linear", limit_direction="both"))
        .reset_index(drop=True)
    )
    df = df.groupby(["report_datetime", "StationName"], as_index=False).mean()
    df = df.rename(columns={"aqhi": "agi", "StationName": "station"})
    df.to_csv(f"./data/air_quality_idx.csv", index=False)


def process_all():
    process_env()
    process_idx()


if __name__ == "__main__":
    process_all()
