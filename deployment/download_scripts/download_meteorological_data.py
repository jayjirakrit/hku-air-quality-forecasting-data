import os
import requests
import zipfile
import io
import pandas as pd
from datetime import date, timedelta
from tqdm import tqdm
import shutil


def clear_output_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def download():
    # Get previous 48 hours date
    today = date.today()
    date_list = [today - timedelta(days=d) for d in range(1, 4)]

    # Create base output directory
    base_output_folder = "./data/raw/"
    try:
        clear_output_folder(base_output_folder)
    except Exception as e:
        pass

    # Base URL for the HK weather archive
    baseurl = "https://app.data.gov.hk/v1/historical-archive/get-file?url="
    types_of_data = [
        "latest_1min_pressure",
        "latest_10min_wind",
        "latest_1min_humidity",
        "latest_since_midnight_maxmin",
    ]

    for data in types_of_data:
        data_name = data.split("_")[-1] if data.split("_")[-1] != "maxmin" else "temperature"
        output_folder = os.path.join(base_output_folder, data_name)
        os.makedirs(output_folder, exist_ok=True)

        # Loop only over the two desired dates
        for single_date in tqdm(date_list, desc=f"{data_name}", unit="day"):
            # format as YYYYMMDD
            date_str = single_date.strftime("%Y%m%d")

            # construct URL (note the inner URL must be URL‑encoded)
            url = (
                baseurl
                + f"https%3A%2F%2Fdata.weather.gov.hk%2FweatherAPI%2Fhko_data%2Fregional-weather%2F{data}.csv"
                + f"&time={date_str}"
            )

            # Download and unzip
            try:
                response = requests.get(url)
                response.raise_for_status()  # fail early on HTTP errors
            except Exception as e:
                print(e)
                continue

            dfs = []
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.filename.lower().endswith(".csv"):
                        content = zip_ref.read(file_info.filename)
                        df = pd.read_csv(io.BytesIO(content))
                        dfs.append(df)

            # Combine, sort, save
            if dfs:
                df_all = pd.concat(dfs, ignore_index=True)
                df_all = df_all.sort_values("Date time")
                output_file = os.path.join(output_folder, f"{data_name}_combined_{date_str}.csv")
                df_all.to_csv(output_file, index=False)
            else:
                tqdm.write(f"No CSV files found for {data_name} on {date_str}")


if __name__ == "__main__":
    download()
