import os
import requests
import zipfile
import io
import pandas as pd
from datetime import date
from dateutil.rrule import rrule, MONTHLY
from tqdm import tqdm


def daterange(start_date, end_date):
    return rrule(MONTHLY, dtstart=start_date, until=end_date)


start_date = date(2024, 1, 1)
end_date = date(2025, 2, 1)

date_iter = daterange(start_date, end_date)

# Create output directory
base_output_folder = "data/andy/"

# Base url
baseurl = "https://app.data.gov.hk/v1/historical-archive/get-file?url="
types_of_data = ["latest_1min_pressure", "latest_10min_wind", "latest_1min_grass", "latest_1min_solar"]
# types_of_data = ['latest_1min_pressure']

# Loop through all the different data
for data in types_of_data:

    data_name = data.split("_")[-1]

    output_folder = base_output_folder + data_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for single_date in tqdm(date_iter, total=len(list(date_iter)), desc=f"{data_name}:"):
        single_date = single_date.strftime("%Y%m%d")

        url = (
            baseurl
            + f"https%3A%2F%2Fdata.weather.gov.hk%2FweatherAPI%2Fhko_data%2Fregional-weather%2F{data}.csv&time={single_date}"
        )

        # Download the file
        response = requests.get(url)

        # List to store all dataframes
        dfs = []

        # Create a ZipFile object from the downloaded content
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Process each CSV file in the zip
            for file_info in zip_ref.filelist:
                # Skip directories, PDFs, and non-CSV files
                if (
                    file_info.filename.endswith("/")
                    or file_info.filename.lower().endswith(".pdf")
                    or not file_info.filename.lower().endswith(".csv")
                ):
                    continue

                # Read CSV content
                content = zip_ref.read(file_info.filename)
                df = pd.read_csv(io.BytesIO(content))
                dfs.append(df)

        # Concatenate all dataframes
        df = pd.concat(dfs, ignore_index=True)
        # Sort by datetime
        df = df.sort_values("Date time")

        # Save the combined CSV
        output_file = os.path.join(output_folder, f"{data_name}_combined_{single_date}.csv")
        df.to_csv(output_file, index=False)
