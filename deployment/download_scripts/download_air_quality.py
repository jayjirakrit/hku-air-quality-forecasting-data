import requests
import pandas as pd
from datetime import date, timedelta
import xml.etree.ElementTree as ET

from tqdm import tqdm


def parse_feed(xml_bytes: bytes) -> list[dict]:
    """
    Parse one AQHI24HrReport XML payload and return a list of records (dicts).
    Each record contains:
      - StationName (str)
      - DateTime    (pd.Timestamp, UTC)
      - one column per numeric tag under <item> (e.g. aqhi, pm10, o3_8hr, etc.)
    """
    root = ET.fromstring(xml_bytes)
    records = []
    for item in root.findall("item"):
        record = {
            "StationName": item.findtext("StationName"),
            "DateTime": pd.to_datetime(item.findtext("DateTime"), utc=True),
        }
        for child in item:
            tag = child.tag
            if tag in {"type", "StationName", "DateTime"}:
                continue
            try:
                record[tag] = float(child.text)
            except (TypeError, ValueError):
                record[tag] = child.text
        records.append(record)
    return records


def parse_pollutant(xml_bytes):
    root = ET.fromstring(xml_bytes)
    records = []
    for elem in root.findall(".//PollutantConcentration"):
        station = elem.findtext("StationName")
        dt = elem.findtext("DateTime")
        rec = {"station": station, "datetime": dt}
        for pollutant in ["NO2", "O3", "SO2", "CO", "PM10", "PM2.5", "NO"]:
            txt = elem.findtext(pollutant)
            try:
                rec[pollutant] = float(txt) if txt not in (None, "-", "") else None
            except ValueError:
                rec[pollutant] = None
        records.append(rec)
    return records


def download_air_quality():
    start = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    end = (date.today() - timedelta(days=3)).strftime("%Y%m%d")

    list_url = f"https://api.data.gov.hk/v1/historical-archive/list-file-versions?url=https%3A%2F%2Fwww.aqhi.gov.hk%2Fepd%2Fddata%2Fhtml%2Fout%2F24pc_Eng.xml&start={end}&end={start}"
    resp = requests.get(list_url)
    resp.raise_for_status()
    data = resp.json()
    timestamps = data.get("timestamps")

    encoded_file_url = "https%3A%2F%2Fwww.aqhi.gov.hk%2Fepd%2Fddata%2Fhtml%2Fout%2F24pc_Eng.xml"
    get_url_base = "https://api.data.gov.hk/v1/historical-archive/get-file" f"?url={encoded_file_url}&time="

    all_records = []
    for t in tqdm(timestamps, desc=f"Air Quality"):
        resp = requests.get(get_url_base + t)
        resp.raise_for_status()
        recs = parse_pollutant(resp.content)
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df.to_csv("./data/raw/air_quality_env.csv", index=False)


def download_air_index():
    start = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    end = (date.today() - timedelta(days=3)).strftime("%Y%m%d")

    list_url = f"https://api.data.gov.hk/v1/historical-archive/list-file-versions?url=https%3A%2F%2Fwww.aqhi.gov.hk%2Fepd%2Fddata%2Fhtml%2Fout%2F24aqhi_Eng.xml&start={end}&end={start}"

    resp = requests.get(list_url)
    resp.raise_for_status()
    data = resp.json()
    timestamps = data.get("timestamps")

    encoded_file_url = "https%3A%2F%2Fwww.aqhi.gov.hk%2Fepd%2Fddata%2Fhtml%2Fout%2F24aqhi_Eng.xml"
    get_url_base = "https://api.data.gov.hk/v1/historical-archive/get-file" f"?url={encoded_file_url}&time="

    all_records = []
    for t in tqdm(timestamps, desc=f"Air Index"):
        resp = requests.get(get_url_base + t)
        resp.raise_for_status()
        recs = parse_feed(resp.content)
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df.to_csv("./data/raw/air_quality_idx.csv", index=False)


def download_all():
    download_air_quality()
    download_air_index()


if __name__ == "__main__":
    download_all()
