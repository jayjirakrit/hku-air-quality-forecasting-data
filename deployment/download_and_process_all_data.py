from deployment.download_scripts import (
    download_air_quality,
    download_meteorological_data,
    processing_air_quality,
    processing_meteorological_data,
)

def download_and_processed_all_data():
    try:
        download_meteorological_data.download()
        download_air_quality.download_all()
    except Exception as e:
        print(f"Download Failed: {e}")

    try:
        processing_meteorological_data.process_all()
        processing_air_quality.process_all()
    except Exception as e:
        print(f"Processing Failed: {e}")
