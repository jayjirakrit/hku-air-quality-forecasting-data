from contextlib import asynccontextmanager
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from tzlocal import get_localzone
from deployment.generate_image_daily import generate_image
import numpy as np
from deployment.download_and_process_all_data import download_and_processed_all_data
from lib.google_cloud import upload_blob
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
scheduler = AsyncIOScheduler(timezone=get_localzone())

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    await batch_generate_input_image_data()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
    
# Scheduler generate past 48 image data and upload to GCS
@scheduler.scheduled_job('cron', hour=0, minute=1)
async def batch_generate_input_image_data():
    save_path = os.getenv("IMAGE_SOURCE_PATH")
    # Prepare processed Data
    download_and_processed_all_data()
    # Generate past 48 hour image data
    image_filled = generate_image("./data/")
    np.save(save_path, image_filled)
    print("Generated tensor of shape", image_filled.shape)
    # Upload image data to GCS
    if os.path.exists(save_path):
        print("File exists. Proceeding to upload.")
        bucket_name = os.getenv("GBS_BUCKET_NAME")
        destination_blob_name = os.getenv("GBS_DESTINATION_FILE")
        source_file = save_path
        upload_blob(bucket_name, source_file, destination_blob_name)
    else:
        print(f"Error: File {save_path} does not exist. Skipping upload.")

