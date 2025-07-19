from google.cloud import storage


def upload_blob(bucket_name: str, source_file_name: str, destination_blob_name: str):
    """Uploads a file to the bucket."""
    try:
        print(f"Start upload {source_file_name} to {destination_blob_name}.")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name, timeout=360)
        print(f"File {source_file_name} uploaded to {destination_blob_name}.")
        return True
    except Exception as e:
        print(f"Error uploading file {destination_blob_name}: {e}")
        return False