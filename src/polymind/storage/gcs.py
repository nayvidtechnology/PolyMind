from pathlib import Path
from google.cloud import storage  # type: ignore
from .base import Storage


class GCS(Storage):
    def __init__(self, bucket: str):
        self.bucket_name = bucket
        self.client = storage.Client()

    def save(self, src: Path, dest: str) -> None:
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(dest)
        blob.upload_from_filename(filename=str(src))

    def load(self, src: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(src)
        blob.download_to_filename(filename=str(dest))
