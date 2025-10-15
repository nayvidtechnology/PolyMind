from pathlib import Path
import boto3
from .base import Storage


class S3(Storage):
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.s3 = boto3.client("s3")

    def save(self, src: Path, dest: str) -> None:
        self.s3.upload_file(Filename=str(src), Bucket=self.bucket, Key=dest)

    def load(self, src: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(Bucket=self.bucket, Key=src, Filename=str(dest))
