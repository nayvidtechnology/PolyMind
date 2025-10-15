from pathlib import Path
from azure.storage.blob import BlobServiceClient  # type: ignore
from .base import Storage


class AzureBlob(Storage):
    def __init__(self, connection_string: str, container: str):
        self.container = container
        self.client = BlobServiceClient.from_connection_string(connection_string)

    def save(self, src: Path, dest: str) -> None:
        blob = self.client.get_blob_client(self.container, dest)
        with open(src, "rb") as f:
            blob.upload_blob(f, overwrite=True)

    def load(self, src: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = self.client.get_blob_client(self.container, src)
        with open(dest, "wb") as f:
            data = blob.download_blob()
            f.write(data.readall())
