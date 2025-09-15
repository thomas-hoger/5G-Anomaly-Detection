import pyshark
from kedro.io import AbstractDataset


class PySharkPcapDataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self):
        return pyshark.FileCapture(
            self._filepath,
            keep_packets=False
        )

    def _save(self, data) -> None:
        raise NotImplementedError("PyShark dataset is read-only")

    def _describe(self):
        return dict(filepath=self._filepath)
