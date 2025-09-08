from kedro.io import AbstractDataset
from scapy.utils import rdpcap, wrpcap


class ScapyPcapDataSet(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self):
        return rdpcap(self._filepath)  # si jamais tu veux relire avec scapy

    def _save(self, data) -> None:
        wrpcap(self._filepath, data)   # `data` = liste de paquets scapy

    def _describe(self):
        return dict(filepath=self._filepath)
