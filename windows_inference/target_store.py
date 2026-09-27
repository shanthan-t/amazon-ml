"""Read-only offset-addressed target lookup used by the frozen inference worker."""
import json
import mmap
from pathlib import Path

import numpy as np

from .retrieval import Retriever


class TargetStore:
    def __init__(self, path):
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise ValueError("Target store is not marked complete")
        self.offsets = np.load(path / "offsets.npy", mmap_mode="r")
        self.file = (path / "records.tsv").open("rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self.data) != manifest["record_bytes"] or len(self.offsets) != manifest["targets"] + 1:
            raise ValueError("Target store size/offset manifest mismatch")

    def get(self, idx):
        if not 0 <= idx < len(self.offsets) - 1:
            raise ValueError(f"Invalid target index: {idx}")
        start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
        entity_id, name, address, country, source = self.data[start:end - 1].decode("utf-8").split("\t")
        return [entity_id, name, address, country, int(source)]

    def close(self):
        self.data.close()
        self.file.close()


class MappedRetriever(Retriever):
    def __init__(self, index, numeric_index, address_index, target_store):
        super().__init__(index, numeric_index, address_index)
        self.store = TargetStore(target_store)

    def fetch(self, ids):
        return {idx: self.store.get(idx) for idx in sorted(ids)}

    def close(self):
        self.store.close()
        super().close()
