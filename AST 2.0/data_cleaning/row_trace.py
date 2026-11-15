"""
row_trace.py — which raw row each row of the cleaned table came from, carried in df.attrs["source_rows"].

pandas deep-copies df.attrs every time a column is read, so a plain list of 50,000 ints there made the whole pipeline
several times slower. This holder is immutable and says so to copy.deepcopy (it returns itself), so carrying it is free.
"""
import numpy as np


class SourceRows:
    __slots__ = ("values",)

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.int64)

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]

    def __repr__(self):
        return f"SourceRows({len(self.values)} rows)"
