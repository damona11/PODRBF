# src/podrbf/data/loader.py

import numpy as np
import zarr


class SnapshotLoader:
    """
    Loads snapshot data from zarr and returns X of shape (M x N).

    For multiple components, stacks vertically:

        X = [ X_u ]   shape (M x N)
            [ X_v ]   shape (M x N)
            [ X_w ]   shape (M x N)

    giving X of shape (n_comp * M x N).

    M = spatial points (flattened)
    N = snapshots
    """

    def __init__(self, path, fields, time_axis=-1):
        self.path      = path
        self.fields    = fields if isinstance(fields, list) else [fields]
        self.time_axis = time_axis

        # open zarr directly — no xarray
        store          = zarr.open(path, mode='r')
        arr            = store[self.fields[0]]

        # raw shape e.g. (512, 256, 200)
        self.raw_shape = arr.shape

        self.N         = arr.shape[time_axis]
        self.M         = int(np.prod(arr.shape) // self.N)
        self.n_comp    = len(self.fields)
        self.shape     = (self.n_comp * self.M, self.N)

    def get_chunk(self, start, end):
        """
        Returns X[start:end, :] of shape (n_comp*(end-start) x N).

        Reads directly from zarr, reshapes to (M x N),
        slices spatial rows start:end per component,
        stacks vertically.
        """
        store      = zarr.open(self.path, mode='r')
        n_rows     = end - start
        result     = np.zeros((self.n_comp * n_rows, self.N))

        for i, field in enumerate(self.fields):
            arr              = store[field][:]              # read raw array
            arr              = arr.reshape(self.M, self.N)  # flatten spatial to M
            row_start        = i * n_rows
            row_end          = row_start + n_rows
            result[row_start:row_end, :] = arr[start:end, :]

        return result