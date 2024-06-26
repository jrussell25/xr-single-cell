from itertools import product
import numpy as np
import xarray as xr
import pandas as pd
from skimage.measure import regionprops_table

__all__ = ["SingleCellAccessor"]


@xr.register_dataset_accessor("single_cell")
class SingleCellAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    @staticmethod
    def _get_meta(properties: tuple[str, ...]) -> pd.DataFrame:
        dummy_regionprops = pd.DataFrame(
            regionprops_table(
                np.ones(1, 1, dtype="u2"), np.ones(1, 1), properties=properties
            )
        )
        return dummy_regionprops.drop(0)

    @staticmethod
    def _get_loop_sizes(
        labels: xr.DataArray, core_dims: tuple[str, str] | None = None
    ) -> dict[str, int]:
        if core_dims is None:
            loop_sizes = {
                labels.dims[i]: labels.sizes[labels.dims[i]]
                for i in range(labels.ndim - 2)
            }
        else:
            loop_sizes = {d: s for d, s in labels.sizes.items() if d not in core_dims}

        return loop_sizes

    def regionprops(
        self,
        labels_name: str,
        intensity: str | None = None,
        properties: tuple[str, ...] = ("label", "bbox"),
        core_dims: tuple[str, str] | None = None,
    ) -> pd.DataFrame:
        loop_sizes = self._get_loop_sizes(self._obj[labels_name], core_dims)
        all_props = []

        for dims in product(*(range(v) for v in loop_sizes.values())):
            idxr = dict(zip(loop_sizes.keys(), dims))
            labels_arr = self._obj[labels_name].isel(idxr).data

            if intensity is not None:
                intensity_arr = self._obj[intensity].isel(idxr).data
            else:
                intensity_arr = None

            frame_props = pd.DataFrame(
                regionprops_table(
                    labels_arr, intensity_image=intensity_arr, properties=properties
                )
            )

            # with redundant cell labels keep track of the specific image
            # in the original dataset from which each object originated
            for col, val in zip(loop_sizes.keys(), dims):
                frame_props[col] = val

            all_props.append(frame_props)

        return pd.concat(all_props, ignore_index=True)
