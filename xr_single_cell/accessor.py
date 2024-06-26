from itertools import product
import xarray as xr
import pandas as pd
from skimage.measure import regionprops_table

__all__ = ["SingleCellAccessor"]


@xr.register_dataset_accessor("single_cell")
class SingleCellAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    @staticmethod
    def _get_loop_sizes(
        labels: xr.DataArray, core_dims: tuple[str, str] | None = None
    ) -> dict[str, int]:
        """
        Create a mapping similar to xr.DataArray.sizes which maps dimension
        names to array sizes but exclude the image dimensions.


        Parameters
        ----------
        labels: xr.DataArray
            DataArray of integers containing labeled regions. Must be at least 2D

        core_dims: tuple[str,str]|None
            Names of dimensions that correspond to each image - typically something like
            ("y","x"). If None, use the last two dimensions of the array as image
            dimensions. Default None

        Returns
        -------
        loop_sizes: dict[str:int]

        """

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
        """
        Compute properties of labeled regions using skimage.regionprops.
        Unlike a typical groupby, implementaion treats a dataset as a set of
        2D images and computes the properties of each region in each one. This
        allows for separately measuring the same object in multiple frames.
        See skimage.measure.regionprops docs for further information.

        Parameters
        ----------
        labels_name: str
            Name of variable in dataset which corresponds to the labels
        intensity: str | None
            Name of variable in the dataset to be used as the intensity.
            If None, it will only be possible to compute geomteric properties
            of the labels.
        properties: tuple[str,...]
            Tuple of properties to compute for each region.
            Default ("labels", "bbox")
        core_dims: tuple[str,str]

        Returns
        -------
        regionprops: pd.DataFrame
            DataFrame containing a row for each labeled object in each image
            and columns corresponding to the computeed properties. For scalar
            properties the column names correspond to the property string. For
            fixed size properties (e.g. bbox) there is a separate column for each
            element of the property with a numbered suffix (e.g. "bbox-1", "bbox-2",
            etc.) For variable sized properties, there will be a single column with
            name matching the property string and dtype object containing variable
            sized arrays.

        """
        if intensity is not None:
            label_sizes = self._obj[labels_name].sizes
            img_sizes = self._obj[intensity].sizes

            assert label_sizes == img_sizes, (
                f"Label sizes must match intensity sizes but found labels with sizes"
                f"{label_sizes} and intensity with sizes {img_sizes}"
            )
        loop_sizes = self._get_loop_sizes(self._obj[labels_name], core_dims)
        all_props = []

        for dims in product(*(range(v) for v in loop_sizes.values())):
            idxr = dict(zip(loop_sizes.keys(), dims))
            labels_arr = self._obj[labels_name].isel(idxr).data

            if intensity is not None:
                intensity_arr = self._obj[intensity].isel(idxr).data
            else:
                intensity_arr = None

            # for now this lets skimage convert everything into numpy arrays
            # possible to refactor to use dask.delayed for lazy version.
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
