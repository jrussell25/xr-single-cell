import pytest
from itertools import product

import xarray as xr
import numpy as np
import pandas as pd

from skimage.measure import regionprops_table
from skimage.draw import disk

import xr_single_cell  # noqa: F401


@pytest.fixture(scope="session")
def dataset4d() -> tuple[xr.Dataset, tuple[str, ...], pd.DataFrame]:
    N = 100
    n_obj = 4
    labels = np.zeros((2, 3, N, N), dtype="u2")

    rng = np.random.default_rng(seed=123)
    images = rng.poisson(50, size=labels.shape)

    properties = ("label", "centroid", "bbox", "coords", "intensity_mean")

    r0 = N / (2 * n_obj)
    d0 = int(N / n_obj)
    props = []
    for dims in product(*(range(x) for x in labels.shape[:-2])):
        for i in range(n_obj):
            row_offset = 2 * (i // 2) + 1
            col_offset = 2 * (i % 2) + 1
            rr, cc = disk(
                (d0 * row_offset, d0 * col_offset),
                r0 + i * dims[0] + dims[1],
                shape=(N, N),
            )
            labels[dims[0], dims[1], rr, cc] = i + 1
        ps = pd.DataFrame(
            regionprops_table(labels[dims], images[dims], properties=properties)
        )
        ps["S"] = dims[0]
        ps["T"] = dims[1]
        props.append(ps)

    labels = xr.DataArray(labels, dims=list("STYX"))
    images = xr.DataArray(images, dims=list("STYX"))
    ds = xr.Dataset({"images": images, "labels": labels})
    regionprops = pd.concat(props, ignore_index=True)

    return ds, properties, regionprops


def test_empty() -> None:
    images = xr.DataArray(np.zeros((1, 1)), dims=list("YX"))
    labels = xr.zeros_like(images, dtype="u2")
    ds = xr.Dataset({"images": images, "labels": labels})
    props = ds.single_cell.regionprops("labels", "images")

    expect = pd.DataFrame(
        columns=["label", *(f"bbox-{i}" for i in range(4))], dtype=int
    )

    pd.testing.assert_frame_equal(props, expect)


def test_labels_only(
    dataset4d: tuple[xr.Dataset, tuple[str, ...], pd.DataFrame],
) -> None:
    ds, properties, regionprops = dataset4d

    ds_in = ds[["labels"]]

    with pytest.raises(AttributeError):
        result = ds_in.single_cell.regionprops("labels", properties=properties)

    properties_in = properties[:-1]
    expect = regionprops.drop(columns="intensity_mean")

    result = ds_in.single_cell.regionprops("labels", properties=properties_in)

    assert result.shape == (24, 10)
    pd.testing.assert_frame_equal(result, expect)


def test_intensity_image(
    dataset4d: tuple[xr.Dataset, tuple[str, ...], pd.DataFrame],
) -> None:
    ds, properties, regionprops = dataset4d

    # 4d
    result = ds.single_cell.regionprops("labels", "images", properties)
    pd.testing.assert_frame_equal(result, regionprops)

    # 3d
    result = ds.isel(S=0).single_cell.regionprops("labels", "images", properties)
    expect = (
        regionprops.loc[regionprops["S"] == 0].drop(columns="S").reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(result, expect)

    # 2d
    result = ds.isel(S=0, T=1).single_cell.regionprops("labels", "images", properties)
    expect = (
        regionprops.loc[(regionprops["S"] == 0) & (regionprops["T"] == 1)]
        .drop(columns=["S", "T"])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(result, expect)

    # 1d
    with pytest.raises(TypeError):
        result = ds.isel(S=0, T=1, Y=2).single_cell.regionprops(
            "labels", "images", properties
        )


def test_names(dataset4d: tuple[xr.Dataset, tuple[str, ...], pd.DataFrame]) -> None:
    ds, properties, regionprops = dataset4d

    ds = ds.rename({"labels": "new_labels", "images": "new_images"})
    for dim in ds.dims:
        ds = ds.rename({dim: dim.lower()})

    regionprops = regionprops.rename(columns={"S": "s", "T": "t"})

    result = ds.single_cell.regionprops("new_labels", "new_images", properties)
    pd.testing.assert_frame_equal(result, regionprops)


def test_core_dims(dataset4d: tuple[xr.Dataset, tuple[str, ...], pd.DataFrame]) -> None:
    ds, properties, regionprops = dataset4d

    ds = ds.isel(S=0, X=slice(None, None, 10))
    result = ds.single_cell.regionprops(
        "labels", "images", properties=properties, core_dims=("T", "Y")
    )

    all_props = []
    for x in range(ds.sizes["X"]):
        img = ds["images"].isel(X=x).data
        labels = ds["labels"].isel(X=x).data
        rps = pd.DataFrame(regionprops_table(labels, img, properties))
        rps["X"] = x
        all_props.append(rps)

    expect = pd.concat(all_props, ignore_index=True)

    pd.testing.assert_frame_equal(result, expect)
