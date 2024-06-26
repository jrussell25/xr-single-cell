# Single-cell accessor for Xarray

##Description
For the accessor assignment, I created this `single_cell` cell accessor
which computes properties of labeled regions from an xr.Dataset containing
images and labels as variables (or just labels).

A typical application would be to collect a timelapse imaging dataset of cells
expressing some fluorescent protein. After segmenting the cells (generating the labels)
its common to want to quantify the average intensity of an individual cell over time.

*Disclosure* This is inspired/based on some other work I did previously
([here](https://github.com/Hekstra-Lab/microutil/blob/main/microutil/single_cell/single_cell.py)
and [here](https://github.com/jrussell25/dask-regionprops))
which did not use accessors but probably should have.

### Installation

Clone this repo from [github](https://github.com/jrussell25/xr-single-cell) and install
with

`pip install -e .[dev]`

Or install without cloning

`pip install git+https://github.com/jrussell25/xr-single-cell.git`

## Usage

Register the `single_cell` accessor by importing this package

`import xr_single_cell`

Then use it compute region properties e.g.

```python

import xarray as xr
import numpy as np

import xr_single_cell

img = xr.Dataset(np.arange(60).reshape(3,4,5), dims=list('tyx'))

labels = xr.zeros_like(img, dtype='u2')
labels.data[:,:2,:2] = 1
labels.data[:,2:,2:] = 2

ds = xr.Dataset({"images":img, "labels":labels})

props = ds.single_cell.regionprops("labels", "images", properties=("label", "centroid",
"intensity_mean"))

```
