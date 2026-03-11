"""Module containing viz utils"""

import os
import time
from pathlib import Path

import cortex
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

WEBGL_PORT = 8765

params_fsaverage_allviews = {
    "figsize": [16, 9],
    "panels": [
        {
            # x0, y0, width, height
            "extent": [0.007, 0.2, 0.99, 0.8],
            "view": {"angle": "flatmap", "surface": "flatmap"},
        },
        {
            "extent": [0.13, 0.04, 0.16, 0.22],
            "view": {
                "hemisphere": "left",
                "angle": "medial_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.71, 0.04, 0.16, 0.22],
            "view": {
                "hemisphere": "right",
                "angle": "medial_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.00, 0.18, 0.16, 0.22],
            "view": {
                "hemisphere": "left",
                "angle": "lateral_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.84, 0.18, 0.16, 0.22],
            "view": {
                "hemisphere": "right",
                "angle": "lateral_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.30, 0.02, 0.17, 0.16],
            "view": {
                "hemisphere": "left",
                "angle": "bottom_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.52, 0.02, 0.17, 0.16],
            "view": {
                "hemisphere": "right",
                "angle": "bottom_pivot",
                "surface": "inflated",
            },
        },
    ],
}


def make_mosaic(data):
    """Reshape data into a mosaic plot

    Parameters
    ---------
    data : array of shape (dim1, dim2, dim3)
        input volume

    Returns
    ------
    mosaic : array
        mosaic matrix that can be plotted with matshow
    """
    # add extra slices top and bottom to make it divisible by 6
    dim1, dim2, dim3 = data.shape
    empty_slice = np.zeros((dim1, dim2, 1))
    n_cols = 10
    n_rows = int(np.ceil(dim3 / n_cols))
    n_extra_slices = n_cols * n_rows - dim3
    n_extra_slices_half = n_extra_slices // 2
    to_concat = [empty_slice] * n_extra_slices_half
    to_concat += [data]
    to_concat += [empty_slice] * n_extra_slices_half
    if n_extra_slices % 2 != 0:
        to_concat += [empty_slice]
    t = np.concatenate(to_concat, -1)
    assert t.shape[-1] == n_cols * n_rows
    # split into rows
    t = np.split(t, n_rows, -1)
    # make matrix with some magic
    t = np.vstack([tt.transpose(1, 0, 2).reshape(dim2, -1, order="F") for tt in t])
    # change order so that plots match standard mosaic order
    t = t[::-1, ::-1]
    return t


def plot_mosaic(mat, vmin=30, vmax=250, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(18, 18))
    im = ax.matshow(mat, vmin=vmin, vmax=vmax, interpolation="nearest", cmap="inferno")
    if title:
        ax.text(
            0,
            0,
            title,
            ha="left",
            va="top",
            fontsize=24,
            bbox={"facecolor": "white", "alpha": 1},
        )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")
    return fig


def setup_pycortex_fsaverage(download_again: bool = False) -> None:
    """Download fsaverage surface for pycortex if not present.

    This function downloads the fsaverage subject data required by pycortex
    for cortical surface visualizations. It should be called once before
    any pycortex visualization functions.

    Parameters
    ----------
    download_again : bool, default=False
        If True, re-download even if already present.
    """
    cortex.utils.download_subject(subject_id="fsaverage", download_again=download_again)


def has_display() -> bool:
    """Check if a display is available for pycortex 3D rendering.

    Returns
    -------
    bool
        True if DISPLAY environment variable is set (X11 available),
        False otherwise.
    """
    return bool(os.environ.get("DISPLAY"))


def upsample_fsaverage6_to_fsaverage(
    data: np.ndarray,
    freesurfer_subjects_dir: Path | str | None = None,
) -> np.ndarray:
    """Upsample vertex data from fsaverage6 to fsaverage resolution.

    Parameters
    ----------
    data : np.ndarray
        Vertex data in fsaverage6 space. Can be 1D (n_vertices,) for single
        map or 2D (n_maps, n_vertices) for multiple maps.
        Expected: ~81,924 vertices (40,962 per hemisphere).
    freesurfer_subjects_dir : Path or str, optional
        Path to FreeSurfer subjects directory containing fsaverage.
        If None, uses $SUBJECTS_DIR environment variable.

    Returns
    -------
    np.ndarray
        Upsampled data in fsaverage space (~327,684 vertices).
    """
    fs_dir = str(freesurfer_subjects_dir) if freesurfer_subjects_dir else None
    return cortex.freesurfer.upsample_to_fsaverage(
        data, "fsaverage6", freesurfer_subjects_dir=fs_dir
    )


def _prepare_fsaverage_vertex(
    data: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    freesurfer_subjects_dir: Path | str | None,
) -> cortex.Vertex:
    """Prepare a pycortex Vertex object from fsaverage6 data.

    This helper handles the common setup needed for surface visualization:
    ensures fsaverage is available, upsamples the data, and creates the Vertex.
    """
    setup_pycortex_fsaverage()
    data_upsampled = upsample_fsaverage6_to_fsaverage(data, freesurfer_subjects_dir)
    return cortex.Vertex(data_upsampled, "fsaverage", cmap=cmap, vmin=vmin, vmax=vmax)


def create_fsaverage6_plot(
    data: np.ndarray,
    output_path: Path,
    cmap: str = "hot",
    vmin: float = 0.0,
    vmax: float = 0.3,
    freesurfer_subjects_dir: Path | str | None = None,
    title: str | None = None,
) -> None:
    """Create cortical surface plot of ISC or similar vertex data.

    Parameters
    ----------
    data : np.ndarray
        Vertex data in fsaverage6 space (will be upsampled to fsaverage).
        Shape should be (n_vertices,) where n_vertices ~ 81,924.
    output_path : Path
        Output file path for the saved figure (PNG).
    cmap : str, default="hot"
        Matplotlib colormap name.
    vmin : float, default=0.0
        Minimum value for color scale.
    vmax : float, default=0.3
        Maximum value for color scale.
    freesurfer_subjects_dir : Path or str, optional
        Path to FreeSurfer subjects directory containing fsaverage.
        If None, uses $SUBJECTS_DIR environment variable.
    title : str, optional
        Title to display on the figure. If None, no title is added.

    Notes
    -----
    - If DISPLAY is available: uses cortex.export.plot_panels() for
      inflated lateral/medial/ventral views (requires WebGL).
    - If no DISPLAY: uses cortex.quickflat.make_figure() for flatmap
      visualization (matplotlib-only, no display required).
    """
    surface = _prepare_fsaverage_vertex(
        data, cmap, vmin, vmax, freesurfer_subjects_dir
    )

    if has_display():
        # Inflated 3D views (requires display/WebGL)
        params = params_fsaverage_allviews
        viewer_params = {
            "labels_visible": [],
            "overlays_visible": ["rois", ],
        }
        fig = cortex.export.plot_panels(
            surface,
            # windowsize=windowsize,
            viewer_params=viewer_params,
            **params,
        )
        if title:
            fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    else:
        # Flatmap visualization (matplotlib, no display needed)
        fig = cortex.quickflat.make_figure(
            surface,
            with_rois=False,
            with_curvature=False,
            colorbar_location="right",
            height=800,
        )
        if title:
            fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved surface plot to: {output_path}")


def start_webgl_viewer(
    data: np.ndarray,
    cmap: str = "hot",
    vmin: float = 0.0,
    vmax: float = 0.3,
    freesurfer_subjects_dir: Path | str | None = None,
    port: int = WEBGL_PORT,
    sleep_seconds: float = 3600,
) -> None:
    """Start an interactive pycortex webgl viewer for fsaverage6 data.

    Parameters
    ----------
    data : np.ndarray
        Vertex data in fsaverage6 space (will be upsampled to fsaverage).
        Shape should be (n_vertices,) where n_vertices ~ 81,924.
    cmap : str, default="hot"
        Matplotlib colormap name.
    vmin : float, default=0.0
        Minimum value for color scale.
    vmax : float, default=0.3
        Maximum value for color scale.
    freesurfer_subjects_dir : Path or str, optional
        Path to FreeSurfer subjects directory containing fsaverage.
        If None, uses $SUBJECTS_DIR environment variable.
    port : int, default=8765
        Port number for the webgl server.
    sleep_seconds : float, default=3600
        Time in seconds to keep the server running (default 1 hour).
    """
    surface = _prepare_fsaverage_vertex(
        data, cmap, vmin, vmax, freesurfer_subjects_dir
    )

    print(f"Starting webgl viewer on port {port}...")
    print(f"Access at: http://localhost:{port}")
    print("Press Ctrl+C to stop the server.")

    cortex.webgl.show(surface, open_browser=False, port=port)
    time.sleep(sleep_seconds)
