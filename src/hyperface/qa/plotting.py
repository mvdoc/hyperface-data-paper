"""Common plotting utilities for QA visualizations."""

import warnings
from typing import Any

# Style presets
VIOLIN_STYLES = {
    "default": {
        "facecolor": "steelblue",
        "edgecolor": "darkslategray",
        "alpha": 0.7,
    },
    "fd": {
        "facecolor": "lightcoral",
        "edgecolor": "darkred",
        "alpha": 0.7,
    },
    "isc": {
        "facecolor": "steelblue",
        "edgecolor": "darkslategray",
        "alpha": 0.7,
    },
}


def style_violin_plot(
    parts: dict,
    style: str = "default",
    custom_style: dict[str, Any] | None = None,
) -> None:
    """Apply consistent styling to violin plot parts.

    This function applies uniform styling to the result of plt.violinplot(),
    ensuring consistent appearance across all QA visualizations.

    Parameters
    ----------
    parts : dict
        Return value from plt.violinplot(). Contains 'bodies' (list of
        PolyCollection) and optional 'cbars', 'cmins', 'cmaxes', 'cmedians'.
    style : str
        Preset style name: "default" (blue), "fd" (coral for framewise
        displacement), "isc" (blue, same as default).
    custom_style : dict, optional
        Override preset with custom colors. Keys: 'facecolor', 'edgecolor', 'alpha'.

    Raises
    ------
    ValueError
        If parts dict doesn't contain 'bodies' key.

    Notes
    -----
    If an invalid style name is provided, falls back to "default" style
    with a warning.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> data = [np.random.randn(100) for _ in range(5)]
    >>> parts = plt.violinplot(data)
    >>> style_violin_plot(parts, style="default")

    >>> # Custom styling
    >>> custom = {"facecolor": "green", "edgecolor": "darkgreen"}
    >>> style_violin_plot(parts, custom_style=custom)
    """
    # Validate input
    if "bodies" not in parts:
        raise ValueError(
            "Invalid violin plot parts dictionary. Expected 'bodies' key. "
            "Make sure you pass the return value from plt.violinplot()."
        )

    # Get style settings
    if custom_style is not None:
        settings = {**VIOLIN_STYLES.get("default", {}), **custom_style}
    elif style in VIOLIN_STYLES:
        settings = VIOLIN_STYLES[style]
    else:
        warnings.warn(
            f"Unknown violin style '{style}', falling back to 'default'. "
            f"Available styles: {list(VIOLIN_STYLES.keys())}",
            stacklevel=2,
        )
        settings = VIOLIN_STYLES["default"]

    facecolor = settings.get("facecolor", "lightblue")
    edgecolor = settings.get("edgecolor", "navy")
    alpha = settings.get("alpha", 0.7)

    # Style the violin bodies
    for pc in parts["bodies"]:
        pc.set_facecolor(facecolor)
        pc.set_edgecolor(edgecolor)
        pc.set_alpha(alpha)

    # Style the statistical markers
    for part_name in ["cbars", "cmins", "cmaxes", "cmedians"]:
        if part_name in parts:
            parts[part_name].set_edgecolor(edgecolor)
            parts[part_name].set_linewidth(1.5)
