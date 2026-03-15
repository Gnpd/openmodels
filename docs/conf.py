# Configuration file for the Sphinx documentation builder.

project = "OpenModels"
copyright = "2026, SF Technologies"
author = "SF Technologies"
release = "0.1.0-alpha.21"
html_title = "OpenModels"
html_short_title = "OpenModels"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

language = "en"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Mock imports that may not be available when building docs
autodoc_mock_imports = ["sklearn", "numpy", "np", "scipy"]

# Disable translation/gettext support
gettext_auto_build = False
gettext_compact = False

# -- numpydoc configuration --------------------------------------------------

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# -- intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Gnpd/openmodels",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/openmodels/",
            "icon": "fa-solid fa-box",
        },
    ],
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
}
