# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import typing as t

import ipsuite

project = "IPSuite"
project_copyright = "2025, zincwarecode"
author = "zincwarecode"
release = ipsuite.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
]


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_title = "IPSuite"
html_short_title = "ipsuite"

html_theme_options: t.Dict[str, t.Any] = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/zincware/ipsuite",
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ],
    "source_repository": "https://github.com/zincware/ipsuite/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "navigation_with_keys": True,
}

# font-awesome logos
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]
