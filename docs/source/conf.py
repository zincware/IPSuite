# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

project = "IPSuite"
project_copyright = "2024, zincwarecode"
author = "zincwarecode"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration




sys.path.insert(0, Path("../../ipsuite/").resolve().as_posix())

extensions = ["sphinx.ext.autodoc", 
              "sphinx.ext.doctest", 
              "sphinx_copybutton", 
              "sphinx.ext.viewcode"]


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
