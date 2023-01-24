# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information

project = 'ForeTiS'
copyright = '2022, GrimmLab @ TUM Campus Straubing (https://bit.cs.tum.de/)'
author = 'Josef Eiglsperger, Florian Haselbeck, Dominik G. Grimm'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'autoapi.extension',
    'sphinx.ext.intersphinx',
    "sphinx.ext.imgconverter",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    'myst_parser',
    'sphinxcontrib.youtube',
    'sphinx.ext.autosectionlabel',
    'nbsphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {"logo_only": True, 'navigation_depth': 5, 'titles_only': False}

# -- Options for EPUB output
epub_show_urls = 'footnote'

exclude_patterns = ['*/docs*']

autoapi_type = 'python'
autoapi_dirs = ['../../']
autodoc_typehints = 'description'
autoapi_ignore = ['*conf*', '*setup*', '*run*']

autoapi_add_toctree_entry = False
autoapi_template_dir = '_autoapi_templates'
