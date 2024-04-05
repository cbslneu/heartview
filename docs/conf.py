# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HeartView'
copyright = '2024, Natasha Yamane, Varun Mishra, and Matthew S. Goodwin'
author = 'Natasha Yamane, Varun Mishra, and Matthew S. Goodwin'
release = '2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'friendly'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'agogo'
html_static_path = ['_static']
html_logo = '../assets/heartview-logo.png'
html_favicon = '../assets/favicon.ico'
html_css_files = ['css/custom.css']

# -- Options for LaTeX output ------------------------------------------------
authors = 'Natasha Yamane, Varun Mishra, and Matthew S. Goodwin'
toctree_only = True
latex_documents = [
    ('index', 'heartview.tex', 'HeartView Documentation',
     authors.replace(', ', '\\and ').replace(' and ', '\\and and '),
     'manual', toctree_only)
]
