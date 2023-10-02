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
copyright = '2023, Natasha Yamane, Varun Mishra, and Matthew S. Goodwin'
author = 'Natasha Yamane, Varun Mishra, and Matthew S. Goodwin'
release = '1.0'

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

html_theme = 'furo'
html_title = 'Documentation'
html_static_path = ['_static']
html_theme_options = {
    'source_repository': 'https://github.com/cbslneu/heartview/',
    'source_branch': 'main',
    'source_directory': 'docs/',
    'light_logo': 'light-heartview-logo.png',
    'dark_logo': 'dark-heartview-logo.png',
}
html_favicon = '../assets/favicon.ico'
