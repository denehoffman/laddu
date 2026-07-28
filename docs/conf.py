from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from importlib.util import find_spec
from pathlib import Path

project = 'laddu'
author = 'Nathaniel Dene Hoffman'
try:
    release = package_version('laddu')
except PackageNotFoundError:
    release = 'development'
version = release

extensions = [
    'myst_parser',
    'autoapi.extension',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx_design',
]
autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
napoleon_numpy_docstring = True
myst_enable_extensions = ['amsmath', 'colon_fence', 'dollarmath', 'fieldlist']
myst_heading_anchors = 3
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
templates_path = ['_templates']

laddu_spec = find_spec('laddu')
if laddu_spec is None or laddu_spec.origin is None:
    msg = 'laddu must be installed before building its API documentation'
    raise RuntimeError(msg)
autoapi_type = 'python'
autoapi_dirs = [str(Path(laddu_spec.origin).parent)]
# ganesh currently emits a few signatures that are valid for type checkers but
# not accepted by AutoAPI's parser. Parse laddu's public root stub here and let
# native autodoc cover the optimizer submodule below.
autoapi_file_patterns = ['__init__.pyi']
autoapi_root = 'reference/generated'
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_options = ['members', 'undoc-members', 'show-inheritance', 'show-module-summary']
suppress_warnings = ['autoapi.python_import_resolution']


_DUPLICATE_ATTRIBUTE_MEMBERS = {
    'laddu.BinnedDataset.dataset',
    'laddu.BinnedDataset.high',
    'laddu.BinnedDataset.index',
    'laddu.BinnedDataset.low',
    'laddu.GenerationReport.acceptance_rate',
    'laddu.GenerationReport.envelope',
    'laddu.GenerationReport.seed',
}


def _skip_duplicate_members(app, what, name, obj, skip, options):
    """Keep class-level attribute prose without indexing members twice."""
    del app, what, obj, options
    if name in _DUPLICATE_ATTRIBUTE_MEMBERS:
        return True
    if name.rsplit('.', maxsplit=1)[-1] in {'FULL', 'TENSOR', 'PHYSICAL', 'UNPHYSICAL'}:
        return True
    return skip


def setup(app):
    app.connect('autoapi-skip-member', _skip_duplicate_members)


html_theme = 'furo'
html_title = 'laddu'
html_logo = '_static/logo.svg'
html_favicon = '_static/logo.svg'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#367979',
        'color-brand-content': '#367979',
        'color-admonition-background': '#f4f8f8',
    },
    'dark_css_variables': {
        'color-brand-primary': '#78a5a5',
        'color-brand-content': '#9bc2c2',
    },
    'source_repository': 'https://github.com/denehoffman/laddu/',
    'source_branch': 'main',
    'source_directory': 'docs/',
}
