import setuptools

# Dependencies
requirements = [
    'blosc2==2.0.0',
    'celery',
    'cython==0.29.21',
    'dash',
    'dash_bootstrap_components',
    'dash_daq',
    'dash_uploader',
    'diskcache',
    'matplotlib',
    'multiprocess',
    'numpy',
    'packaging==21.3',
    'pandas',
    'plotly',
    'psutil',
    'pyEDFlib',
    'scipy'
]

setuptools.setup(
    name = 'heartview',
    version = '1.0',
    author = 'Natasha Yamane, Varun Mishra, and Matthew S. Goodwin',
    description = 'A signal quality assessment pipeline for wearable cardiovascular data',
    license = '',
    packages = setuptools.find_packages(),
    install_requires = requirements,
    url = 'https://github.com/cbslneu/heartview',
    python_requires = '>=3.8.6'
)

