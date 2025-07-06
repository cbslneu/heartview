============
Installation
============

In your **Terminal**:

1. Clone the HeartView GitHub repository into a directory of your choice\:

::

    $ git clone https://github.com/cbslneu/heartview.git

2. Set up and activate a virtual environment using **Python 3.9 through 3.13** inside the ``heartview`` project directory.

If you're unsure whether Python is installed, you can check by running:

::

    $ python3 --version

If you see an error or an unexpected version (e.g., Python 2.x), install the latest compatible Python 3 version from https://www.python.org/downloads/.

Once Python is available, create a virtual environment in the project directory:

::

    $ cd heartview
    $ virtualenv venv -p python3

**Note:** ``virtualenv`` can be installed via ``pip3 install virtualenv``.

3. Activate your newly created virtual environment:

::

    $ source venv/bin/activate

4. Install all project dependencies. (This may take a while.)

::

    $ pip3 install -r requirements.txt