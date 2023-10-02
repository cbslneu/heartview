============
Installation
============

In your **Terminal**:

1. Clone the HeartView GitHub repository into a directory of your choice.

::

    $ git clone https://github.com/cbslneu/heartview.git

2. Set up and activate a virtual environment in the ``heartview`` project directory.

::

    $ cd heartview
    $ virtualenv venv -p python3

**Note:** ``virtualenv`` can be installed via ``pip3 install virtualenv``.

3. Activate your newly created virtual environment.

::

    $ source venv/bin/activate

4. Install all project dependencies. (This may take a while.)

::

    $ pip3 install -r requirements.txt