===============
Getting Started
===============

Overview
========
HeartView Beat Editor is a user-friendly tool providing out-of-the-box
functionality for visualizing and editing electrocardiograph and
photoplethysmograph data.

Manual Installation
===================
The Beat Editor requires Node. Please refer to the
installation instructions on their `website <https://nodejs
.org/en/download/>`_.
Once Node is installed, you can check if the installation was successful by running:

.. code-block:: bash

    $ node -v
    $ npm -v

If the command returns a version number, Node is installed correctly. Please
refer to the `Node.js documentation
<https://nodejs.org/docs/latest/api/>`_ for additional troubleshooting.

Once Node is installed, proceed with Beat Editor installation.
From the main ``heartview`` root directory, navigate to the ``beat-editor``
subdirectory and run the following command to install all required modules
for the Beat Editor:

.. code-block:: bash

    $ cd beat-editor
    $ npm install

Next we need to install the backend server dependencies.
From the ``beat-editor`` subdirectory, navigate to the ``server``
and run the following command to install all required modules:

.. code-block:: bash

    $ cd server
    $ npm install


Launching the Beat Editor
=========================
To launch the Beat Editor, we have to boot up the backend server first:

1. Navigate to the ``beat-editor/server`` subdirectory.

.. code-block:: bash

    $ cd beat-editor/server

2. Start the backend server by running:

.. code-block:: bash
    
    $ npm start

If the terminal displays this message, then the server is running successfully:

.. code-block:: bash
  
    >server@1.0.0 start
    >node app.js

    Server is running on port 3001

3. Open a new terminal and navigate to the main ``beat-editor`` subdirectory.

.. code-block:: bash

    $ cd beat-editor

4. Start the front-end by running:

.. code-block:: bash
    
    $ npm start

If the terminal displays this message, then the front-end is running successfully:

.. code-block:: bash

    Compiled successfully!

    You can now view beat-editor in the browser.

      Local:            http://localhost:3000
      On Your Network:  http://10.0.0.251:3000

    **Note:** The development build is not optimized.
    To create a production build, use npm run build.

    webpack compiled successfully


If the terminal displays an error or warning, please kill
the process by pressing ``CTRL`` + ``C`` and try running the command to start the front-end again.

Accessing the Beat Editor
=========================
Open your web browser and go to: http://localhost:3000.
You should see the Beat Editor interface, where you can visualize and edit cardiac data.

Terminating the Beat Editor
===========================
1. In the terminal where the backend server is running, press ``CTRL`` + ``C`` to stop the server.
2. In the terminal where the front-end is running, press ``CTRL`` + ``C`` to stop the front-end.
