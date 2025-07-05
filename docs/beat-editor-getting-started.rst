===============
Getting Started
===============

Overview
========
HeartView Beat Editor is a user-friendly tool for visualizing and editing cardiac data.
The editor provides out-of-the-box functionality for visualizing and editing several
types of cardiac data, including ECG and PPG. 

Manual Installation
===================
The Beat Editor requires Node, please refer to the
installation instructions on their website: https://nodejs.org/en/download/.
Once Node is installed, you can check if the installation was successful by running:

::

    node -v
    npm -v

If the command returns a version number, Node is installed correctly.
If you encounter any issues, please refer to the Node.js documentation for troubleshooting.


Once Node is installed, follow these steps to set up the Beat Editor:
Make sure you're in the ``beat-editor`` directory, then run the following command to install the
required modules for the Beat Editor:

::

    npm install


Launching Beat Editor
=====================
To launch the Beat Editor, we have to boot up the backend server first:

1. Open a terminal and navigate to the `beat-editor/server` directory.

::

    cd beat-editor/server

2. Start the backend server by running:

::
    
    npm start

If the terminal displays this message, then the server is running successfully:

::
  
    server@1.0.0 start
    node app.js

    Server is running on port 3001

3. Open another terminal and navigate to the `beat-editor` directory.

::

    cd beat-editor

4. Start the front-end by running:

::
    
    npm start

If the terminal displays this message, then the front-end is running successfully:

::

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
Open your web browser and go to: http://localhost:3000
You should see the Beat Editor interface, where you can visualize and edit cardiac data.

Terminating the Beat Editor
===========================
1. In the terminal where the backend server is running, press ``CTRL`` + ``C`` to stop the server.
2. In the terminal where the front-end is running, press ``CTRL`` + ``C`` to stop the front-end.





