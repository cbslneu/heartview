=====================
Beat Editor I/O Files
=====================

The HeartView Beat Editor reads and writes data formatted as **JSON files**.

Creating a Beat Editor File
===========================

Input JSON files must include specific keys to represent time-series data and beat annotations.
Additionally, filenames should end with the ``'_edit.json'`` suffix to be recognized by the editor.

You can generate this file automatically using
`heartview.write_beat_editor_file() <api.html#heartview.heartview.write_beat_editor_file>`_,
or create one manually using the following required keys:

Required Keys
-------------

- ``Timestamp`` (or alternatively ``Sample``) representing the time at which each data point occurs.
  
  - ``Timestamp``: Unix epoch time in milliseconds.
  - ``Sample`` can be used instead if the data is indexed by sample number instead of actual time.

- ``Signal``: The actual ECG/PPG values.

- ``Beat``: Annotations of 1s marking where heartbeats occur in the signal.

Optional Key
------------

- ``Artifact`` *(optional)*: Annotations of 1s marking where artifactual heartbeats occur in the signal. See `SQA.Cardio.identify_artifacts() <api.html#heartview.pipeline.SQA.Cardio.identify_artifacts>`_ for artifact identification methods.

Loading Edit Data
=================
Input JSON files must be placed in the ``beat-editor/data`` subdirectory to
be recognized by the Beat Editor.

Processing Edited Data
======================
The Beat Editor saves all edited data as a JSON file, using the same base
filename as the input file, but with the ``'_edited.json'`` suffix.

All edited files are written to the ``beat-editor/saved`` subdirectory and
may include entries with the following keys:

- ``x``: The x-coordinate (time or sample index) of the edited beat.
- ``y``: The signal value at the edited beat location.
- ``from``: The start of a segment marked as 'Unusable'.
- ``to``: The end of a segment marked as 'Unusable'.
- ``editType``: The type of edit performed. Possible values are ``ADD``, ``DELETE``, or ``UNUSABLE``.

Edited data can be processed using `heartview.process_beat_edits() <api
.html#heartview.heartview.process_beat_edits>`_ as shown in the following
example workflow:

.. code-block:: python

    import pandas as pd
    from heartview import heartview

    # Assuming you are working from the heartview/ project root
    orig_data = pd.read_json('beat-editor/data/sample_edit.json')
    edits = pd.read_json('beat-editor/saved/sample_edited.json')
    processed_data = heartview.process_beat_edits(orig_data, edits)

    # Write the processed data
    processed_data.to_csv('processed_data.csv', index = False)