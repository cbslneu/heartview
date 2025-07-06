=============
API Reference
=============

Data Pre-Processing
===================

ECG
...

.. automodule:: heartview.pipeline.ECG
   :members:

PPG
...

.. automodule:: heartview.pipeline.PPG
   :members:

ACC
...

.. automodule:: heartview.pipeline.ACC
   :members:

Signal Quality Assessment
=========================

.. automodule:: heartview.pipeline.SQA
   :members:

Devices
=======
Actiwave Cardio
...............

.. autoclass:: heartview.heartview.Actiwave
    :members:
    :undoc-members:

Empatica E4
...........

.. autoclass:: heartview.heartview.Empatica
    :members:
    :undoc-members:

Beat Editor
===========
.. _write-beat-editor-file:

.. autofunction:: heartview.heartview.write_beat_editor_file

.. _process_beat_edits:

.. autofunction:: heartview.heartview.process_beat_edits
