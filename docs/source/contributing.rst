.. _contributing:

############
Contributing
############

We appreciate all contributions. If you are planning to contribute a bug fix for an open issue,
please open a github issue or comment on a thread and we're happy to provide any guidance.

Developer Setup
===============

Mac Setup
---------

You can develop directly on a Mac (M1) for most parts. Using the steps in the README works. Here is a condensed version for easy copy & paste:

.. code-block:: bash

   conda create --name <your_name> python=3.10
   conda activate <your_name>
   pip install --upgrade "jax[cpu]" torch
   pip install -r test_requirements.txt
   pip install -e .
   pytest test

VSCode
~~~~~~

It is recommended to use VSCode on Mac. You can follow the instructions in the `VSCode Python tutorial <https://code.visualstudio.com/docs/python/python-tutorial>`_ to set up a proper Python environment.

The recommended plugins are:

* VSCode's official Python plugin
* Ruff formatter
* Python Debugger

You should also change the Python interpreter to point at the one in your conda environment.


Linux setup
-----------

1. Create a VM
2. Attach a disk
3. Install miniconda
