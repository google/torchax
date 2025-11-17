# Contributing

We appreciate all contributions. If you are planning to contribute a bug fix for an open issue, please comment on the thread and we're happy to provide any guidance. You are very welcome to pick issues from ``good first issue`` and ``help wanted`` labels.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core in a different direction than you might be aware of.

## Developer Setup

### Mac Setup

You can develop directly on a Mac (M1) for most parts. Using the steps in the README works. Here is a condensed version for easy copy & paste:

```bash
   conda create --name <your_name> python=3.11
   conda activate <your_name>
   pip install --upgrade "jax[cpu]" torch
   pip install -r test-requirements.txt
   pip install -e .
   pytest test
```
### Ruff

```
ruff check torchax test test_dist examples --fix
ruff format torchax test test_dist examples
```

### VSCode

It is recommended to use VSCode on Mac. You can follow the instructions in the `VSCode Python tutorial <https://code.visualstudio.com/docs/python/python-tutorial>`_ to set up a proper Python environment.

The recommended plugins are:

* VSCode's official Python plugin
* Ruff formatter
* Python Debugger

You should also change the Python interpreter to point at the one in your conda environment.
