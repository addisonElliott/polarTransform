================
Getting Started
================

.. rubric:: Brief overview of polarTransform and how to install.

Introduction
============
polarTransform is a Python package for converting images between the polar and Cartesian domain. It contains many
features such as specifying the start/stop radius and angle, interpolation order (bicubic, linear, nearest, etc), and
much more.

License
============
polar-transform has an MIT-based `license <https://github.com/addisonElliott/polarTransform/blob/master/LICENSE>`_.

Installing
============
Prerequisites
-------------
* Python 3
* Dependencies:
   * numpy
   * scipy
   * scikit-image

Installing polarTransform
-------------------------
polarTransform is currently available on `PyPi <https://pypi.python.org/pypi/polarTransform/>`_. The simplest way to
install alone is using ``pip`` at a command line::

  pip install polarTransform

which installs the latest release.  To install the latest code from the repository (usually stable, but may have
undocumented changes or bugs)::

  pip install git+https://github.com/addisonElliott/polarTransform.git


For developers, you can clone the pydicom repository and run the ``setup.py`` file. Use the following commands to get
a copy from GitHub and install all dependencies::

  git clone pip install git+https://github.com/addisonElliott/polarTransform.git
  cd polarTransform
  pip install .

or, for the last line, instead use::

  pip install -e .

to install in 'develop' or 'editable' mode, where changes can be made to the local working code and Python will use
the updated polarTransform code.

Test and coverage
=================
To test the code on any platform, make sure to clone the GitHub repository to get the tests and::

  python tests/test_polarTransform.py

Using polarTransform
====================
Once installed, the package can be imported at a Python command line or used in your own Python program with ``import polarTransform``. See the :doc:`user-guide` for more details of how to use the package.

Support
===============
Bugs can be submitted through the `issue tracker <https://github.com/addisonElliott/polarTransform/issues>`_.

Pull requests are welcome too!

Next Steps
===============
To start learning how to use polar-transform, see the :doc:`user-guide`.
