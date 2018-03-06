================
Getting Started
================

.. rubric:: Brief overview of polar-transform and how to install.


Introduction
============

polar-transform is a Python package for converting images between the polar and cartesian domain. It contains many features such as specifying the start/stop radius and angle, interpolation order (bicubic, linear, nearest, etc), and much more.

License
=======

polar-transform has an MIT-based `license
<https://github.com/addisonElliott/polar-transform/blob/master/LICENSE>`_.

Installing
==========

Prerequisites
-------------

* Python 2.7, 3.4 or later
* Dependencies:
   * numpy
   * scipy
   * skimage


Installing polar-transform
------------------

polar-transform is currently available on `PyPi <https://pypi.python.org/pypi/pydicom/>`_
. The simplest way to install pydicom alone is using ``pip`` at a command line::

  pip install -U polar-transform

which installs the latest release.  To install the latest code from the repository
(usually stable, but may have undocumented changes or bugs)::

  pip install -U git+https://github.com/pydicom/pydicom.git


For developers, you can clone the pydicom repository and run 
the ``setup.py`` file. Use the following commands to get a copy 
from GitHub and install all dependencies::

  git clone https://github.com/pydicom/pydicom.git
  cd pydicom
  pip install .

or, for the last line, instead use::

  pip install -e .

to install in 'develop' or 'editable' mode, where changes can be made to the
local working code and Python will use the updated pydicom code.


Test and coverage
=================

To test the installed code on any platform, change to the directory of 
pydicom's setup.py file and::

  python setup.py test

This will install `pytest <https://pytest.org>`_ if it is not 
already installed.


Using pydicom
=============

Once installed, the package can be imported at a Python command line or used
in your own Python program with ``import pydicom``.
See the `examples directory
<https://github.com/pydicom/pydicom/tree/master/examples>`_
for both kinds of uses. Also see the :doc:`User Guide </pydicom_user_guide>`
for more details of how to use the package.

Support
=======

Bugs can be submitted through the `issue tracker
<https://github.com/addisonElliott/polar-transform/issues>`_.  Besides the example directory,

Next Steps
==========

To start learning how to use polar-transform, see the :doc:`pydicom_user_guide`.