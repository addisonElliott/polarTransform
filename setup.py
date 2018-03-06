try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

long_description = """
Library that can converts between polar and cartesian domain. See the github page for more information.
"""

setup(name='polar-transform',
      version='1.0.0',
      description='Library that can converts between polar and cartesian domain.',
      long_description=long_description'Library that can converts between polar and cartesian domain. See the github page for more information.',
      author='Addison Elliott',
      author_email='addison.elliott@gmail.com',
      url='https://github.com/addisonElliott/polar-transform',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3'
      ],
      keywords='polar transform cartesian conversion logPolar linearPolar cv2 opencv radius theta angle',
      project_urls={
          'Documentation': 'https://github.com/addisonElliott/polar-transform/issues',
          'Source': 'https://github.com/addisonElliott/polar-transform',
          'Tracker': 'https://github.com/addisonElliott/polar-transform/issues',
      },
      python_requires='>=3',
      py_modules=['polarTransform'],
      license="MIT License",
      install_requires=['numpy', 'scipy', 'scikit-image']
     )