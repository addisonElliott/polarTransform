from setuptools import setup
import os

currentPath = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(currentPath, 'README.rst'), 'r') as f:
    long_description = f.read()

long_description = '\n' + long_description
setup(name='polarTransform',
      version='1.0.0',
      description='Library that can converts between polar and cartesian domain.',
      long_description=long_description,
      author='Addison Elliott',
      author_email='addison.elliott@gmail.com',
      url='https://github.com/addisonElliott/polarTransform',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3'
      ],
      keywords='polar transform cartesian conversion logPolar linearPolar cv2 opencv radius theta angle',
      project_urls={
          'Documentation': 'http://polartransform.readthedocs.io',
          'Source': 'https://github.com/addisonElliott/polarTransform',
          'Tracker': 'https://github.com/addisonElliott/polarTransform/issues',
      },
      python_requires='>=3',
      py_modules=['polarTransform'],
      license='MIT License',
      install_requires=[
          'numpy', 'scipy', 'scikit-image']
      )
