import os

from setuptools import setup, find_packages

from polarTransform._version import __version__

currentPath = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(currentPath, 'README.rst'), 'r') as f:
    long_description = f.read()

long_description = '\n' + long_description
setup(name='polarTransform',
      version=__version__,
      description='Library that can converts between polar and cartesian domain with images and individual points.',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      author='Addison Elliott',
      author_email='addison.elliott@gmail.com',
      url='https://github.com/addisonElliott/polarTransform',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
      ],
      keywords='polar transform cartesian conversion logPolar linearPolar cv2 opencv radius theta angle image images',
      project_urls={
          'Documentation': 'http://polartransform.readthedocs.io',
          'Source': 'https://github.com/addisonElliott/polarTransform',
          'Tracker': 'https://github.com/addisonElliott/polarTransform/issues',
      },
      python_requires='>=3',
      packages=find_packages(),
      include_package_data=True,
      license='MIT License',
      install_requires=[
          'numpy', 'scipy', 'scikit-image']
      )
