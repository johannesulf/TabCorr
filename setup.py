from setuptools import setup
from tabcorr import __version__

setup(name='tabcorr',
      version=__version__,
      description='Tabulated Correlation functions for halotools',
      url='https://github.com/johannesulf/TabCorr',
      author='Johannes U. Lange',
      author_email='julange.astro@pm.me',
      packages=['tabcorr'],
      install_requires=['numpy', 'scipy', 'astropy', 'h5py', 'tqdm'],
      zip_safe=False)
