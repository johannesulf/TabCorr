from setuptools import setup

setup(name='tabcorr',
      version='0.8.1',
      description='Tabulated correlation functions for halotools',
      url='https://github.com/johannesulf/TabCorr',
      author='Johannes U. Lange',
      author_email='julange.astro@pm.me',
      packages=['tabcorr'],
      install_requires=['numpy', 'scipy', 'astropy', 'h5py', 'tqdm'],
      zip_safe=False)
