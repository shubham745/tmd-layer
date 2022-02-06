from setuptools import setup, find_packages

setup(
  name = 'tmd-layer',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'TMD Layer',
  author = 'Shubham Agarwal',
  author_email = 'shubham745@gmail.com',
  url = 'https://github.com/shubham745/tmd_layer',
  keywords = [
    'artificial intelligence',
    'Stochastic Differential Equation',
    'attention mechanism'
  ],
  install_requires=[
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
