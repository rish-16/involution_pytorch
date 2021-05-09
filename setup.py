from setuptools import setup, find_packages

setup(
  name = 'involution_pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Involution Operation - Pytorch',
  long_description = 'Unofficial wrapper around Involution wrapper by HKUST, ByteDance AI, and Peking University',
  author = 'Rishabh Anand',
  author_email = 'mail.rishabh.anand@gmail.com',
  url = 'https://github.com/rish-16/involution_pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'involution',
    'convolution'
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