import os
from setuptools import find_packages, setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='fragvae',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    url='',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    license='Apache 2.0',
    author='John Armitage',
    author_email='jwarmitage@gmail.com',
    description='Fragment Variational autoencoder for use with molecular graphs, as described in  'UPDATE!!!',
    install_requires=['numpy','pandas','tensorflow','rdkit','cairosvg','matplotlib','keyboard','Pillow'],
)
