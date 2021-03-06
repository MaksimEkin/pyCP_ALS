from setuptools import setup, find_packages
from glob import glob
__version__ = "0.0.1"

# add readme
with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

# add dependencies
with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

setup(
    name='pyCP_ALS',
    version=__version__,
    author='Maksim E. Eren',
    author_email='meren1@umbc.edu',
    description='Python CP-ALS',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    package_dir={'pyCP_ALS': 'pyCP_ALS'},
    platforms = ["Linux", "Mac", "Windows"],
    include_package_data=True,
    setup_requires=[
        'numpy==1.19.2', 'tqdm', 'sparse', 'scipy'
    ],
    url='https://github.com/MaksimEkin/pyCP_ALS',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.8.5',
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.8.5',
    install_requires=INSTALL_REQUIRES,
    license='License :: BSD3 License',
)
