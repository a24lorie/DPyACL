from setuptools import setup

setup(
    name='dpyacl',
    version='0.1.0',
    author='Alfredo Lorie',
    author_email='a24lorie@gmail.com',
    description='Distributed Python Active Learning library',
    long_description=open('README.md', encoding='UTF-8').read(),
    long_description_content_type='text/markdown',
    license='GNU',
    url='https://github.com/a24lorie/DPyACL',
    install_requires=['dask', 'dask-ml', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'prettytable'],
    packages=['dpyacl',
              'dpyacl.core',
              'dpyacl.experiment',
              'dpyacl.learner',
              'dpyacl.metrics',
              'dpyacl.oracle',
              'dpyacl.scenario',
              'dpyacl.strategies'
    ],
    package_dir={
        'dpyacl': 'dpyacl',
        'dpyacl.learner': 'dpyacl/core',
        'dpyacl.core': 'dpyacl/experiment',
        'dpyacl.experiment': 'dpyacl/learner',
        'dpyacl.metrics': 'dpyacl/metrics',
        'dpyacl.oracle': 'dpyacl/oracle',
        'dpyacl.scenario': 'dpyacl/scenario',
        'dpyacl.strategies': 'dpyacl/strategies'
    }
)