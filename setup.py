from setuptools import setup

setup(
    name='dpyacl',
    version='0.3.1',
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
              'dpyacl.core.misc',
              'dpyacl.experiment',
              'dpyacl.learner',
              'dpyacl.metrics',
              'dpyacl.oracle',
              'dpyacl.scenario',
              'dpyacl.strategies',
              'dpyacl.strategies.single_label'
    ],
    package_dir={
        'dpyacl': 'dpyacl',
        'dpyacl.core': 'dpyacl/core',
        'dpyacl.core.misc': 'dpyacl/core/misc',
        'dpyacl.experiment': 'dpyacl/experiment',
        'dpyacl.learner': 'dpyacl/learner',
        'dpyacl.metrics': 'dpyacl/metrics',
        'dpyacl.oracle': 'dpyacl/oracle',
        'dpyacl.scenario': 'dpyacl/scenario',
        'dpyacl.strategies': 'dpyacl/strategies',
        'dpyacl.strategies.single_label': 'dpyacl/strategies/single_label'
    }
)
