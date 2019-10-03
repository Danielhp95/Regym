from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as readme:
    long_description = readme.read()

test_requirements = ['dill', 'tqdm', 'pytest']

setup(
    name='regym',
    version='0.0.1',
    description='Framework to carry out (Multi Agent) Reinforcement Learning experiments. Developed by PhD heros at the University of York.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Danielhp95/Generalized-RL-Self-Play-Framework',
    author='IGGI PhD Programme',
    author_email='danielhp95@gmail.com',

    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: Reinforcement Learning',
        'Programming Language :: Python'
    ],

    packages=find_packages(),
    zip_safe=False,

    install_requires=['gym',
                      'matplotlib',
                      'docopt',
                      'pyyaml',
                      'pip',
                      'tensorboardx',
                      'opencv-python',
                      'torch',
                      'torchvision',
                      'gym-rock-paper-scissors==0.1',
                      'cvxopt',
                      'seaborn'] + test_requirements,

    python_requires=">=3.6",
)
