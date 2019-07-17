# Regym: Reinforcement learning Framework research framework

Framework to carry out both Single-Agent and Multi-Agent Reinforcement Learning experiments. Developed by PhD heros at the University of York. This framework has been in constant development since December 2018, and will continue to evolve to add new features and algorithms for many more years!

## Features

+ PyTorch implementation of: [DQN](https://arxiv.org/abs/1312.5602),[Double DQN](https://arxiv.org/abs/1509.06461),[Double Dueling DQN](https://arxiv.org/abs/1511.06581),[A2C](https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752),[REINFORCE](https://danielhp95.github.io/policy-gradient-algorithms-a-review),[PPO](https://arxiv.org/abs/1707.06347)...
+ Every implementation is compatible with [OpenAI gym](https://github.com/openai/gym) and [Unity](https://github.com/Unity-Technologies/ml-agents) environments.
+ Self-Play training scheme for Multi-Agent environments, as introduced [here](https://danielhp95.github.io/assets/pdfs/COG-2019-submission.pdf).
+ Emphasis on cross-compatibility and clear interfaces to add new algorithms. See [Adding a new algorithm](docs/adding-a-new-algorithm.md).
+ (In development) Test suite to test and benchmark each algorithm on: compatibility on Discrete / Continuous observation / action spaces. Proof of learning, proof of reproducability.
+ (In development) Parallel actors per algorithm.

## Documentation

All relevant documentation can be found in the [docs](docs/readme.md). Refer to source code for more specific documentation.

## Installation

### Using `pip` 

This project has not yet been uploaded to PyPi. This will change soon!

<!--
This project can be found in [PyPi](LINK TO PYPI project) (Python Package Index). It can be installed via
`pip`:

`pip install regym`
-->

### Installing from source

Firstly, clone this repository:

```
git clone https://github.com/Danielhp95/Generalized-RL-Self-Play-Framework
```

Secondly, install it locally using the `-e` flag in the `pip install` command:
```
cd Generalized-RL-Self-Play-Framework/
pip install -e .
```

### Dependencies

Python dependencies are listed in the file [`setup.py`](./setup.py). This package enforces Python version `3.6` or higher. 

If you would like Python `2.7` or other Python versions `<3.6` to work, feel free to open an issue.

### License

Read [License](LICENSE)

### Papers

List of papers that used this framework.

+ [A Generalized Framework for Self-Play Training](https://danielhp95.github.io/assets/pdfs/COG-2019-submission.pdf)
