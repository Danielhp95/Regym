Remember to activate virtual environment, if running any of the commands below generates an error about uninstalled dependencies, just install them.


I suggest you run the SAC test which trains in RockPaperScissors against a rockAgent.

```python
  pytest -s -v -k "RPS" regym/tests/rl_algorithms/sac_test.py
```

If SAC was working correctly, the test should pass. The critic is learning, as the Q(s, a) where a is R/P/S grow towards the right values (R:0, P:1, S:-1).

I've set it up so that when SAC tries to update, it prints the gradients of the `nn.Linear` output layer. As you can see, they are all 0. As in, when we compute `loss_pi` and run `loss_pi.backward()`, we see that all the gradients are zero.

Idea:
  - Clearly the issue has to emanate from the fact that taking the derivative of the operation `F.softmax(logits...)` in `CategoricalHead` is yielding zero gradients. But how to fix this?
