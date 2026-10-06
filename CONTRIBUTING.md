# Contributing

We welcome contributions! If you have any questions, [open an issue](https://github.com/flechtenberg/pulpo/issues) or [get in touch directly with the `pulpo` developers ](mailto:fabian.lechtenberg@upc.edu)


## Contributing to the code, examples or documentation

If you want to contribute to the development our code with a new feature, want to share your pulpo-example or add to the documentation, please follow the [GitHub contribution workflow (fork, branch, PR)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests) to share your work.

To run the tests, install `pulpo` in editable mode with the `test` extra and run `pytest` from the repository root:

```bash
pip install -e ".[bw25,test]"
python -m pytest tests
```

The [test README](https://github.com/flechtenberg/pulpo/blob/master/tests/README.md) explains how to test both Brightway versions.

## Report bugs or errors

Something is not working as expected? You have two options:

### 🥈 Report an error
Please open a new issue in the `pulpo` [repository](https://github.com/flechtenberg/pulpo/issues), describing the error and where you found it.
A member of the pulpo developer community will then take care of the issue, but it may take some time for your issue to be resolved.

### 🥇  Fix an error yourself
If you have a solution to the error, you can [create a fork](https://github.com/flechtenberg/pulpo/forks) of the `pulpo` repository, make your changes and [create a pull request](https://github.com/flechtenberg/pulpo/pulls). The developers will assess the changes and be eternally grateful!

[code of conduct]: codeofconduct
