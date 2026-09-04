# Contributing to OpenModels

We appreciate your interest in contributing to OpenModels and welcome contributions from everyone. Whether you have bug reports, ideas for new features, or want to improve the code, your contributions are valuable. Check the [open issues](https://github.com/Gnpd/openmodels/issues) to see what we're working on.

OpenModels is a project designed for production code, focusing on maintaining high-quality standards. Please adhere to the following guidelines when contributing:

## Project Functionality

- **Unit Testing:** Every function in OpenModels must be accompanied by thorough unit tests. This ensures the reliability and robustness of the codebase.

We welcome your contributions and appreciate your commitment to maintaining the quality of OpenModels. Please follow the guidelines outlined below to ensure a smooth and effective contribution process.

## Reporting Bugs

If you find a bug or unexpected behavior, please open a detailed issue on the GitHub repository. Include error messages and steps to reproduce, along with sample code or data if possible.

## Suggesting Enhancements

For new features or enhancements, open an issue with a detailed description and benefits. Include example code or use cases to illustrate how the feature would be used.

## Submitting Changes

To contribute code changes:

1. Open an issue describing the changes you want to implement.
2. We'll review and discuss the issue to determine the scope.
3. Once agreed, create a new branch for your contribution (we follow trunk-based development).
4. Write code and tests, adhering to OpenModels coding style and conventions.
5. Run tests using the provided framework to ensure your changes don't introduce errors.
6. Submit a pull request with a detailed description and the problem your changes solve.

We'll review promptly and provide feedback. If changes are requested, make them quickly to keep the process moving.

## Code Style

Follow OpenModels coding style:

- Indent with four spaces
- Use descriptive variable names
- Avoid magic numbers or hard-coded strings
- Format code using [Black](https://black.readthedocs.io/en/stable/)

Before submitting your changes, please ensure your code passes formatting, linting, and type checks by running. The project utilizes [Taskfile](https://taskfile.dev/) as a task runner to automate and standardize development flows.

```bash
task lint
task format
task type-check
task test 
```
For convenience: 

```bash
task check # lint, format and type check
task test
```

## Test Coverage

Ensure your changes don't reduce OpenModels's test coverage. Check locally with:

```bash
task test:cov
```

## Documentation

Update OpenModels documentation if your changes affect the API or functionality.

## License

By contributing to OpenModels, you agree that your contributions will be licensed under its [MIT license](https://github.com/Gnpd/openmodels/blob/main/LICENSE).
