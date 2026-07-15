# AGENTS.md

## Project purpose

This repository is a scientific research framework.

Its purpose is to enable rapid experimentation with deep learning methods.

Primary goals:

- flexibility
- reproducibility
- modularity
- maintainability

Secondary goals:

- readability
- documentation
- ease of extension

This is NOT production software.

## Refactoring philosophy

Preserve scientific behaviour.

Avoid changing numerical outputs unless explicitly requested.

Avoid large rewrites.

Prefer incremental refactoring.

Explain architectural decisions.

## Notebooks

Jupyter notebooks are valuable research artifacts.

Do not automatically convert notebooks into Python scripts.

Instead:

- extract reusable logic into Python modules
- keep notebooks for exploration and experiments

## Coding style

Prefer explicit code over clever abstractions.

Prefer modularity over inheritance.

Avoid unnecessary dependencies.

Use type hints where practical.

Document public APIs.

## During modernization

Before implementing large changes:

- explain the reasoning
- identify risks
- preserve backward compatibility whenever practical