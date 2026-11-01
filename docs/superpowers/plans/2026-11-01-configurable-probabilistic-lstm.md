# Configurable Probabilistic LSTM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the custom LSTM into a configurable research model for probabilistic gate perturbations and parameter sharing.

**Architecture:** Keep explicit custom cell/layer/stack modules; move noise sampling and trainable statistics into a reusable probabilistic module. Match standard recurrent-model input/output/state conventions and retain checkpoint compatibility.

**Tech Stack:** Python, PyTorch, Jupyter notebooks.

**Spec:** `docs/superpowers/specs/2026-11-01-configurable-probabilistic-lstm-design.md`

## Global Constraints

- No new dependencies.
- Keep notebook defaults at Gaussian pre-gate noise, per-node scale, zero fixed mean.
- Preserve legacy checkpoint loading and existing Colab workflow.
- The Python time-step loop remains a known performance tradeoff for custom gate access.

## Review Focus

- Invalid dimensions/options must fail with clear errors.
- Per-node, per-layer, and network sharing must produce the promised parameter count and sharing behavior.
- State tensors must match dtype, device, batch layout, direction count, and projection dimensions.
- Evaluation sampling policies must be deterministic or stochastic exactly as configured.
- Legacy state dictionaries and notebook checkpoint formats must remain readable.

### Task 1: Probabilistic activation configuration

**Files:** Modify `models/probact.py`.

**Interfaces:** Add `ProbabilisticActivation`; retain `EWTrainableMuSigma(num_parameters, prob_params)` as a compatibility wrapper.

- [ ] Implement configurable Gaussian perturbation, sharing shapes, trainable/fixed mean and scale, injection point, evaluation sampling policy, and clear option validation.
- [ ] Keep the old constructor and state keys loadable.
- [ ] Commit `feat: add configurable probabilistic activation module`.

### Task 2: Recurrent model architecture

**Files:** Modify `models/custom_lstms.py`.

**Interfaces:** Keep `prob_lstm(...)` and `my_lstm(...)`; add activation configuration, sharing scope, optional initial states, bidirectionality, and projection options. Return `(output, (h_n, c_n))` using PyTorch-like shapes.

- [ ] Simplify factory/module construction and validate dimensions/options.
- [ ] Use dtype/device-matched zero states, standard LSTM parameter initialization, and avoid ineffective one-layer dropout warnings.
- [ ] Preserve legacy cell parameter names where possible.
- [ ] Commit `feat: make custom LSTM architecture configurable`.

### Task 3: Notebook and user guide updates

**Files:** Modify `passengers_LSTM_ProbAct.ipynb`, `colab_train.ipynb`, `analyze_model.ipynb`, and `README.md`.

- [ ] Update both training examples to use batch-first outputs and expose architecture/probability controls.
- [ ] Keep Colab resume paths and model checkpoint compatibility.
- [ ] Document parameter sharing, sampling policy, architecture options, and the custom loop's performance tradeoff.
- [ ] Commit `docs: explain configurable probabilistic LSTM experiments`.

## Verification

- Static-compile the changed Python modules.
- Parse every changed notebook as JSON and inspect changed code cells for consistent shapes and configuration.
- Review the final diff, checkpoint keys, and git status before committing and pushing `development`.
