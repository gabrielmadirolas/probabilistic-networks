## 1. Repository overview

  This is a compact research framework for testing a probabilistic modification of an LSTM on the classic international airline-passengers time series.

  The primary workflow is:

  CSV time series
    → sliding-window tensors
    → custom probabilistic LSTM
    → linear readout
    → MSE training
    → periodic state-dict checkpoints
    → parameter-trajectory inspection

  The research hypothesis is implemented in the custom LSTM: each LSTM gate preactivation receives learned, sampled element-wise noise. The repository is experiment-led rather than
  package-led: the main training and analysis logic lives in notebooks, while models/ contains the reusable neural-network primitives.

  There is no package configuration, dependency lock file, CLI, test suite, or central experiment configuration. The tracked repository contains two active notebooks, two active model
  modules, the dataset, and 21 checkpoints. Git-ignored files preserve older JIT/non-JIT model variants and Jupyter autosaves.

  ## 2. Directory analysis

   Directory             Responsibility
  ━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Repository root       Notebook-driven experiment entry points, architecture notes, and project guidance.
  ────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   data                  The sole input dataset: 144 monthly airline-passenger observations from 1949–1960.
  ────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   models                Reusable PyTorch implementations of the probabilistic activation/noise mechanism and custom LSTM. It is an implicit namespace package—there is no __init__.py.
  ────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   checkpoints           Tracked snapshots from epochs 0 through 10,000 in 500-epoch intervals. Each is a model state_dict, not a complete resumable experiment.
  ────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   .ipynb_checkpoints    Jupyter-generated notebook autosaves; not tracked and should be treated as generated artifacts.
  ────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   models/__pycache__    Generated Python bytecode; not source.
  ────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   .git                  Version history. It shows the current active implementation moved from TorchScript toward torch.compile.

  ARCHITECTURE.md is intentionally only a placeholder. AGENTS.md provides sound modernization constraints: preserve scientific behavior, keep notebooks as research artifacts, and favor
  incremental changes.

  ## 3. Notebook inventory

   Notebook            Purpose                                                                Classification        Reusable code to extract eventually
  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   passengers_LSTM_    Defines and trains the probabilistic LSTM on airline passengers,       Experiment            Dataset loading/windowing, AirModel, training loop, checkpoint
   ProbAct.ipynb       evaluates RMSE, saves checkpoints, and plots predictions. It also                            management, evaluation metrics, and plotting helpers.
                       contains unrelated Python/PyTorch scratch cells.
  ──────────────────  ─────────────────────────────────────────────────────────────────────  ────────────────────  ─────────────────────────────────────────────────────────────────────
   analyze_model.ip    Loads every checkpoint and inspects the learned noise parameters       Exploratory           Checkpoint discovery/loading and parameter-summary utilities.
   ynb                 and selected LSTM weights/biases.
  ──────────────────  ─────────────────────────────────────────────────────────────────────  ────────────────────  ─────────────────────────────────────────────────────────────────────
   .ipynb_checkpoin    Jupyter autosave of the training notebook. Its training cell uses      Obsolete/generated    None; retain only as an autosave if needed.
   ts/                 500 epochs, whereas the active notebook contains the 10,000-epoch
   passengers_LSTM_    run.
   ProbAct-
   checkpoint.ipynb
  ──────────────────  ─────────────────────────────────────────────────────────────────────  ────────────────────  ─────────────────────────────────────────────────────────────────────
   .ipynb_checkpoin    Jupyter autosave of the analysis notebook.                             Obsolete/generated    None.
   ts/
   analyze_model-
   checkpoint.ipynb

  The active training notebook records a CPU run of approximately 608 seconds. Recorded RMSE decreases from 225.75/425.01 (train/test) at epoch 0 to 14.31/71.57 at epoch 10,000. These
  are useful historical results, but they are not reproducible from the saved artifacts alone.

  ## 4. Python modules

  - models/probact.py: active probabilistic-noise implementation.
      - TrainableSigma and TrainableMuSigma are earlier/general experimental activation variants.
      - EWTrainableMuSigma is the active critical class. It learns element-wise sigma, optionally learns element-wise mu, samples Gaussian noise, and returns mu + sigma * ε (or a
        sigmoid-bounded sigma formulation when alpha and beta are nonzero).

      - In the active LSTM path, the input x is used only for shape/device; this class generates additive noise rather than applying a conventional activation to x.

  - models/custom_lstms.py: active custom recurrent implementation.
      - prob_lstm(...) constructs the experimental probabilistic LSTM.
      - my_lstm(...) constructs its deterministic custom-LSTM counterpart.
      - ProbLSTMCell computes conventional LSTM gate affine terms, adds EWTrainableMuSigma noise to all 4 × hidden_size gate values, then applies sigmoid/tanh gate functions.
      - LSTMCell, LSTMLayer, and MyStackedLSTMWithDropout provide the deterministic cell, time unrolling, layer stacking, dropout, batch-axis handling, and zero-state initialization.

  - models/probact_jit.py: ignored historical TorchScript version. Its API differs from the active version: configuration is passed as keyword arguments and its core class inherits
    jit.ScriptModule.

  - models/custom_lstms_jit.py: ignored historical TorchScript custom LSTM. It uses hard-coded probabilistic initialization rather than caller-provided prob_params.
  - models/probact_nojit.py: ignored intermediate variant. Its noise function includes F.relu(x), so it is scientifically different from the active implementation.
  - models/custom_lstms_nojit.py: ignored archival development file. It contains historical bidirectional, layer-normalized, and reference-LSTM test code, and executes several test
    functions at import time. It is not a safe reusable module.

  ## 5. Dependency graph

  passengers_LSTM_ProbAct.ipynb
   ├── pandas / NumPy / matplotlib
   ├── data/airline_passengers.csv
   ├── models.custom_lstms
   │    └── models.probact.EWTrainableMuSigma
   └── PyTorch: compile → Adam → MSE → checkpoint state_dicts

  analyze_model.ipynb
   └── checkpoints/*.pth

  custom_lstms_jit.py / custom_lstms_nojit.py
   └── historical alternatives; not used by active notebooks

  The active notebook imports models.probact directly and imports models.custom_lstms, probact, although the direct EWTrainableMuSigma import is only used in commented code.

  ## 6. Training pipeline

  1. Read data/airline_passengers.csv; select the Passengers column as raw float32.
  2. Split chronologically: first two-thirds (96 points) for training and final third (48 points) for testing.
  3. Create independent sliding-window examples with lookback=4.
      - Training: 92 examples, shape (92, 4, 1).
      - Testing: 44 examples, shape (44, 4, 1).
      - Target is the input shifted by one timestep, so loss is computed across all four predicted positions, not only the final forecast.

  4. Construct AirModel:
      - one-layer probabilistic LSTM;
      - input dimension 1, hidden size 500;
      - zero recurrent and linear dropout;
      - linear 500 → 1 readout.

  5. Compile with torch.compile, select CUDA if available, move full train/test tensors to that device.
  6. Train using shuffled batches of 32, Adam (lr=0.001), and MSE for 10,001 loop iterations.
  7. Every 500 epochs:
      - save model.state_dict() as checkpoints/saved_model_epoch{N}.pth;
      - run full train and test predictions;
      - swap output axes because the custom LSTM returns (sequence, batch, hidden);
      - print RMSE.

  The checkpoints confirm this architecture: one 500-unit layer, 2,000 gate values, and one learned sigma per gate. Since the compiled model is saved directly, state keys carry the
  _orig_mod. prefix.

  ## 7. Evaluation and inference

  There are two distinct evaluation paths:

  - In the training notebook, periodic evaluation computes full-window RMSE on train and test tensors. It then plots only the final output timestep from each window against the
    original series.

  - In the analysis notebook, evaluation means parameter inspection rather than predictive inference: it iterates checkpoint files and reports mu when present, sigma mean/standard
    deviation, plus selected gate weights and biases.

  There is no standalone inference entry point, no held-out forecasting API, no model reconstruction/loading helper, and no uncertainty evaluation protocol. A consumer must recreate
  the exact AirModel, compile-state compatibility, and configuration before calling load_state_dict.

  ## 8. Configuration

  Experiment configuration is entirely embedded in notebook cells:

  - data location: ./data/airline_passengers.csv
  - split: prop_train = 2/3
  - window: lookback = 4
  - model: hidden size 500, one layer, batch-first input
  - optimization: batch size 32, Adam, learning rate 0.001, 10,000 epochs
  - checkpoint cadence: 500 epochs
  - probabilistic parameters: mean_mu, std_mu, mean_sigma, std_sigma, alpha, beta

  There is no typed configuration object, YAML/JSON config, CLI override, run identifier, random seed, environment capture, or checkpointed optimizer/RNG/config state.

  ## 9. Technical debt and fragility

  - Duplicated model implementations: six model files contain substantially overlapping code; variants are not API-compatible and sometimes have different scientific semantics.
  - Notebook as orchestration layer: model definition, data preparation, training, evaluation, checkpointing, and configuration are coupled in one notebook.
  - Global-state coupling: create_dataset reads global train.shape[1] instead of the supplied dataset; AirModel closes over notebook globals.
  - Hard-coded paths and names: relative data/ and checkpoints/ paths; fixed checkpoint prefix; analysis hard-codes _orig_mod.lstm.layers.0... state keys.
  - Weak resumability: checkpoints omit optimizer state, epoch metadata, configuration, random state, metrics, and environment. Resume selection is by file modification time, not epoch
    number.

  - Destructive default: with resume=False, running the notebook deletes all existing checkpoints.
  - Non-determinism: no random seeds are set. Noise is sampled during both training and evaluation; eval() does not disable it. Reported RMSE is therefore stochastic.
  - Device fragility: EWTrainableMuSigma explicitly moves CUDA noise to cuda:0, rather than the tensor’s own device. This breaks multi-GPU/non-default-device use.
  - Unclear parameter constraints: sigma is unconstrained and can become negative; that may be intentional because Gaussian scale sign does not alter its distribution, but it should be
    documented rather than “fixed.”

  - Output-shape surprise: batch_first=True affects input only; output is still sequence-first and every caller must swap axes.
  - Misleading/unused code: unused imports, print statements in constructors, extensive commented alternatives, and notebook scratch exercises obscure the experiment.
  - No regression tests: historical equivalence tests exist only in the ignored archival module and run during import.
  - No dependency declaration: environment recreation is implicit.

  ## 10. Research-critical components to preserve

  These deserve behavior-locking tests before refactoring:

  1. The additive stochastic perturbation of the LSTM’s four gate preactivations in ProbLSTMCell.
  2. The exact prob_params interpretation, including the distinction between fixed scalar mu and learned vector mu, initial distributions, and optional sigmoid-bounded scale.
  3. The location and granularity of learned noise: one parameter per gate coordinate (4 × hidden_size), shared across time and batch.
  4. The custom unrolled LSTM’s initialization, gate order, zero initial states, dropout placement, and sequence-first output convention.
  5. The current raw-data window construction and shifted-sequence loss, including the decision to plot only the final predicted timestep.
  6. Checkpoint compatibility, especially the compiled-model _orig_mod. key format and the existing 21 archived scientific checkpoints.

  ## 11. Incremental modernization roadmap

  1. Document the current baseline. Add a concise architecture/readme section and a run manifest describing the recorded 10,000-epoch experiment, without changing any behavior.
  2. Add characterization tests. Fix seed(s) in tests and verify deterministic-cell equivalence, probabilistic-noise shape/placement, state-dict keys, tensor shapes, and window
     generation.

  3. Extract pure data utilities. Move CSV loading and window construction into a small module; preserve raw scaling, split boundaries, and target semantics exactly.
  4. Extract AirModel and configuration. Create a minimal model module plus a dataclass/config dictionary, retaining the notebook as an experiment front end.
  5. Extract reusable training/evaluation helpers. Centralize device movement, axis conversion, RMSE, prediction plotting, and checkpoint discovery—without changing defaults.
  6. Make checkpointing additive. Introduce an optional new-format checkpoint containing config, epoch, optimizer, RNG, and metrics while retaining an adapter for existing state
     dictionaries.

  7. Stabilize checkpoint analysis. Replace hard-coded state-dict strings with a compatibility-aware inspection utility that supports both compiled and uncompiled keys.
  8. Quarantine historical variants. Move JIT/no-JIT files to a clearly labeled archive/experimental area or document their status; do not delete them until comparison tests establish
     which behaviors matter.

  9. Clarify stochastic evaluation. Add an explicit, optional Monte Carlo prediction/evaluation path and distinguish it from single-sample historical RMSE.
  10. Address portability after baselining. Replace hard-coded cuda:0 allocation with the input/device-aware equivalent, preserving random-number semantics as closely as tests permit.
  11. Add lightweight project metadata. Declare supported Python/PyTorch dependencies and a reproducible environment, but avoid imposing a heavy framework or rewriting the notebook
     workflow.

## Target architecture

```text
prob_project/
├── probact/                         # Lightweight research package
│   ├── activations/
│   │   ├── base.py                  # ProbabilisticActivation protocol/base API
│   │   ├── gaussian.py              # Gaussian additive-noise variants
│   │   ├── parameterizations.py     # Global / element-wise / fixed / trainable scales
│   │   └── constraints.py           # Identity, softplus, bounded-sigmoid transforms
│   ├── models/
│   │   ├── cells.py                 # Standard and probabilistic LSTM cells
│   │   ├── recurrent.py             # Stacked recurrent wrappers
│   │   ├── heads.py                 # Regression/classification readout heads
│   │   └── factories.py             # Architecture construction from config
│   ├── data/
│   │   ├── base.py                  # Dataset/task protocol
│   │   ├── timeseries.py            # Windowing and forecasting datasets
│   │   ├── tabular.py               # Future task-specific datasets
│   │   └── registry.py              # Named dataset builders
│   ├── training/
│   │   ├── loops.py                 # Small, explicit train/eval loops
│   │   ├── metrics.py               # Task metrics
│   │   ├── checkpointing.py         # Backward-compatible checkpoint I/O
│   │   └── reproducibility.py       # Seeding and run metadata
│   ├── experiments/
│   │   ├── config.py                # Dataclass experiment configuration
│   │   ├── runner.py                # Programmatic experiment entry point
│   │   └── analysis.py              # Parameter and uncertainty summaries
│   └── tests/
│       ├── test_activations.py
│       ├── test_cells.py
│       ├── test_data.py
│       └── test_checkpoints.py
├── notebooks/
│   ├── airline_probabilistic_lstm.ipynb
│   ├── activation_ablation.ipynb
│   └── checkpoint_analysis.ipynb
├── configs/
│   ├── airline_lstm_baseline.yaml
│   └── airline_lstm_probact.yaml
├── data/
├── checkpoints/
└── README.md
```

This is intentionally a small research package, not an enterprise training platform. The notebook remains the interactive front end; the package owns reusable scientific logic.

## Core abstraction: probabilistic activations

```python
class ProbabilisticActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        ...
```

A probabilistic activation should be an independently usable `nn.Module`, rather than being embedded only inside an LSTM. This enables direct use in MLPs, CNNs, Transformers, recurrent cells, or custom architectures.

A Gaussian additive formulation could be represented conceptually as:

```text
output = deterministic_component(x) + location + scale * noise
```

where each component is selectable:

| Concern | Examples | Reason |
|---|---|---|
| Deterministic component | identity, ReLU, tanh | Supports the active LSTM behavior and historical variants without copying implementations. |
| Location | fixed zero, global trainable scalar, element-wise trainable vector | Makes mean-shift experiments explicit. |
| Scale | fixed scalar, global trainable scalar, element-wise trainable vector | Directly supports the intended parameterization studies. |
| Scale constraint | unconstrained, absolute, softplus, bounded sigmoid | Separates the scientific decision about variance parameterization from noise sampling. |
| Noise distribution | standard Gaussian initially; extensible later | Keeps the initial framework focused while leaving a clean extension point. |
| Evaluation mode | sampled, deterministic mean, Monte Carlo | Makes stochastic evaluation an explicit experimental choice. |

The key design decision is composition rather than a proliferation of classes such as `GlobalTrainableBoundedGaussianActivation`. A small activation object can receive a location parameterization, a scale parameterization, and a constraint. This makes ablations precise and avoids duplicating sampling code.

## Model architecture

`models/` should contain architecture-specific integration points, not copies of activation mathematics.

```text
ProbabilisticActivation
        │
        ├── ProbabilisticLSTMCell
        │     └── inject into selected gate preactivations
        ├── ProbabilisticMLP
        │     └── insert between linear layers
        └── future architectures
              └── choose explicit insertion points
```

For recurrent models, the LSTM cell should accept an activation/noise module and an explicit injection policy:

```python
ProbabilisticLSTMCell(
    input_size=...,
    hidden_size=...,
    gate_activation=activation,
    inject_into=("input", "forget", "cell", "output"),
)
```

This preserves the current research mechanism—noise added to all four gate preactivations—while allowing meaningful experiments such as perturbing only candidate-cell gates or only recurrent contributions. Injection location must be part of the experiment configuration and checkpoint metadata because it is scientifically consequential.

A model factory can build a baseline PyTorch model or a custom probabilistic model from configuration. This makes baseline-versus-ProbAct comparisons use identical dataset, optimizer, metric, and checkpoint workflows.

## Dataset and task design

Use a minimal dataset-builder protocol:

```python
class TaskData:
    train_loader
    validation_loader
    test_loader
    task_type       # regression, classification, forecasting
    input_shape
    output_shape
```

Each dataset module should return `TaskData` plus task metadata. For example, the airline dataset builder owns chronological splitting, sliding windows, target construction, and plotting coordinates.

This separates task semantics from experiment code. It allows a notebook to switch from airline forecasting to another time-series, tabular regression, or classification task by changing one configuration field, while avoiding a large dataset framework.

The existing shifted-window target should remain as a named forecasting mode, alongside a future final-step-only forecast mode. They should not silently replace one another because they produce different training objectives.

## Configuration and experiment runs

Use nested dataclasses in Python, optionally serialized to YAML:

```text
ExperimentConfig
├── data
│   ├── name
│   ├── split
│   └── task-specific options
├── model
│   ├── architecture
│   ├── hidden size / layers / dropout
│   └── probabilistic injection policy
├── activation
│   ├── distribution
│   ├── location parameterization
│   ├── scale parameterization
│   ├── scale constraint
│   └── evaluation sampling mode
├── optimizer
├── training
└── reproducibility
```

Configuration should be plain and inspectable, rather than relying on a heavyweight configuration framework. A dataclass gives notebooks autocomplete and validation; YAML gives repeatable named runs and easy diffs.

Every run should save its resolved configuration beside checkpoints. This directly addresses the current inability to reconstruct an experiment from model weights alone.

## Training, evaluation, and uncertainty

Keep training loops explicit and task-agnostic:

```text
TaskData + model + optimizer + loss
  → train_epoch()
  → evaluate(sample_mode="sample" | "mean" | "monte_carlo")
  → checkpoint()
```

The existing single-sample behavior should be retained as a valid mode for historical comparison. A Monte Carlo mode should additionally report predictive mean, predictive spread, and uncertainty-aware metrics where appropriate. This is important because probabilistic activations make ordinary one-pass RMSE an incomplete description of behavior.

The framework should not introduce a generic “trainer” hierarchy. A few small functions make scientific assumptions visible in notebooks and are easier to adapt during research.

## Checkpoint architecture

New checkpoints should be structured dictionaries:

```text
{
  "format_version": ...,
  "model_state": ...,
  "optimizer_state": ...,
  "epoch": ...,
  "config": ...,
  "metrics": ...,
  "rng_state": ...,
}
```

A compatibility loader should support the existing bare state dictionaries and normalize compiled-model prefixes such as `_orig_mod.`. This preserves the current 21 checkpoints while allowing future experiments to be resumed and interpreted reliably.

Checkpoint analysis should operate on model objects or normalized state dictionaries, not hard-coded parameter strings. It can then report any activation’s location/scale statistics consistently across architectures.

## Notebook role

Notebooks should remain first-class experiment records:

```text
notebook
  → select/load config
  → construct dataset and model
  → run experiment helpers
  → inspect plots, checkpoints, and ablations
```

They should retain narrative, plots, exploratory diagnostics, and one-off hypothesis tests. Reusable functions should move into the package only after they have stabilized.

This retains the repository’s research workflow while preventing notebook cell order, globals, and scratch code from becoming hidden framework requirements.

## Testing boundary

Tests should protect scientific invariants, not over-constrain exploration:

- deterministic custom LSTM matches an equivalent reference LSTM where expected;
- each activation parameterization has the intended parameter shape and trainability;
- sampled output shape/device behavior is correct;
- fixed seeds give controlled regression behavior;
- data-window and split semantics remain stable;
- legacy checkpoints load successfully;
- baseline and probabilistic model factories produce compatible output shapes.

The most important target is a framework where a new experiment means selecting a dataset, architecture, activation parameterization, and evaluation mode—not copying an LSTM implementation or rewriting a training notebook.
