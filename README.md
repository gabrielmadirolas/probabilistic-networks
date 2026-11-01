# Probabilistic networks

This repository is a small research project around a custom PyTorch LSTM with probabilistic gate offsets. The current training example forecasts monthly airline passenger counts from `data/airline_passengers.csv`.

## Run the training notebook in Google Colab

The Colab notebook is [`colab_train.ipynb`](colab_train.ipynb). It clones the model code and CSV from the `development` branch, uses whichever device Colab assigned, and saves checkpoints to Google Drive so a later runtime can resume training.

### 1. Push the development branch once

Colab can only access branches that exist on GitHub. From the repository on your computer, save the notebook and guide in a commit, then push the branch:

```bash
git add README.md .gitignore colab_train.ipynb analyze_model.ipynb
git commit -m "Add Colab training workflow"
git push -u origin development
```

This publishes the current local `development` branch. The `main` branch remains the original baseline.

### 2. Open the notebook

After the push completes, open [the Colab training notebook](https://colab.research.google.com/github/gabrielmadirolas/probabilistic-networks/blob/development/colab_train.ipynb). Sign in to Colab if prompted.

### 3. Choose a GPU if one is offered

In Colab, choose **Runtime → Change runtime type → GPU**, then save. The GPU model and availability vary; the notebook prints the device it receives. If no GPU is available, it falls back to CPU.

### 4. Run the notebook

Choose **Runtime → Run all**. The first setup cell clones only `models/` and `data/` from the branch, avoiding the repository's historical checkpoint files. It then asks you to connect Google Drive. Approve that prompt if you want checkpoints saved there.

The standard Colab runtime already includes the libraries used here; the notebook does not reinstall PyTorch, which avoids replacing Colab's CUDA-enabled build.

### Resuming or starting another run

- The first run starts at epoch 0 and saves model weights every 500 epochs under `My Drive/probabilistic-networks/colab_runs/airline_probact_run_01`.
- If Colab disconnects, reopen the notebook, keep the same `RUN_NAME`, and run all cells again. It resumes from the most recent saved model and optimizer state.
- To train longer, increase `TOTAL_EPOCHS` and keep the same `RUN_NAME`.
- To start a separate run, change `RUN_NAME`.
- To restart the same run from scratch, set `RESUME_TRAINING=False`. Before training, the notebook deletes that run's saved model checkpoints, latest training state, and temporary checkpoint files from its run directory. Keep it `True` to resume.
- `USE_TORCH_COMPILE` is off by default for a simpler first run. It can be enabled later if you want to experiment with compilation.

In `passengers_LSTM_ProbAct.ipynb`, `resume=True` continues from the latest file matching `saved_models_path`; `resume=False` deletes those matching model files and starts a fresh run.

The saved model files can be analyzed separately; `latest_training_state.pt` also stores the optimizer state used for resuming.

## Repository contents

- `models/custom_lstms.py` implements the custom LSTM cells and stacked recurrent layers.
- `models/probact.py` implements trainable mean and scale parameters for Gaussian gate noise.
- `passengers_LSTM_ProbAct.ipynb` is the original research notebook, including exploratory Python cells.
- `colab_train.ipynb` is the focused, resumable Colab training path.
- `analyze_model.ipynb` inspects both the original checkpoints and the checkpoints produced in the Colab Drive folder. When run in Colab, it mounts Drive and reads `airline_probact_run_01` by default.

## Experiment controls

Both training notebooks expose the recurrent architecture and probabilistic
settings near the model configuration cell. `hidden_size`, `n_LSTM`,
`bidirectional`, `proj_size`, `dropout`, and `batch_first` control the custom
LSTM architecture. `proj_size=0` disables projection; otherwise it must be
smaller than `hidden_size`. The model accepts optional `(h_0, c_0)` states and
returns `(output, (h_n, c_n))`; with `batch_first=True`, output is
`[batch, sequence, features]`.

The probabilistic controls are:

- `probabilistic_sharing="per_node"`: one mean/scale per LSTM gate and hidden
  node, independently for each layer and direction (the current default).
- `probabilistic_sharing="per_layer"`: one learned mean/scale shared by all
  gates, nodes, and directions in each layer.
- `probabilistic_sharing="per_network"`: one learned mean/scale shared across
  all recurrent layers and directions.
- `noise_distribution="normal"` or `"uniform"`: choose Gaussian or
  variance-matched uniform noise.
- `noise_location="pre_activation"` or `"post_activation"`: perturb the gate
  values before or after sigmoid/tanh. Post-activation noise can move sigmoid
  gate values outside `[0, 1]`, which is useful to test but changes standard
  LSTM gate semantics.
- `sample_mode="train_only"`, `"always"`, or `"never"`: control whether random
  samples are used during evaluation. The default keeps evaluation
  deterministic; use `"always"` for Monte Carlo evaluation.
- `train_mu` and `train_sigma`: control whether the mean offset and scale are
  learned. `prob_params` continues to set their initialization, including
  `mean_mu`, `std_mu`, `mean_sigma`, and `std_sigma`.
- Learned scales use softplus so they remain positive and can still move when
  initialized at zero. Legacy model weights are converted automatically when
  loaded. Resuming a checkpoint created before this scale change starts with a
  fresh optimizer, because its saved optimizer moments use the old parameter
  coordinates.

The non-probabilistic `custom_lstms.my_lstm(...)` factory supports the same
layer, direction, projection, and state conventions for a baseline. The custom
Python time-step loop makes gate experiments possible but can be slower than
PyTorch's fused LSTM kernels on long sequences; the short sequences in the
airline notebook keep that tradeoff modest. `torch.compile` remains an optional
experiment, not a requirement.

Colab's free compute is subject to changing resource limits, and GPU access is not guaranteed. Runtime files are temporary, which is why this training notebook stores checkpoints in Drive. See [Colab's FAQ](https://research.google.com/colaboratory/faq.html) for current details.
