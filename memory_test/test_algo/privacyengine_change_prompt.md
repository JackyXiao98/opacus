# Prompt for Integrating fastDP's Bookkeeping Algorithm into Opacus

## High-Level Goal

Your task is to integrate the "bookkeeping" (also known as "ghost clipping") algorithm from the `fastDP` library into the `opacus` library. The final goal is to enable the script at `/Users/bytedance/Desktop/Github/opacus/memory_test/test_algo/single_experiment.py` to use this new, memory-efficient DP-SGD training method.

This involves analyzing the implementation in `fastDP` and porting the relevant logic into `opacus`, ensuring the new functionality is well-integrated with Opacus's existing `PrivacyEngine` and `DPOptimizer` framework.

## Key Files for Reference

Before you begin, thoroughly analyze the following files to understand the differences between the two implementations:

-   **Source of the Bookkeeping Algorithm (`fastDP`)**:
    -   `/Users/bytedance/Desktop/Github/opacus/fastDP/privacy_engine.py`: Note the `__init__` parameters like `clipping_mode`, `clipping_style`, `origin_params`, and the logic for the "ghost differentiation trick".
    -   `/Users/bytedance/Desktop/Github/opacus/fastDP/autograd_grad_sample.py`: This is the core of the per-sample gradient computation and clipping. Pay close attention to the `add_hooks` function and the custom backward hooks.

-   **Target for Integration (`opacus`)**:
    -   `opacus/privacy_engine.py`: The main entry point for DP in Opacus. This will need to be extended.
    -   `opacus/grad_sample/`: The directory containing Opacus's current gradient sampling mechanism. You will need to integrate the logic from `fastDP`'s `autograd_grad_sample.py` here.
    -   `opacus/optimizers/optimizer.py`: The `DPOptimizer` will need to be adapted to work with the gradients produced by the new method.
    -   `/Users/bytedance/Desktop/Github/opacus/memory_test/test_algo/single_experiment.py`: The script that will ultimately use the new functionality.

## Detailed Implementation Steps

Follow these steps to perform the integration:

### Step 1: Extend `opacus.PrivacyEngine`

Modify `opacus/privacy_engine.py` to add support for bookkeeping.

1.  **Update `PrivacyEngine.__init__`**:
    -   Add the following parameters to the `__init__` method's signature, mirroring `fastDP`:
        -   `clipping_mode: str = "ghost"` (or similar to activate the new logic).
        -   `clipping_style: str = "all-layer"` (to support 'all-layer', 'layer-wise', 'param-wise', 'block-wise').
        -   `origin_params: Optional[List[str]] = None` (for the ghost differentiation trick).
        -   `clipping_fn: str = 'automatic'`
        -   `numerical_stability_constant: float = 1e-6`
        -   `torch_seed_is_fixed: bool = False`
        -   `num_GPUs: int = 1`

2.  **Implement Ghost Differentiation Logic**:
    -   Inside `__init__`, add the logic from `fastDP.PrivacyEngine` that uses `origin_params` to selectively set `param.requires_grad = False` on non-origin parameters. Store the original `requires_grad` state in a new attribute like `param.initially_requires_grad`.

3.  **Update Hook Management**:
    -   The `PrivacyEngine` is responsible for adding hooks to the model. You will need to create a new hook-adding function (e.g., `_add_bookkeeping_hooks`) or modify the existing one to pass the new parameters (`clipping_mode`, `clipping_style`, etc.) to the gradient sampling module.

### Step 2: Integrate `fastDP`'s Gradient Hooking Mechanism

This is the most critical part. You need to port the per-sample gradient computation logic from `fastDP/autograd_grad_sample.py` into the `opacus/grad_sample/` directory.

1.  **Create New Hooking Logic**:
    -   In `opacus/grad_sample/`, create a new file or modify an existing one (like `gsm_base.py`) to contain the hooking logic from `fastDP`.
    -   Implement a new `add_hooks` function that replicates the behavior of the one in `fastDP/autograd_grad_sample.py`. This function should:
        -   Take `clipping_style`, `block_heads`, `clipping_fn`, etc., as arguments.
        -   Register custom backward hooks on supported layers (e.g., `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`).

2.  **Implement Backward Hooks**:
    -   The backward hooks are the core of the algorithm. For each supported layer, the hook should:
        -   Compute per-sample gradients without materializing the entire per-sample gradient tensor.
        -   Perform clipping on-the-fly. This clipping can be done per-parameter, per-layer, or per-block, depending on the `clipping_style`.
        -   Sum the clipped per-sample gradients.
        -   Store the final summed and clipped gradient in a custom attribute on the parameter, such as `param.private_grad`. **Do not** store it in `param.grad`.

### Step 3: Adapt `opacus.DPOptimizer`

Modify `opacus/optimizers/optimizer.py` to work with the pre-computed gradients.

1.  **Update `_create_noisy_clipped_gradient`**:
    -   The existing `DPOptimizer` computes clipped gradients inside its `step` or a helper method. This needs to change.
    -   Modify the method responsible for gradient creation (e.g., `_create_noisy_clipped_gradient` or similar logic inside `step`) to adopt the `fastDP` approach.
    -   The new logic should iterate through the model parameters and:
        -   Check for the existence of the `param.private_grad` attribute.
        -   If it exists, move this value to `param.grad`.
        -   Delete `param.private_grad` to free memory.
        -   The logic for adding noise should be applied *after* this step, but note that `fastDP`'s `privacy_engine` does not add noise itself, it seems to be done in the optimizer step. In Opacus, the optimizer is responsible for adding noise. Ensure that noise is added correctly to the summed, clipped gradients. The `fastDP` approach seems to add noise inside `reduce_gradients_DP_stage_1` for DeepSpeed, but for a standard optimizer, it should be done in the optimizer step before the update. Let's stick to the Opacus convention: the optimizer adds noise.

### Step 4: Update the Experiment Script

Finally, modify `/Users/bytedance/Desktop/Github/opacus/memory_test/test_algo/single_experiment.py` to use the newly integrated algorithm.

1.  **Refactor `run_dpsgd_experiment`**:
    -   This function currently uses `GradSampleModuleFastGradientClipping` and `DPOptimizerFastGradientClipping`. These are likely custom classes for a specific implementation.
    -   Modify the function to use the standard `opacus.PrivacyEngine`.
    -   Instantiate `PrivacyEngine` with the new parameters to enable bookkeeping. For example:
      ```python
      from opacus import PrivacyEngine

      privacy_engine = PrivacyEngine(
          model,
          batch_size=config["batch_size"],
          sample_size=DATASET_SIZE, # You'll need to define this
          epochs=EPOCHS, # And this
          target_epsilon=TARGET_EPSILON, # And this
          clipping_mode='ghost', # Or your chosen name
          clipping_style='all-layer',
          max_grad_norm=1.0,
      )
      privacy_engine.attach(optimizer)
      ```
    -   The training loop should now work with the standard `optimizer.step()` and `optimizer.zero_grad()` calls, as the `PrivacyEngine` has patched the optimizer. The custom logic with `GradSampleModuleFastGradientClipping` should be removed in favor of the `PrivacyEngine` API.

Please generate the necessary code modifications for the files listed above to implement this functionality. Ensure the changes are consistent and follow the architectural patterns of the `opacus` library.