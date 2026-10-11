# Predictor migration references

These small, pickle-free NPZ fixtures preserve the numerical checks from
`test_jax_torch_parity.py` at commit `2a55a2c`. They were generated on CPU with
PyTorch 2.14.1, `torch.manual_seed(69)` reset for each case, and the input seeds
recorded in `test_predictor_migration.py`. JAX 0.11.2 matched every reference
before the fixtures were saved.

Each model case contains its portable checkpoint metadata and arrays.
`expected_0` stores the original implementation's standardized Rollout. Loss
cases contain the scalar loss. The BPTT case stores loss, both layers' weight
and bias gradients, then the first layer's weights after one AdamW step.

The original implementation and live framework comparison tests remain on
`legacy_pytorch_training`. Main tests these frozen references without installing
the legacy framework.
