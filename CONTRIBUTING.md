# Contributing

Public redistribution of this staged repository is pending the licensing issue described in [LICENSE.md](LICENSE.md).

For implementation changes, preserve the tensor interfaces and saved topology buffers. Include a focused regression check for rank masking, routing, checkpoint restoration or metrics as appropriate. Run:

```bash
python -m unittest discover -s tests -v
python train.py --help
python test.py --help
python eval.py --help
```

Keep data, weights, credentials, absolute personal paths and generated experiment artifacts out of commits. Report the training seed, data split, checkpoint selection rule, precision, TTA and metric aggregation when discussing results. Do not describe a smoke test as a reproduced paper result.
