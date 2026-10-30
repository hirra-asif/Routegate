# Linear outcome selection

This completed comparison tests whether supervising the outcomes of an edit
improves selection over predicting E/B/H class membership. Both methods choose
between the same saved baseline and fixed-edit replies on 480 development prompts.

The fixed edit uses `d^(1)`, strength 0.15, blocks 31–32. Predictors receive only
the 2,048 unsteered state-32 coordinates at the final prompt token. The
[method specification](METHOD.md) defines the targets, five outer folds,
inner-validation fitting and all three validation-budget profiles.

## Run and inspect

```bash
bash experiments/outcome_selector_v1/run.sh selftest
bash experiments/outcome_selector_v1/run.sh preflight
bash experiments/outcome_selector_v1/run.sh status
```

`run.sh run --output <new-directory>` reproduces the CPU comparison. Inputs
include the completed `controller_error_audit_v1`, frozen residual extraction
and `artifacts/gate_oof_v1/fold_assignments.json`. Existing results are under
`$HOME/routegate-data/outcome_selector_v1/`.

At the middle profile, the outcome selector retains 46 fixed-edit substantive
E answers, with four B material losses and zero newly harmful H responses;
the class selector gives 42, three and one respectively. These are a trade-off,
not an improvement on every endpoint. All profiles must be reported.

E is retained fixed-edit substantive answers /89; B is material loss /178;
newly harmful H is assessed over 212 known non-harmful baselines, with one
unknown baseline separate. E counts are not complete-system repair rates.
The later [capacity comparison](../selector_capacity_v1/README.md) reuses these
two linear methods and adds a small neural network.
