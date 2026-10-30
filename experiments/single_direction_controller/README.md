# Learned-strength steering

The controller reads unsteered state 32 and predicts one bounded coefficient
`g_R(x)`. It applies the historical `d^(1)` repair direction at strength
`0.15 g_R(x)` at zero-based blocks 31–32. Both edited sites use the same
coefficient. The language model's weights remain fixed.

Development generations and judging are complete. The
[recorded arm protocol](DYNAMIC_ARMS.md) preserves the original implementation
and execution details. The [controller outcome audit](../controller_error_audit_v1/README.md)
compares saved fixed and dynamic replies on the same prompts.

The executed final system uses an initial unsteered prompt pass to read the
coefficient, followed by a new prompt pass and generation. The intervention
acts on the final prompt token during prefill. This is learned edit strength;
the separate final RouteGate gate makes a binary decision about a different
fixed three-layer edit.

The two-site controller and its feature timing must be described as implemented.
No learned layer-selection policy is established by these experiments.
