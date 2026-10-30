"""RouteGate: offline residual-geometry analysis package.

Scope. This package is the offline analysis half of the project: it reads a
frozen, hash-verified activation extraction and asks what can be read out of it.
It never loads a language model, never generates text, and declares no deep
learning runtime as a dependency.

Vocabulary used throughout:

    state 0        the embedding output at the final prompt token
    states 1-48    transformer-block outputs
    E              over-refusal, encoded as y = 1
    B              benign compliance, encoded as y = 0
    H              a third class present in the corpus, not analysed at this stage

Modules:

    validation   input gates -- hashes, counts, stale and held-out refusals
    data         the accepted development corpus, or a refusal to load
    splits       frozen fold recovery and fresh stratified folds
    probes       the scikit-learn probe pipeline and out-of-fold scoring
    metrics      scikit-learn metrics assembled per state
    controls     alternative feature sets for asking what a number is made of
    selection    nested state selection, and the permutation null that prices it
    geometry     descriptive class separation and projections
    plotting     shared figure style
    runrecord    writing an auditable run directory

Causal patching adds four modules. They are the first part of this package that
needs a model, so torch and transformers are imported lazily and are optional
dependencies; nothing above requires them:

    intervention what is applied, where, and how big it is
    hooks        installing, using and removing the activation hooks
    donors       deterministic donor pairing, frozen before any model runs
    runtime      the pinned model, prompt rendering and deterministic generation
    patchrecord  the per-generation output record, and the rules for resuming

Population steering -- the layer-wise causal scan over all 48 transformer
blocks -- adds three more. Only the last of them touches torch:

    ebh          the three-class frame, and one frozen five-fold division of it
    steering     cross-fitted, H-orthogonalised repair directions and their bank
    steerrecord  the steering run's own per-generation record and resume rules

Two more read migrated historical artefacts rather than deriving anything:

    actuator     the frozen RQ2 repair direction, loaded without pickle
    triplets     retrospective validation of the ten operative E/B/H cases

The dual-direction controller stage adds one, and it needs no model either:

    probeheads   the repair and safety probe heads, cross-fitted against the
                 direction bank's frozen 480-row division of the corpus

``steering`` and ``actuator`` are different objects and are kept apart on
purpose. ``steering`` builds class-mean, H-orthogonalised directions from the
480-row corpus and scans them at strength 0.2; that is an ablation. ``actuator``
loads the single B-minus-E direction RQ2 froze from ten matched triplets at
strength 0.15, which is what every reported RQ2 and static-steering result rests
on. Neither substitutes for the other. See ``docs/artifact_taxonomy.md``.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "actuator",
    "controls",
    "data",
    "donors",
    "ebh",
    "geometry",
    "hooks",
    "intervention",
    "metrics",
    "patchrecord",
    "plotting",
    "probeheads",
    "probes",
    "runrecord",
    "runtime",
    "selection",
    "splits",
    "steering",
    "steerrecord",
    "triplets",
    "validation",
]
