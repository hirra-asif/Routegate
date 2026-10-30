# Five objects, and why they are not each other

Four of the five things below get called "the direction" somewhere in this
project's history, and two of them get called "the pairs". They are different
artefacts, built at different times, from different data, by different rules, for
different claims. Confusing any two of them produces a statement that is wrong in
a way no test can catch, because every individual number in it is real.

Recorded 2026-08-21, at the point the frozen RQ2 actuator was migrated into this
repository. Every hash below was re-verified on that date.

## The table

| | **1. Matched triplets** | **2. Frozen RQ2 actuator** | **3. 89 E–B donor table** | **4. Unpaired E/B/H corpus** | **5. Population direction bank** |
| --- | --- | --- | --- | --- | --- |
| **what it is** | 10 hand-matched E/B/H prompt triples | one global unit direction, 3 layers × 2048 | a metadata pairing of corpus rows | the labelled development corpus | 5 folds × 49 states of cross-fitted directions |
| **where** | `SteerMoE-main:data/pathway_probe_cases.json` | `frozen/rq2_actuator/` in the data root | `outputs/causal_repair/donor_pairs.csv` | `frozen/corpus/` + `frozen/residual_extraction/` | `outputs/causal_repair/orthogonalized_population_scan/direction_bank/` |
| **identifying digest** | `60905eb1…4a03bad0` | `ee23afd9…5bc8ea` (archive) | `4049896e…5c986ac4f3` | `09434496…353e1d08` (corpus) | `6e945d73…329d570c` (manifest) |
| **built** | 2026-07-02 → 2026-07-06 | 2026-07-21 | 2026-08-06 | 2026-08-04 | 2026-08-07 |
| **paired?** | **yes**, 1:1, by hand | **yes** — it is a mean of per-pair deltas | **no** — 89 E targets share 52 B donors, max reuse 5 | **no**, by design | **no** — class means |
| **H arm?** | **yes** | **no** | **no** | yes, as a class | **yes** — as the orthogonalisation term |
| **estimator** | n/a (prompts) | mean of unit(B − E) per pair | n/a (a pairing) | n/a (a corpus) | `−unit((Ē−B̄) − proj onto (H̄−B̄))` |
| **depth** | n/a | decoder blocks 31/32/33 = states 32/33/34 | n/a | 49 stored states | all 48 blocks, searched |
| **strength** | n/a | **0.15**, frozen | n/a | n/a | **0.2**, declared for the scan |
| **standing** | development source set | **the actuator every reported RQ2 and static-steering result rests on** | input to a confirmatory patching stage | the analysis corpus | **an ablation**, scan incomplete |

## Where each one is allowed to appear in a claim

**1. The original matched triplets.** 17 case records; 16 carry all three arms
(`gta_car_vs_real_car` has no donor); the 10 tiered cases are the operative set
and all 10 are complete. Each also carries two later independent donors, so each
is really a five-tuple: E, H, and B donors `d1_historical`, `d2_independent`,
`d3_independent`. Matched **manually and semantically** — hold the task fixed,
move only the safety axis — never by metadata or embedding. The E arms are
prompts the model actually over-refused in a judged headroom run; the donors were
revised until the model actually answered them.

They are development data. They are **not** in the 480-row corpus: the 50 prompt
hashes of the frozen ten intersect the corpus's 480 in **zero** places, and the
70 hashes of the full bank likewise. That disjointness is a property worth
stating whenever the actuator is described, because it is what makes the actuator
donor-free with respect to the corpus it is applied to.

**2. The frozen RQ2 actuator.** `consensus23_global_frozen`, fingerprint
`4855cc95…9b1dbc2c`, primary; `historical_d1_global_frozen`, fingerprint
`404ce498…0f477058`, comparator. Constructed from (1): per-pair `donor − target`
deltas, unit-normalised per pair and per layer, averaged over the two independent
donors within a case, renormalised, averaged over the ten cases, renormalised.
**No class mean is taken anywhere in it, and no H term enters it at any point.**
Applied additively at the decoder block's *output*, at the final prompt token,
during prefill only, with a target-relative L2 budget read from the block's
*input*. Strength 0.15.

This is the only actuator any reported result rests on. See
`manifests/rq2_actuator.json` and `src/routegate/actuator.py`.

**3. The 89-row E–B donor table.** Built here, on 2026-08-06, by a deterministic
metadata ladder over the accepted corpus: `source_family` → `topic_standardized`
→ `task_frame` → prompt length in characters, 77 exact and 12 by fallback. Donor
reuse is permitted and counted. It has **no H arm**, it is **not a bijection**,
and it is **not the original triplets** — it is an input to the confirmatory
matched-donor patching stage, which has not run.

The failure mode this row exists to prevent: reading "89 E–B pairs" as though it
were the matched triplet set, and concluding the actuator was estimated from 89
corpus pairs. It was estimated from ten hand-built ones whose prompts are not in
the corpus at all.

**4. The unpaired E/B/H corpus.** E 89 / B 178 / H 213 = 480 analysis-eligible
rows of 509, with 480 unique `row_id`s and 480 unique prompt hashes and no row in
two classes. **Unpaired by design, and correctly so.** A discriminative probe
consumes rows independently; a difference of class means is a difference of two
independently estimated means. Pairing would discard 89 B and 124 H rows and
would select on the covariates the source control exists to measure. No
row-level E/B/H matching is to be constructed for these 480 rows.

**5. The population direction bank.** Class means over the corpus, per fold and
per state, with the over-refusal signature orthogonalised against the
justified-refusal signature and negated. Cross-fitted: a fold's direction never
sees the rows it will be applied to. Declared operating strength 0.2 — carried
forward from the pilot so that one axis is fixed while the block axis is searched,
**not a measured optimum**. Its scan is 113 generations of 2 416 and has selected
nothing.

It is an ablation. It does not replace the frozen actuator, it is not "the
actuator at a different layer", and a number from it must never be reported as an
actuator result.

## The four sentences that must not be written

1. "The frozen actuator applies at strength 0.2." — 0.2 is object 5's scan
   strength. The actuator is 0.15.
2. "The actuator direction is orthogonalised against H." — object 5 is. The
   actuator is not; it has no H term at all.
3. "The actuator was estimated from 89 matched E–B pairs." — object 3 is 89
   pairs, has no H arm, and was built a month and a half later. The actuator came
   from ten triplets.
4. "The 480-case dataset needs pairing." — object 4 is unpaired by design, and
   pairing it would be a methodological error, not a repair.

## Related records

- `manifests/rq2_actuator.json` — the actuator's gate; hashes, archive spec,
  variant table, layer map, intervention semantics.
- `manifests/frozen_inputs.json` — the six frozen inputs behind objects 4 and 5.
- `docs/data_lineage.md` — how object 4's labels were reached.
- `docs/decision_log.md`, entries of 2026-08-21 — the strength correction, what
  the RQ1 grid actually ranked, and why the `.pt` is kept rather than converted.
- `docs/open_questions_pre_controller.md` — what is deliberately still open.
