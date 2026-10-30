# Controller outcome audit

A paired audit of the completed 480-prompt development comparison. It joins
recorded baseline, fixed two-layer repair and learned-strength replies with
their original judgements. It examines gains and losses, fold variation and
identical-response label conflicts.

The fixed edit is `repair_015_blocks31_32`; the learned-strength arm is
`dynamic_raw`. This audit does not refit either method or change any label.

## Read the completed audit

Results are under `$HOME/routegate-data/controller_error_audit_v1/`:

| File | Contents |
| --- | --- |
| `REVIEW.html` | Readable paired cases and breakdowns |
| `summary.json` | Group, fold, source and topic summaries |
| `audit_rows.csv` | One row per development prompt |
| `cases.jsonl` | Responses, outcomes and provenance used by the outcome selectors |
| `input_provenance.json` | Input identity and hashes |

On common resolved prompts, dynamic strength retains 18 fewer substantive E
answers and introduces seven fewer harmful H responses than fixed strength.
These are paired changes on the resolved subsets, not full-inventory rates.

## Reproduce

The wrapper runs its standard-library self-tests and then builds the audit.
Use a new output directory because the completed directory and archive are protected:

```bash
bash experiments/controller_error_audit_v1/run.sh --output "$HOME/routegate-data/controller_error_audit_reproduction"
```

It reads recorded development inputs and makes no generation or judge calls.
The audit identifies patterns in controller errors; it does not establish why a
particular checkpoint or fold performed poorly.
