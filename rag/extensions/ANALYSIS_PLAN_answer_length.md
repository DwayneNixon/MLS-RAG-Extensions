# Analysis plan: is k = 2 faster because of fewer records or shorter answers?

Commit this file before launching `run_answer_length.sh`. The run script refuses
to start if it is uncommitted or modified, so the commit time precedes the results.

## Why this run

The report recommends k = 2 because it roughly halves generation time with no
detectable loss. On saved data, generation time is about 16-17 ms per answer
token and under 0.1 ms per prompt token (R^2 0.99 on Development, 0.96 on
Practice; `report/report_numbers.py`, follow-up section). So k = 2 is faster
mainly because its answers are shorter (median 140 against 258 tokens on
Development). If k = 5 with short answers is as fast as k = 2, the recommendation
should be "shorter answers", which keeps the evidence k = 2 throws away (k = 2
loses 0.13-0.17 on the queries where it drops every gold record k = 5 had).

## Design

- Split: Development, all 125 queries, cached PyTorch-ASR transcripts.
- Arms, generated in one session and interleaved query by query with a rotating
  order (`generate_length_arms.py`):
  - `k5`: dense top 5, unchanged pipeline (same-session latency reference);
  - `k2`: dense top 2, unchanged pipeline (same-session latency reference);
  - `k5_cap150`: dense top 5, `max_tokens = 150`;
  - `k5_short`: dense top 5, user prompt followed by one line,
    "Answer in at most 100 words."
- Unchanged: encoder, retriever, context builder, system prompt, EuroLLM-22B at
  temperature 0.1, supplied judge at temperature 0.0, one judgement.
- Judged: `k5_cap150` and `k5_short`. The `k5` and `k2` re-runs are generated for
  latency only and are not judged.
- Score: (correctness + grounding + completeness) / 12.

## Exclusion rule (fixed in advance)

Same as the Practice depth plan: a query is excluded from an arm only if the
judge output is unparseable or out of range after 8 requests, twice. Transport
errors never exclude a query. Quality comparisons use the 123 Development
queries scored in every report configuration, minus any excluded here.

## Primary analyses

1. Latency (same session): median `generation_ms` of each arm, with the first
   query dropped as warm-up, and the paired median ratio to the `k2` re-run.
   "Matches k = 2" means the arm's median is within 10% of the `k2` re-run median.
2. Quality: paired difference against the saved Development k = 5 (judge repeat 1),
   as in the report, for `k5_short` and `k5_cap150`; mean, SE = sd / sqrt(n),
   two-sided 95% interval. Also against the saved k = 2.
3. Sub-scores: the same paired differences for correctness, grounding and
   completeness separately (0-4 scale). Completeness is the one a length limit
   is expected to hurt.
4. For `k5_cap150`: the number of answers with `finish_reason = length`
   (truncated), and the score difference among truncated and untruncated answers.

## How each outcome will be reported

| Outcome | Statement |
| --- | --- |
| `k5_short` within 10% of k2 latency, 95% interval for quality vs k5 includes 0 | Shorter answers give k = 2's speed without dropping evidence; recommend a length instruction rather than k = 2. |
| `k5_short` within 10% of k2 latency, quality interval entirely below 0 | Short answers cost quality; report the loss and the trade-off against k = 2's subgroup loss. |
| `k5_short` not within 10% of k2 latency | The length instruction does not shorten answers enough; prompt length or evidence also matters; keep k = 2 with its subgroup caveat. |
| `k5_cap150` heavily truncated | Truncation, not brevity, is measured; report it as such and rely on `k5_short`. |

With 123 queries, only quality differences of about 0.07 or more can be
detected. "No detectable loss" is stated with that limit.

## Not part of this plan

Other caps, other wordings, Practice runs and any subgroup not listed are
exploratory and will be labelled as such. No arm will be added or changed after
seeing scores.
