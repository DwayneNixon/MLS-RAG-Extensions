# Analysis plan: retrieval depth on the Practice split

Commit this file before launching `run_practice_depth.sh`. The commit timestamp is
the evidence that the plan precedes the results. The run script refuses to start
if this file is uncommitted or modified.

## Why this run

On Development (123 paired queries) the depth comparisons have a standard error
of about 0.025, so effects under about 0.07 cannot be resolved. The report
recommends k = 2 on the grounds of "no detectable loss". This run tests that
recommendation on queries that played no part in it.

The Practice split (350 queries) was used to choose the hybrid weight and to
inspect reranker retrieval metrics. No answers were generated or judged on
Practice at any retrieval depth, and depth has no tuned parameter, so Practice
is unseen data for this comparison. [Confirm this is true before committing.]

## Design

- Split: Practice, all 350 queries, cached PyTorch-ASR transcripts.
- Arms: dense k = 5 (reference), k = 2, k = 20. One generation each.
- Pipeline: unchanged encoder, retriever, context builder, generator
  (EuroLLM-22B, temperature 0.1) and judge (temperature 0.0, one repeat).
- Order: k5, k2, k20. Each arm is generated and judged before the next starts.
- Score: (correctness + grounding + completeness) / 12, as in the report.
- Reference: Practice k = 5, single judgement. For Development, k = 5 judge
  repeat 1, as in the report.

## Exclusion rule (fixed in advance)

A query is excluded from an arm only if the judge returns unparseable or
out-of-range output after 8 requests, and again after 8 more on a second
attempt. Connection errors, time-outs and HTTP 429/5xx responses never exclude
a query; the run waits and retries. Excluded IDs are written to
`excluded.jsonl`. A comparison uses the queries scored in both of its arms. If
more than 5 queries would be excluded from one arm, the run stops and the
endpoint is treated as faulty.

## Primary analysis

Non-inferiority of k = 2 against k = 5 on Practice, margin 0.03.

- Statistic: mean paired difference d = score(k=2) - score(k=5), with
  SE = sd / sqrt(n).
- Test: non-inferiority holds if the lower bound of the two-sided 90% interval
  (a one-sided 95% test) is above -0.03.

## Secondary analyses

1. The same test on Practice and Development pooled (Development: the 123
   queries scored in every depth configuration). Development has already been
   seen, so this is supporting, not confirmatory.
2. k = 20 minus k = 5 as a two-sided estimate with a 95% interval, on Practice
   and pooled.

## How each outcome will be reported

| Outcome on Practice | Statement |
| --- | --- |
| Lower 90% bound above -0.03 | Non-inferiority established: k = 2 loses less than 0.03. |
| Interval includes zero, lower bound at or below -0.03 | Consistent with non-inferiority but not established; report the bound actually obtained ("k = 2 loses less than X"). |
| Whole interval below zero | k = 2 is worse; report the size of the loss and qualify the recommendation as a cost-quality trade. |
| Upper 90% bound below -0.03 | k = 2 is inferior by more than the margin; withdraw the recommendation. |

All four are reportable. The Development result will be reported unchanged.

## Expected precision

Scaling the Development standard error (0.024 at n = 123) gives about 0.014 on
Practice and about 0.012 pooled. If the true difference is zero, the chance of
establishing non-inferiority at 0.03 is about two in three on Practice alone and
about 80% pooled. A "not established" outcome is therefore plausible even if
k = 2 is truly no worse.

## Not part of this plan

Subgroup analyses, other margins and any comparison not listed above are
exploratory and will be labelled as such.

## Another note
This was checked by searching the repository and the cluster checkout for
Practice answer files on 11 November 2026; only the README sample was found.