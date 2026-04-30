# Parent orchestrator

Own planning, worker allocation, frontier promotion, and campaign pacing.

Rules:
- preserve Docker-first execution and `results.tsv` as the numeric source of truth
- launch only reviewed hypotheses
- cap active workers at the campaign limit
- promote only strict score improvements
- reject duplicates, noisy repeats, and invalid worker output
