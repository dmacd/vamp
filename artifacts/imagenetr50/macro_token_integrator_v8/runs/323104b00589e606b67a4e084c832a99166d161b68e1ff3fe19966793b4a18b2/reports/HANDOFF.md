# GPT Pro handoff: ImageNet-R macro-token ceiling

The run is complete and ungated. The selected macro-token model changed locked-test accuracy relative to the data-matched v6 MLP by +2.123 points at stage 31 and +2.061 points at stage 50. Its signed differences from stage-matched joint IID were -5.171 and -0.978 points. At stage 50 it differed from local E²-LoRA by -5.106 points.

The selected architecture is depth 1 at learning rate 0.0003. Start with `stage_summary.csv`, then inspect `owner_diagnostics.csv`, `clean_candidates.csv`, and `resource_accounting.csv`. The central scientific questions are whether macro tokens consistently beat the data-matched v6 MLP, how much gap remains to stage-matched joint IID and the true-node oracle, and whether the frozen versus end-to-end owner results indicate missing owner information or failure of the class objective to use it. Do not treat the published or local E2-LoRA values as protocol gates or claim SOTA from this two-stage ceiling study.
