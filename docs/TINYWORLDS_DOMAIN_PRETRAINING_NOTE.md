# Deferred Note: Domain-Pretraining a TinyWorlds-Scale Model

Status: recorded 2026-07-18 for possible later investigation; not active work.

## Idea

Pretrain a TinyStories-scale causal model on KG-oriented text generated from
many disposable symbolic worlds. The base would learn the domain grammar,
ontology, and proof language; continual evaluation would then measure facts
learned from entirely held-out worlds rather than simultaneous acquisition of
both language and facts. Sharing ontology/templates would test new bindings in
a familiar language; holding them out would additionally test language
transfer.

## Measured Local Budget

The original corpus contains 463,833,291 pinned-tokenizer tokens, equivalent
to 1,811,849 fixed 256-token sequences. The local RTX 4090 trains the current
8-layer, width-256 model at about 165,000–172,000 tokens/second with batch 32;
its 50,257-token embedding makes it 19.70M actual parameters, and batch 64
exceeds GPU memory.

| Exposure | Pure GPU time | Practical wall-time budget |
| --- | ---: | ---: |
| One corpus pass | 45–55 minutes | 1–1.5 hours |
| Three passes | 2.3–2.8 hours | 3–4 hours |
| Five passes | 3.8–4.6 hours | 5–6 hours |
| Roughly 20 passes | 15–18 hours | 18–24 hours |

A disposable 8,000-token-vocabulary benchmark reached 291,000 tokens/second
and 8.93M parameters, cutting pure one-pass time to about 27 minutes. The
original work used a 10,000-token vocabulary and reports sub-day single-GPU
training. Its reported 20-epoch 33M run is a reference exposure, not a required
schedule here.

Current rendering is serial at about 36 stories/second: roughly 14 hours for
an equivalent corpus, or plausibly 1–2 hours after deterministic 16-core
sharding. Exact tokenization took 84 seconds. Pretraining should consume
streaming token shards rather than the larger aligned JSONL representation.

## Conditions for a Meaningful Experiment

- Use an 8,000–16,000-token domain vocabulary; random hash spelling must not be
  the primary learning burden.
- Generate many independent worlds, with benchmark names, facts, proofs, and
  topology excluded by automated overlap audits.
- Remove candidate filler and preserve exact scoring controls; pretraining
  cannot repair a biased evaluator.
- Compare KG-only and mixed ordinary-English/KG corpora.
- Run one pass first, then inspect held-out NLL, candidate accuracy,
  proof-depth generalization, and novel binding before authorizing 3–5 passes.

## Engineering Needed Before Timing Starts

The repository has base-update primitives but no production pretraining
runner. Later work needs deterministic streaming/shuffle, learning-rate
scheduling, validation, checksummed checkpoints, and exact optimizer/RNG
resume. Those engineering and corpus-design costs are outside the GPU times.

References: [TinyStories paper](https://arxiv.org/abs/2305.07759) and the
[model-owner training discussion](https://huggingface.co/roneneldan/TinyStories-33M/discussions/8).
