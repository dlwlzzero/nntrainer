# The bit-preserving prompt set (plan 141 step 6)

The reusable accuracy set for every bit-preserving lever of the
2026-09-28 direction change (contract `0001` decisions table): a variant
passes when its generated text is byte-identical to A's on all eight
prompts at G = 64. First used by `docs/measurements/141-dspq-moe.md`.

Token counts: the model's own `tokenizer.json`
(`/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/tokenizer.json`,
md5 `7b8067a580173d3eb1697afae3b456f5`, `tokenizers` 0.22.2, no special
tokens added; the device passes the file as `"$(cat <file>)"`, which drops
the final newline and does not change the count). Every prompt is at most
512 tokens (`init_seq_len: 512` until #222, 1024 in the config of record).
`causal_lm.cpp` registers the prefill's token only when the prompt is
shorter than `init_seq_len`, so p01 (512 tokens) printed its text without
the first generated token under 512 and prints it under 1024: across the
two configs compare p01's texts with the new one's first token dropped,
or by the position-indexed `[PPL] decode step=` lines.

| id | file | domain | tokens | md5 |
|---|---|---|---|---|
| p01 | `../77-prompt512.txt` | narrative (the harbour town); a near-tie at decode step 1 (`town` p = 0.134, #134), the most sensitive prompt known | 512 | `fc65c1588dc66dd764c7013fe96cbb75` |
| p02 | `bitset-02-code.txt` | a buggy Python `median`, explain and fix | 207 | `7d2a17d3cc097d0567c9463c5eaeff88` |
| p03 | `bitset-03-math.txt` | multi-step arithmetic word problem, step by step | 117 | `c804b037d3bb87bf09259504fcae1c79` |
| p04 | `bitset-04-korean.txt` | a Korean paragraph (Joseon Hanyang) and three questions | 326 | `c5d4595b859a4e6abc3a035353aac180` |
| p05 | `bitset-05-json.txt` | extract an order note as JSON | 207 | `a21ecdeea9d43d1b1366bde523347a62` |
| p06 | `bitset-06-dialogue.txt` | multi-turn chat, the assistant's next turn | 276 | `6efd387197d6078366152f83a9b72a69` |
| p07 | `bitset-07-facts.txt` | encyclopedic article (the transistor) | 402 | `db0251e1bc57f072f815802cd93c260d` |
| p08 | `bitset-08-short.txt` | "List ten fruits", a short prefill | 24 | `67b657c1261c2c12022907c2117a9a54` |

Two prompt lengths for the LFM2.5 table of record (P64 / P512 / P1024,
#222 / #225; P512 is p01), cut with the same tokenizer at a token
boundary and checked to re-encode to exactly that count:

| id | file | what | tokens | md5 |
|---|---|---|---|---|
| P64 | `p64.txt` | p01's first 64 tokens | 64 | `c0d3e9ffb0c9565e51fa044c53ac71d7` |
| P1024 | `p1024.txt` | p01 + p07 + p05 + p03 joined by a blank line, cut at 1024 tokens (ends inside p05) | 1024 | `2e47c5f45f538babcc7b4a7bb48a4e70` |

#222's sitting made its P64 / P1024 files by the same recipe on the farm
machine; they were not kept, so these two are rebuilt and need not be
byte-identical to them.

## mc-40: the multiple-choice benchmark of `htp_moe_ppl` (plan 194 P3)

`mc-40.tsv` holds 40 four-option general-knowledge, arithmetic and
reading questions written for this repo (id, question, options A-D, the
right letter; 10 answers per letter). `tools/htp/mc_ids.py` renders each as
`<question>\nA. ..\nB. ..\nC. ..\nD. ..\nAnswer:` with the same tokenizer
(no special tokens) and writes `mc-40/qNN.txt` (the rendering without its
last token, `:`) and `mc-40/qNN.ids` (that token, then the right letter's
token: ` A` 334, ` B` 378, ` C` 340, ` D` 388; `mc-40/alts.txt`). One app
run per question, G = 1, `NNTR_PPL_DECODE=qNN.ids
NNTR_PPL_DECODE_ALTS=$(cat alts.txt)`, scores P(letter | rendering);
`tools/htp/mc_score.py` picks the first maximum of the four logits and
sums the right letter's nll. A homemade set: it catches a systematic break
(wrong routing, a dead head), not a 1 % drift, which the decode PPL sees.
