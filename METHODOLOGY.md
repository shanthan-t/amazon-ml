# Methodology — V6 Entity Resolution Pipeline

## Entity Resolution Task

The Amazon ML Challenge requires matching business entities from Source 1
against noisy records in Source 2 and Source 3. Each Source 1 entity has a
business name, business address, and country. The task is to identify which
Source 2/Source 3 records refer to the same real-world business.

Evaluation uses **F0.5**, which weights precision twice as heavily as recall.
This motivated a precision-first design throughout the pipeline.

## Candidate Generation

Exhaustive pairwise comparison of ~1.7M × ~10M records is intractable.
The pipeline uses multi-route blocking to reduce each Source 1 entity to a
small candidate set (typically 50–500 candidates).

### Blocking Routes

1. **E (Exact name):** Full normalized name + country. Highest precision.
2. **P6 (6-char prefix):** First 6 characters of compact (whitespace-removed) name + country.
3. **P4 (4-char prefix):** First 4 characters of compact name + country. Broader recall.
4. **ST (Sorted token pairs):** Pairs of significant tokens (≥3 chars, not stopwords), sorted. Handles word reordering.
5. **W (Word):** Individual significant words (≥5 chars) + country.
6. **AN (Address numeric):** Numeric sequences (≥3 digits) from normalized address + country.

Each route has a posting cap of 500 to prevent common keys from generating
enormous candidate lists.

### Rare-Address Overlap Route

For entities with distinctive address tokens, the pipeline selects the two
lowest-frequency eligible address tokens (frequency ≤ 5000 in the target
index). Candidates retrieved through these tokens must satisfy:

- At least 2 shared nonnumeric address tokens, AND
- Token Dice coefficient ≥ 0.55 OR address fuzz ratio ≥ 70

If the eligible pool exceeds 500 candidates, it is discarded entirely for
that entity (the address is not discriminative enough).

Up to **25** qualifying rare-address additions are ranked by address-level
features (conflict penalty, address Jaccard, fuzz ratio, numeric exactness,
name fuzz ratio) and added to the baseline candidates. Baseline candidates
are subtracted before the top-25 cap to maximize the novelty of additions.

## Feature Engineering

Each (S1, candidate) pair produces exactly **39 float32 features**:

- **11 name features:** exact match, token Jaccard, fuzz ratio, 6-char/4-char
  prefix match, token sort ratio, token set ratio, partial ratio, common token
  count, common token ratio, first word match.
- **5 address features:** exact match, token Jaccard, fuzz ratio, token sort
  ratio, numeric Jaccard.
- **7 structural features:** country equality, name/address length ratios and
  gaps, S1/target address missing indicators.
- **3 contextual features:** India indicator, target-is-S2 indicator, short
  name indicator.
- **2 numeric features:** numeric set exact match, numeric conflict (different
  numbers present in both addresses).
- **5 extended features:** route metadata (rare overlap flag, original route
  count, route agreement), name/address token containment, char-trigram
  Jaccard for name and address, script mismatch indicator.
- **3 ICU transliteration features:** After applying `Any-Latin; Latin-ASCII`
  transliteration via ICU 77.1, compute fuzz ratio, token Jaccard, and
  char-trigram Jaccard on the transliterated names.

### Cross-Script Handling

The ICU transliteration features are essential for matching entities across
different writing systems (Devanagari, CJK, Arabic, Cyrillic, etc.). The
`name_script_mismatch` feature detects when query and target use different
Unicode script blocks.

## XGBoost Matching

Four XGBoost gradient-boosted tree models (`B_cross_script` family), each
trained on three of four entity-disjoint development folds. Each model
independently scores every candidate pair.

## Ensemble Scoring

The four model probabilities are averaged arithmetically (float32):

```
final_score = mean(fold0_prob, fold1_prob, fold2_prob, fold3_prob)
```

## Decision Policy

Precision-oriented thresholds motivated by the F0.5 metric:

- **Global threshold:** 0.98 — only candidates with averaged probability ≥ 0.98
  are accepted.
- **Numeric-conflict threshold:** 0.99 — candidates where the query and target
  have conflicting address numbers require an even higher confidence.
- **Maximum matches:** 11 per Source 1 entity.

Accepted matches are ordered by descending score.

## Why Candidate Generation Matters

Without effective candidate generation, the pipeline would need to score
billions of pairs. The multi-route blocking strategy with a rare-address
overlap refinement reduces the average candidate set to a tractable size
while preserving high recall. The top-25 bounded selection provides a
precision/recall tradeoff for the rare-address route.

## Why Precision Was Prioritized

F0.5 = (1 + 0.25) × (precision × recall) / (0.25 × precision + recall).
With a 0.98 threshold, the pipeline strongly favors precision over marginal
recall gains. False positives are penalized ~2× more than false negatives.

## Entity Aggregation

Each Source 1 entity is processed independently. Workers own disjoint S1
ranges and write separate outputs. The merge step concatenates shard outputs
in canonical S1 order and validates the complete contract before publishing.
