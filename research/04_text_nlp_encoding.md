# Research: Text Processing & NLP Encoding Atoms

## Goal

Find best-in-class, pure-function implementations for text preprocessing,
tokenization, and NLP encoding primitives. Target repos: `sciona-atoms-ml`
(classical text) and `sciona-atoms-dl` (transformer-related).

## CDG stages this research covers (~25 stages)

- Text cleaning / normalization (Jigsaw Toxicity, Mercari, Avito)
- Tokenization — BPE/WordPiece/SentencePiece wrappers (Chaii QA, CommonLit, many NLP CDGs)
- Character-to-token offset mapping (Feedback Prize Writing)
- BIO tag decoding / span extraction (Feedback Prize, Kaggle NER)
- String similarity features — Levenshtein, Jaro-Winkler (Foursquare Location)
- Beam search decoding (BMS Molecular Translation)
- Text back-translation augmentation concept (Jigsaw Toxicity)
- Feature hashing for text (Outbrain Click Prediction)
- Regex-based text filtering (Make Data Count)
- Grapheme-to-phoneme conversion (Alice Lyric Alignment)
- Length constraint filtering on predicted spans (Feedback Prize)

## What to research

### 1. Text cleaning utilities
- URL removal, HTML stripping, unicode normalization
- Punctuation isolation, lowercasing, whitespace normalization
- Spelling correction (simple dictionary lookup, not ML-based)
- Pure functions: `clean_text(text: str, operations: list[str]) -> str`
- Source: standard regex patterns from Kaggle competition kernels

### 2. String similarity functions
- Levenshtein distance: `levenshtein(s1: str, s2: str) -> int`
- Jaro-Winkler similarity: `jaro_winkler(s1: str, s2: str) -> float`
- Jaccard similarity on character n-grams
- Source: python-Levenshtein (GPL — check alternatives), jellyfish (BSD), or pure Python
- Note: prioritize BSD/MIT licensed implementations

### 3. BIO tag decoding
- Convert BIO/BILOU tag sequences to span tuples
- `bio_decode(tags: list[str], tokens: list[str]) -> list[tuple[str, int, int]]`
- Standard NER post-processing
- Source: seqeval library or pure Python implementation

### 4. Character-to-token offset mapping
- Map character-level annotations to token-level positions
- `char_to_token_offsets(char_spans: list[tuple[int,int]], offset_mapping: list[tuple[int,int]]) -> list[tuple[int,int]]`
- Uses HuggingFace tokenizer offset_mapping output

### 5. Beam search decoding
- Generic beam search over vocabulary
- `beam_search(log_probs_fn, start_token, end_token, beam_width, max_length) -> list[tuple[list[int], float]]`
- Pure function taking a scoring callable, returning top-k sequences
- Source: OpenNMT, fairseq, or custom implementations

### 6. Feature hashing (text)
- Murmurhash-based feature hashing for high-cardinality categoricals
- `feature_hash(tokens: list[str], n_features: int) -> NDArray[sparse]`
- Source: sklearn HashingVectorizer internals (BSD)
- Note: we have `hashing_vectorizer_token` as a partial match — research whether
  a higher-level atom wrapping the per-document hashing is needed

### 7. Span filtering by length
- `filter_spans_by_length(spans: list[tuple], min_lengths: dict[str, int]) -> list[tuple]`
- Remove predicted text spans shorter than class-specific thresholds

### 8. N-gram extraction
- Character and word n-gram generation
- `char_ngrams(text: str, n: int) -> list[str]`
- `word_ngrams(tokens: list[str], n: int) -> list[tuple[str,...]]`

## Research questions

1. What are the pure Python / numpy implementations?
   (No spaCy, no NLTK heavy deps — lightweight is better)
2. For string similarity: what's the fastest pure Python Levenshtein?
   (C extensions OK if BSD licensed)
3. For beam search: what's the cleanest numpy-friendly implementation?
   (Must be agnostic to model framework)
4. What contracts are natural? (Levenshtein >= 0, Jaro-Winkler in [0,1],
   BIO tags valid sequence)
5. For tokenization wrappers: should we wrap HuggingFace tokenizers or
   implement BPE from scratch? (Recommend: thin wrapper with explicit I/O)

## Output format

For each candidate atom, provide:
```
Name: bio_decode
Description: Convert a BIO tag sequence and token sequence into labeled spans.
Source: URL to the best reference implementation, paper, or library source
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: data_extraction, searching, data_assembly, or analysis as appropriate
Signature: (tags: list[str], tokens: list[str]) -> list[tuple[str, int, int]]
Pure function boundary: strings, lists, arrays, and explicit parameters in,
  decoded text artifacts or numeric features out; no network calls, model state,
  global RNG, or file I/O.
Contracts:
  - require: len(tags) == len(tokens)
  - require: tags use a documented scheme such as BIO or BILOU
  - ensure: returned span indices are within the token sequence
Witness: short token/tag sequence with one multi-token entity and one singleton
  entity; verify returned labels and offsets.
Dependencies: pure Python or numpy preferred; lightweight BSD/MIT text libraries
  acceptable when they materially improve correctness
CDG stages covered: feedback_prize/bio_decode, foursquare/string_similarity, ...
```
