# Research: Back-Translation Augmentation

## Goal

Determine whether back-translation (EN→FR→EN for text augmentation) can be
represented as an atom, or whether it should be classified as external_tool.
Target repo: `sciona-atoms-dl` if atom, otherwise reclassify.

## CDG stages this research covers (1 stage)

- `toxic_comment_1st/back_translation_augmentation`: Translate English text
  to French, German, and Spanish, then translate back to English to generate
  paraphrased training examples (MarianMT/Opus-MT)

## What to research

### 1. Is back-translation decomposable?
- The operation is: `text → translate(text, src, tgt) → translate(result, tgt, src)`
- The translation step requires a pretrained neural MT model (MarianMT)
- This is fundamentally model inference, not a mathematical operation

### 2. If opaque: what's the contract?
- `back_translate(text: str, pivot_language: str, model_path: Path) -> str`
- Input: source text + pivot language code
- Output: paraphrased text in source language
- The atom would be opaque (`is_opaque=true`) wrapping MarianMT inference

### 3. If external_tool: why?
- Requires downloading/loading large pretrained models (~300MB per language pair)
- Model loading has side effects (GPU memory, model cache)
- May be better classified as `external_tool` since it invokes an external
  model rather than implementing an algorithm

### 4. MarianMT implementation
- Source: HuggingFace transformers (Apache-2.0)
- Models: Helsinki-NLP/opus-mt-en-fr, opus-mt-fr-en, etc.
- Pure function boundary: loaded model + text → translated text

## Research questions

1. Should this be an opaque atom or external_tool?
   (Recommend: opaque wrapper — it has a clear I/O contract even though
   internals are a neural network)
2. What are the standard pivot languages for English back-translation?
   (French, German, Russian are most common in NLP competition solutions)
3. Are there lightweight alternatives that don't require 300MB models?
   (No — back-translation inherently requires translation models)
4. Can we separate the "translate" atom from "back-translate" composition?
   (Yes — `translate(text, src, tgt, model)` is the atom;
   back-translation is orchestration of two translate calls)

## Output format

```
Name: translate_text
Description: Translate text between languages using a pretrained MarianMT model.
  Opaque pretrained model — internal architecture not decomposed.
Source: URL to HuggingFace MarianMT documentation
License: Apache-2.0 (HuggingFace transformers)
Concept type: neural_network
is_opaque: true
Signature: (text: str, src_lang: str, tgt_lang: str,
            model_path: Path) -> str
Pure function boundary: text string and language codes in, translated string
  out; model must be pre-loaded and passed explicitly — no implicit downloads,
  global model cache, or GPU state management inside the atom.
Contracts:
  - require: len(text) > 0
  - require: src_lang != tgt_lang
  - ensure: len(result) > 0
Witness: shape-only — verify non-empty output for a short input sentence.
  Implementation raises NotImplementedError (opaque).
Dependencies: transformers (Apache-2.0), torch
CDG stages covered: toxic_comment_1st/back_translation_augmentation
```
