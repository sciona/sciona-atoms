# Research: Audio & Speech Processing Atoms

## Goal

Find best-in-class, pure-function implementations for audio preprocessing,
feature extraction, and alignment primitives. Target repo: `sciona-atoms-signal`.

## CDG stages this research covers (~16 stages)

- Audio resampling to target sample rate (Bengali Speech, BirdCLEF)
- Volume normalization (Bengali Speech)
- Mel spectrogram generation (BirdCLEF, DCASE Sound Event Detection)
- MFCC extraction (BirdCLEF)
- Log-mel spectrogram (standard audio feature)
- Source separation / vocal isolation (Alice Lyric Alignment — HTDemucs concept)
- Grapheme-to-phoneme conversion (Alice Lyric Alignment)
- Forced alignment / DTW for audio-text alignment (Alice Lyric Alignment)
- CTC decoding — greedy and beam search (Bengali Speech, ASL Fingerspelling)
- Median filter for frame-level prediction smoothing (DCASE SED)
- Audio chunking / windowing (Cornell Birdcall, BirdCLEF)
- Spectrogram augmentation (SpecAugment — time/frequency masking)

## What to research

### 1. Audio resampling
- Polyphase resampling (scipy.signal.resample_poly) or sinc interpolation
- `resample_audio(signal: NDArray, orig_sr: int, target_sr: int) -> NDArray`
- Source: librosa.resample internals (ISC license), scipy (BSD)
- Key: anti-aliasing filter before downsampling

### 2. Mel spectrogram
- STFT -> power spectrum -> mel filterbank -> log scaling
- Break into composable atoms:
  - `stft_magnitude(signal, n_fft, hop_length, window) -> NDArray`
  - `mel_filterbank(n_mels, n_fft, sr, fmin, fmax) -> NDArray`
  - `apply_mel_filterbank(power_spectrum, mel_fb) -> NDArray`
  - `log_mel(mel_spectrum, ref, amin, top_db) -> NDArray`
- Source: librosa (ISC), torchaudio (BSD)
- Pure numpy: FFT via np.fft, mel filters via analytical formula

### 3. MFCC
- Mel spectrogram -> DCT -> first N coefficients
- `mfcc(log_mel_spectrum: NDArray, n_mfcc: int) -> NDArray`
- DCT via scipy.fft.dct

### 4. SpecAugment
- Park et al. 2019 — time masking and frequency masking
- `spec_augment_time_mask(spectrogram, num_masks, max_width) -> NDArray`
- `spec_augment_freq_mask(spectrogram, num_masks, max_width) -> NDArray`
- Pure numpy: zero out random contiguous bands

### 5. Forced alignment (DTW variant)
- Dynamic Time Warping for phoneme-to-audio alignment
- `dtw_alignment(frame_probs: NDArray, phoneme_sequence: list[int]) -> list[tuple[int,int]]`
- Constrained DTW where phoneme order is fixed
- Source: dtw-python (GPL — find MIT alternative), or custom numpy DTW

### 6. CTC greedy decoding
- Collapse repeated tokens, remove blanks
- `ctc_greedy_decode(log_probs: NDArray, blank_id: int) -> list[int]`
- Pure Python/numpy, no framework dependency

### 7. CTC beam search decoding
- Prefix beam search with language model fusion option
- `ctc_beam_decode(log_probs: NDArray, beam_width: int, blank_id: int) -> list[tuple[list[int], float]]`
- Source: prefix beam search implementations

### 8. Audio windowing / chunking
- Split long audio into fixed-length overlapping windows
- `audio_windows(signal: NDArray, window_size: int, hop_size: int) -> NDArray`
- Pure numpy strided view

## Research questions

1. What is the pure numpy mel spectrogram pipeline?
   (librosa uses numpy internally — extract the core math)
2. For forced alignment: what is the constrained DTW algorithm?
   (Monotonic alignment, no backtracking — different from standard DTW)
3. For CTC decoding: what is the prefix beam search algorithm?
   (Hannun 2014 — find clean numpy implementation)
4. What are natural contracts? (sample rate > 0, n_fft power of 2,
   mel_fmin < mel_fmax, MFCC n_mfcc <= n_mels)
5. What are the numerical stability considerations?
   (log-mel needs floor/amin, CTC needs log-space computation)

## Output format

All atoms should use concept_type `signal_transform` or `signal_filter` as
appropriate.

For each candidate atom, provide:
```
Name: log_mel_spectrogram
Description: Convert a waveform into a log-mel spectrogram using STFT, mel
  filterbank projection, and stable log scaling.
Source: URL to the best reference implementation, paper, or library source
License: MIT, BSD, Apache-2.0, ISC, or public domain; flag any incompatible license
Concept type: signal_transform or signal_filter
Signature: (signal: NDArray, sample_rate: int, n_fft: int, hop_length: int,
            n_mels: int, fmin: float, fmax: float) -> NDArray
Pure function boundary: waveform arrays and explicit parameters in, transformed
  arrays or decoded token IDs out; no audio file I/O, model state, GPU state,
  global RNG, or external services.
Contracts:
  - require: sample_rate > 0
  - require: n_fft > 0 and hop_length > 0
  - require: 0 <= fmin < fmax <= sample_rate / 2
  - ensure: result has n_mels frequency bins
Witness: short sine wave or impulse with fixed parameters; verify output shape
  and finite values.
Dependencies: numpy only preferred; scipy/librosa acceptable when license and
  dependency weight are justified
CDG stages covered: birdclef/log_mel, bengali_speech/ctc_decode, ...
```
