/* Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE. */
/* SAI feature front ends: the same C runs on the MCU and (through a Python binding) in training. The arithmetic is
   identical (no FMA contraction, fixed summation order); sinf/cosf/logf come from each toolchain's libm, so values
   may differ in the last bit between the host and an MCU libm - far below the model's INT8 input resolution.

   No heap, no libc beyond <math.h> (sinf, cosf, logf, sqrtf). Work memory is the caller's (SAI_FEAT_WORK_FLOATS). */
#ifndef SAI_FEATURES_H
#define SAI_FEATURES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SAI_FEAT_MAX_FFT 1024
#define SAI_FEAT_MAX_MELS 64
/* floats of work memory every call needs: re + im FFT buffers and the mel energies */
#define SAI_FEAT_WORK_FLOATS (2 * SAI_FEAT_MAX_FFT + SAI_FEAT_MAX_MELS)

typedef enum { SAI_FEAT_OK = 0, SAI_FEAT_BAD_ARG = 1 } sai_feat_status_t;

/* Log band energies of one window (vibration, current, ...): Hann window, zero pad to n_fft (power of two,
   <= SAI_FEAT_MAX_FFT), power spectrum of bins 1..n_fft/2, `bands` equal-width bands,
   out[b] = log10(energy_b / n + 1e-6). */
sai_feat_status_t sai_band_energy(const float *x, int n, int n_fft, int bands, float *work, float *out);

typedef struct {
    int sample_rate;   /* Hz */
    int frame;         /* samples per frame (<= n_fft) */
    int stride;        /* samples between frames */
    int n_fft;         /* power of two <= SAI_FEAT_MAX_FFT */
    int n_mels;        /* <= SAI_FEAT_MAX_MELS */
    int n_coef;        /* MFCC coefficients kept (<= n_mels) */
    float f_lo, f_hi;  /* mel filterbank edges, Hz */
} sai_mfcc_cfg_t;

/* Number of frames for n input samples: 1 + (n - frame) / stride (0 when n < frame). */
int sai_mfcc_frames(const sai_mfcc_cfg_t *cfg, int n);

/* MFCC of a signal: per frame Hann window, power spectrum, triangular mel filterbank, log, orthonormal DCT-II.
   out is frames x n_coef, row major. Audio is float in [-1, 1]. */
sai_feat_status_t sai_mfcc(const float *audio, int n, const sai_mfcc_cfg_t *cfg, float *work, float *out);

#ifdef __cplusplus
}
#endif
#endif
