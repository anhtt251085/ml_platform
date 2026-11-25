/* SAI feature front ends; see sai_features.h. Float, single precision, deterministic order of operations. */
#include "sai_features.h"

#include <math.h>

#define SAI_PI 3.14159265358979323846f

static int is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

/* In-place iterative radix-2 complex FFT. Twiddles from sinf/cosf per stage (no tables, no heap). */
static void fft(float *re, float *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * SAI_PI / (float)len;
        for (int k = 0; k < len / 2; k++) {
            float wr = cosf(ang * (float)k), wi = sinf(ang * (float)k);
            for (int i = 0; i < n; i += len) {
                int a = i + k, b = i + k + len / 2;
                float xr = re[b] * wr - im[b] * wi;
                float xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr; im[a] += xi;
            }
        }
    }
}

/* Hann-windowed power spectrum of x[0..n) zero padded to n_fft; leaves |X[k]|^2 in re[0..n_fft/2]. */
static void power_spectrum(const float *x, int n, int n_fft, float *re, float *im) {
    for (int i = 0; i < n_fft; i++) {
        float w = (i < n && n > 1) ? 0.5f - 0.5f * cosf(2.0f * SAI_PI * (float)i / (float)(n - 1)) : 0.0f;
        re[i] = i < n ? x[i] * w : 0.0f;
        im[i] = 0.0f;
    }
    fft(re, im, n_fft);
    for (int k = 0; k <= n_fft / 2; k++) re[k] = re[k] * re[k] + im[k] * im[k];
}

sai_feat_status_t sai_band_energy(const float *x, int n, int n_fft, int bands, float *work, float *out) {
    if (!x || !work || !out || !is_pow2(n_fft) || n_fft > SAI_FEAT_MAX_FFT || n < 2 || n > n_fft || bands < 1 ||
        bands > n_fft / 2)
        return SAI_FEAT_BAD_ARG;
    float *re = work, *im = work + SAI_FEAT_MAX_FFT;
    power_spectrum(x, n, n_fft, re, im);
    int bins = n_fft / 2;                    /* bins 1..bins (DC dropped) */
    for (int b = 0; b < bands; b++) {
        int lo = 1 + (b * bins) / bands, hi = 1 + ((b + 1) * bins) / bands;
        float e = 0.0f;
        for (int k = lo; k < hi; k++) e += re[k];
        out[b] = log10f(e / (float)n + 1e-6f);
    }
    return SAI_FEAT_OK;
}

static float hz_to_mel(float f) { return 2595.0f * log10f(1.0f + f / 700.0f); }
static float mel_to_hz(float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); }

int sai_mfcc_frames(const sai_mfcc_cfg_t *c, int n) {
    if (!c || c->stride < 1 || n < c->frame) return 0;
    return 1 + (n - c->frame) / c->stride;
}

sai_feat_status_t sai_mfcc(const float *audio, int n, const sai_mfcc_cfg_t *c, float *work, float *out) {
    if (!audio || !c || !work || !out || !is_pow2(c->n_fft) || c->n_fft > SAI_FEAT_MAX_FFT ||
        c->frame > c->n_fft || c->frame < 2 || c->n_mels < 1 || c->n_mels > SAI_FEAT_MAX_MELS ||
        c->n_coef < 1 || c->n_coef > c->n_mels || c->sample_rate < 1 || !(c->f_hi > c->f_lo))
        return SAI_FEAT_BAD_ARG;
    float *re = work, *im = work + SAI_FEAT_MAX_FFT, *mel = work + 2 * SAI_FEAT_MAX_FFT;
    int frames = sai_mfcc_frames(c, n), bins = c->n_fft / 2;
    float m_lo = hz_to_mel(c->f_lo), m_hi = hz_to_mel(c->f_hi);
    for (int f = 0; f < frames; f++) {
        power_spectrum(audio + f * c->stride, c->frame, c->n_fft, re, im);
        for (int m = 0; m < c->n_mels; m++) {
            /* triangle m spans mel points m, m+1, m+2 of n_mels+2 equally spaced points */
            float f0 = mel_to_hz(m_lo + (m_hi - m_lo) * (float)m / (float)(c->n_mels + 1));
            float f1 = mel_to_hz(m_lo + (m_hi - m_lo) * (float)(m + 1) / (float)(c->n_mels + 1));
            float f2 = mel_to_hz(m_lo + (m_hi - m_lo) * (float)(m + 2) / (float)(c->n_mels + 1));
            float e = 0.0f;
            for (int k = 0; k <= bins; k++) {
                float hz = (float)k * (float)c->sample_rate / (float)c->n_fft, w = 0.0f;
                if (hz > f0 && hz <= f1) w = (hz - f0) / (f1 - f0);
                else if (hz > f1 && hz < f2) w = (f2 - hz) / (f2 - f1);
                e += w * re[k];
            }
            mel[m] = logf(e + 1e-6f);
        }
        for (int k = 0; k < c->n_coef; k++) {
            float s = 0.0f;
            for (int m = 0; m < c->n_mels; m++) s += mel[m] * cosf(SAI_PI * (float)k * ((float)m + 0.5f) / (float)c->n_mels);
            out[f * c->n_coef + k] = s * sqrtf((k == 0 ? 1.0f : 2.0f) / (float)c->n_mels);
        }
    }
    return SAI_FEAT_OK;
}
