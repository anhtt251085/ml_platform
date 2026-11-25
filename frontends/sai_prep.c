/* SAI window preprocessing; see sai_prep.h. */
#include "sai_prep.h"

#if defined(__GNUC__) && !defined(__clang__)
#define SAI_NO_CONTRACT __attribute__((optimize("fp-contract=off")))
#else
#define SAI_NO_CONTRACT
#endif

SAI_NO_CONTRACT
int sai_prep_window(const float *raw, int c_raw, int t_raw, const int *chans, int c_out, int decim, int demean,
                    float *out) {
    if (!raw || !out || c_raw < 1 || t_raw < 1 || c_out < 1 || decim < 1 || t_raw / decim < 1) return -1;
    if (!chans && c_out != c_raw) return -1;
    int t_out = t_raw / decim;
    for (int o = 0; o < c_out; o++) {
        int c = chans ? chans[o] : o;
        if (c < 0 || c >= c_raw) return -1;
        const float *x = raw + (long)c * t_raw;
        float *y = out + (long)o * t_out;
        for (int t = 0; t < t_out; t++) {
            float s = 0.0f;
            for (int k = 0; k < decim; k++) s += x[t * decim + k];
            y[t] = decim == 1 ? s : s / (float)decim;
        }
        if (demean) {
            float s = 0.0f;
            for (int t = 0; t < t_out; t++) s += y[t];
            float m = s / (float)t_out;
            for (int t = 0; t < t_out; t++) y[t] = y[t] - m;
        }
    }
    return t_out;
}
