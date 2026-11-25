/* Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE. */
/* SAI window preprocessing: channel selection, decimation (block average), per-window mean removal.
   The same C runs in training (Python binding) and on the MCU, so the model sees identical inputs.
   No heap; sequential float32 sums in a fixed order. Compile without -ffast-math. */
#ifndef SAI_PREP_H
#define SAI_PREP_H

#ifdef __cplusplus
extern "C" {
#endif

/* raw: c_raw channels x t_raw samples, channel-major.
   chans: c_out channel indices into raw (NULL = 0..c_raw-1, c_out = c_raw).
   decim >= 1: each output sample is the mean of `decim` consecutive raw samples (t_out = t_raw / decim).
   demean: subtract each output channel's mean over the window.
   out: c_out x t_out. Returns t_out, or -1 on bad arguments. */
int sai_prep_window(const float *raw, int c_raw, int t_raw, const int *chans, int c_out, int decim, int demean,
                    float *out);

#ifdef __cplusplus
}
#endif
#endif
