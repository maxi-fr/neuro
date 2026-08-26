# 01 — Canonical Observable reduction and its healthy envelope

**What to build:** The Observable gets one definition and one reference to hinge against. A single
NumPy function turns a raw EEG trajectory into log-power Frames at a given Observable geometry, and
the healthy envelope is measured by quantiling over Frames that same function produced. The torch
and jax reductions that stay behind for the waveform path stop being independent conventions and
become twins tested against the canonical one. Nothing downstream changes yet: the waveform hinge
keeps reading the envelope it reads today.

**Blocked by:** None — can start immediately.

## Acceptance criteria

- [ ] One NumPy function takes raw EEG and an Observable geometry and returns log-power Frames, in
      raw units, applying in order: periodic Hann taper, density scaling, DC exclusion, band slice,
      bin pooling, Frame Kernel on power, log floor, log.
- [ ] A Frame's sample support is `(kernel_width - 1) * hop + segment`, and the Frame count the
      function reports for a given span reflects that support.
- [ ] The torch reduction the spectral training Loss uses agrees with the canonical function to
      floating-point tolerance on random input, across band, pooling and Frame Kernel settings.
- [ ] The jax reduction inside the waveform spectral hinge agrees with the canonical function to
      floating-point tolerance on random input.
- [ ] The envelope builder accepts an Observable geometry and computes its quantile over Frames the
      canonical function produced, rather than over per-bin periodograms pooled afterwards.
- [ ] The envelope artifact records the geometry it was measured at, and the loader surfaces it.
- [ ] The envelope arrays the waveform spectral hinge and the mean-square Observable read today are
      still written and still load, so the waveform path is untouched by this ticket.
