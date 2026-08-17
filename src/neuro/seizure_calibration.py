# Spread-detection calibration, shared by the simulator (neuro.seizure) and the analysis layer
# (neuro.metrics). It lives in its own leaf module -- importing nothing from neuro -- so that
# neuro.metrics does not have to reach through neuro.seizure -> neuro.jansen_rit -> neuro.config,
# which would make neuro.config unable to import neuro.metrics at all.

SPREAD_WINDOW_S = 1.0
SPREAD_HOP_S = 0.25
SPREAD_PERSIST_S = 1.0
SEIZURE_PTP_MV = 5.0
