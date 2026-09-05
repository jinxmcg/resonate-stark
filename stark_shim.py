"""stark_qa imports PrimeKG from PyTDC's tdc.resource (used only to build the raw KB);
PyTDC >= 0.4 does not build here, so provide the name before importing stark_qa.
We always load the PROCESSED SKB (download_processed=True), which never calls it."""
import sys, types
try:
    import tdc.resource  # noqa: F401
except Exception:
    m = types.ModuleType("tdc.resource"); m.PrimeKG = None
    sys.modules["tdc.resource"] = m
