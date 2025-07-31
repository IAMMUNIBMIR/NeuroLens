from __future__ import annotations
import time, csv, io, os
from dataclasses import dataclass
from collections import deque
from typing import Optional, Deque
import numpy as np

DEFAULT_MAXLEN = 500

@dataclass
class MetricRow:
    t_ms: float
    latency_ms: float
    kl_drift: Optional[float]
    top_label: Optional[str]
    top_prob: Optional[float]

class MetricsStore:
    def __init__(self, maxlen: int = DEFAULT_MAXLEN):
        self.rows: Deque[MetricRow] = deque(maxlen=maxlen)

    def add(self, latency_ms: float, probs=None, labels=None, baseline=None, t_ms: Optional[float] = None):
        if t_ms is None:
            t_ms = time.time() * 1000.0
        kl = None
        top_label = None
        top_prob = None
        if probs is not None:
            p = np.asarray(probs, dtype=np.float32)
            p = np.clip(p, 1e-6, 1.0); p = p / p.sum()
            if baseline is None:
                env = os.environ.get("BASELINE_PROBS")
                if env:
                    try:
                        baseline = np.array([float(x) for x in env.split(",")], dtype=np.float32)
                    except Exception:
                        baseline = None
            if baseline is None:
                baseline = np.ones_like(p) / p.size
            q = np.asarray(baseline, dtype=np.float32)
            q = np.clip(q, 1e-6, 1.0); q = q / q.sum()
            kl = float(np.sum(p * (np.log(p) - np.log(q))))
            idx = int(np.argmax(p))
            if labels is not None and idx < len(labels):
                top_label = str(labels[idx])
            top_prob = float(p[idx])
        self.rows.append(MetricRow(t_ms=float(t_ms), latency_ms=float(latency_ms),
                                   kl_drift=kl, top_label=top_label, top_prob=top_prob))

    def summary(self):
        if not self.rows:
            return {"count": 0}
        lat = np.array([r.latency_ms for r in self.rows], dtype=np.float32)
        p50 = float(np.percentile(lat, 50))
        p95 = float(np.percentile(lat, 95))
        return {"count": len(self.rows), "p50_ms": p50, "p95_ms": p95, "last_kl": self.rows[-1].kl_drift}

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["timestamp_ms","latency_ms","kl_drift","top_label","top_prob"])
        for r in self.rows:
            w.writerow([int(r.t_ms),
                        f"{r.latency_ms:.3f}",
                        "" if r.kl_drift is None else f"{r.kl_drift:.6f}",
                        r.top_label or "",
                        "" if r.top_prob is None else f"{r.top_prob:.6f}"])
        return buf.getvalue()

def timed_infer(callable_fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = callable_fn(*args, **kwargs)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return out, dt_ms

# Streamlit-aware singleton
def get_store(st_module=None) -> MetricsStore:
    if st_module is None:
        try:
            import streamlit as st_module  # type: ignore
        except Exception:
            st_module = None
    if st_module is not None and hasattr(st_module, "session_state"):
        if "metrics_store" not in st_module.session_state:
            st_module.session_state["metrics_store"] = MetricsStore()
        return st_module.session_state["metrics_store"]
    # Fallback if Streamlit isn't available (tests)
    global _GLOBAL_STORE
    try:
        return _GLOBAL_STORE  # type: ignore[name-defined]
    except NameError:
        _GLOBAL_STORE = MetricsStore()
        return _GLOBAL_STORE
