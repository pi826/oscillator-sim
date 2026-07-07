"""Numerical constants and classification thresholds, collected in one place."""

# --- integration -----------------------------------------------------------
DEFAULT_DT: float = 0.005
DEFAULT_N: int = 20
DEFAULT_SEED: int = 12345

# --- natural frequency distribution (S1 / graph modes) ---------------------
OMEGA_MEAN: float = 1.0
OMEGA_STD: float = 0.3

# --- S1 classification ------------------------------------------------------
# r_m >= this value counts as "r_m is (approximately) 1".
R_ONE_THRESHOLD: float = 0.95

# --- cotangent nearest-neighbor model ---------------------------------------
# |cot| is clipped to this value so the fixed-step RK4 stays stable when two
# neighboring phases (nearly) collide, where the exact vector field diverges
COT_CLIP: float = 100.0

# --- curve sampling -----------------------------------------------------------
# polyline samples for the displayed curve in S1 mode (before the GUI's
# resolution multiplier)
CURVE_DISPLAY_SAMPLES: int = 720
# resolution multipliers offered in the GUI; the default is applied to both
# the S1 display polyline and the graph-mode arclength sampling
RESOLUTION_CHOICES: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
DEFAULT_RESOLUTION: float = 2.0

# --- graph mode -------------------------------------------------------------
# number of samples used to detect self-intersections of the curve
CURVE_DETECT_SAMPLES: int = 1024
# number of samples per unit of curve parameter used for arclength tables
# (before the resolution multiplier)
CURVE_ARCLEN_SAMPLES: int = 4096
# two crossing points closer than this (relative to curve diameter) are merged
VERTEX_MERGE_TOL: float = 1e-3
# tolerance for matching curve parameters in the S1-compliant branching rule
PARAM_MATCH_TOL: float = 1e-6
# Monte-Carlo sample size for the uniform pair-distance reference distribution
KS_MC_SAMPLES: int = 100_000
# maximum vertex crossings processed for one oscillator within a single step
MAX_CROSSINGS_PER_STEP: int = 64

# --- sphere mode ------------------------------------------------------------
SPHERE_OMEGA_MEAN: float = 1.0
SPHERE_OMEGA_STD: float = 0.3
SPHERE_R_SYNC: float = 0.9  # r above this -> sync
SPHERE_R_ZERO: float = 0.2  # r below this -> candidate for uniform / ring
SPHERE_UNIFORM_TOL: float = 0.12  # ||S - I/3||_F below this -> uniform
SPHERE_RING_TOL: float = 0.12  # eigenvalue distance to (1/2, 1/2, 0) -> ring
