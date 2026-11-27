import math
import sys

try:
    import scipy
    from scipy.stats import chi2
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

apps = {
    "Facebook":  (1364, 1700, 465, 1700),
    "Instagram": (1262, 1700, 384, 1700),
    "Snapchat":  (1040, 1700, 478, 1700),
    "TikTok":    (1190, 1700, 476, 1700),
    "X":         (1350, 1700, 307, 1700),
    "YouTube":   (1420, 1700, 696, 1700),
}

Z = 1.96

print("=" * 72)
print("STATISTICAL INFERENCE VERIFICATION")
print("=" * 72)
print("Python:", sys.version.replace("\n", " "))
if SCIPY_AVAILABLE:
    print("SciPy:", scipy.__version__)
else:
    print("SciPy: not installed")
print()

log_rr = []
variances = []

for app, (a, n_r, c, n_n) in apps.items():

    p_r = a / n_r
    p_n = c / n_n

    # Risk / prevalence difference
    rd = p_r - p_n

    se_rd = math.sqrt(
        p_r * (1 - p_r) / n_r +
        p_n * (1 - p_n) / n_n
    )

    rd_low = rd - Z * se_rd
    rd_high = rd + Z * se_rd

    # Prevalence ratio / risk ratio
    rr = p_r / p_n

    logrr = math.log(rr)

    var_logrr = (
        1 / a - 1 / n_r +
        1 / c - 1 / n_n
    )

    se_logrr = math.sqrt(var_logrr)

    rr_low = math.exp(logrr - Z * se_logrr)
    rr_high = math.exp(logrr + Z * se_logrr)

    log_rr.append(logrr)
    variances.append(var_logrr)

    print(app)
    print(f"  RD = {rd*100:.4f} pp "
          f"(95% CI {rd_low*100:.4f} to {rd_high*100:.4f})")
    print(f"  NRPR = {rr:.6f} "
          f"(95% CI {rr_low:.6f} to {rr_high:.6f})")
    print()

# ------------------------------------------------------------
# Crude pooled estimate
# ------------------------------------------------------------

a_total = sum(v[0] for v in apps.values())
nr_total = sum(v[1] for v in apps.values())
c_total = sum(v[2] for v in apps.values())
nn_total = sum(v[3] for v in apps.values())

p_r = a_total / nr_total
p_n = c_total / nn_total

rd = p_r - p_n

se_rd = math.sqrt(
    p_r * (1 - p_r) / nr_total +
    p_n * (1 - p_n) / nn_total
)

pooled_rr = p_r / p_n

var_pooled_logrr = (
    1 / a_total - 1 / nr_total +
    1 / c_total - 1 / nn_total
)

se_pooled_logrr = math.sqrt(var_pooled_logrr)

print("=" * 72)
print("CRUDE POOLED")
print("=" * 72)

print(
    f"RD = {rd*100:.6f} pp "
    f"(95% CI {(rd-Z*se_rd)*100:.6f} "
    f"to {(rd+Z*se_rd)*100:.6f})"
)

print(
    f"NRPR = {pooled_rr:.6f} "
    f"(95% CI "
    f"{math.exp(math.log(pooled_rr)-Z*se_pooled_logrr):.6f} "
    f"to "
    f"{math.exp(math.log(pooled_rr)+Z*se_pooled_logrr):.6f})"
)

# ------------------------------------------------------------
# Inverse-variance fixed-effect model on log risk ratios
# ------------------------------------------------------------

weights_fixed = [1 / v for v in variances]

fixed_logrr = (
    sum(w * y for w, y in zip(weights_fixed, log_rr))
    / sum(weights_fixed)
)

fixed_se = math.sqrt(1 / sum(weights_fixed))

fixed_rr = math.exp(fixed_logrr)
fixed_low = math.exp(fixed_logrr - Z * fixed_se)
fixed_high = math.exp(fixed_logrr + Z * fixed_se)

# ------------------------------------------------------------
# Cochran Q
# ------------------------------------------------------------

Q = sum(
    w * (y - fixed_logrr) ** 2
    for w, y in zip(weights_fixed, log_rr)
)

df = len(apps) - 1

# ------------------------------------------------------------
# DerSimonian-Laird tau-squared
# ------------------------------------------------------------

sum_w = sum(weights_fixed)
sum_w_sq = sum(w ** 2 for w in weights_fixed)

C = sum_w - (sum_w_sq / sum_w)

tau2 = max(0.0, (Q - df) / C)

# ------------------------------------------------------------
# Random-effects model
# ------------------------------------------------------------

weights_random = [
    1 / (v + tau2)
    for v in variances
]

random_logrr = (
    sum(w * y for w, y in zip(weights_random, log_rr))
    / sum(weights_random)
)

random_se = math.sqrt(1 / sum(weights_random))

random_rr = math.exp(random_logrr)
random_low = math.exp(random_logrr - Z * random_se)
random_high = math.exp(random_logrr + Z * random_se)

# ------------------------------------------------------------
# I-squared
# ------------------------------------------------------------

I2 = max(0.0, (Q - df) / Q) * 100

print()
print("=" * 72)
print("META-ANALYSIS")
print("=" * 72)

print(
    f"Fixed-effect NRPR = {fixed_rr:.6f} "
    f"(95% CI {fixed_low:.6f} to {fixed_high:.6f})"
)

print(f"Q = {Q:.6f}")
print(f"df = {df}")

if SCIPY_AVAILABLE:
    q_p = chi2.sf(Q, df)
    print(f"Q p-value = {q_p:.12g}")

print(f"tau^2 (DerSimonian-Laird) = {tau2:.9f}")
print(f"I^2 = {I2:.6f}%")

print(
    f"Random-effects NRPR = {random_rr:.6f} "
    f"(95% CI {random_low:.6f} to {random_high:.6f})"
)