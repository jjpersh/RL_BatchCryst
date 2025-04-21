import numpy as np
import sys
import copy

# Add PharmaPy to path
sys.path.append(r"C:\Users\jjper\Documents\RESEARCH\takeda\PharmaPy")

from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Interpolation import PiecewiseLagrange

# --- Helper function ---
def compute_d50_span(results):
    final_distrib = results.distrib[-1, :]
    x_sizes = results.x_cryst

    pdf = final_distrib / np.sum(final_distrib)
    cdf = np.cumsum(pdf)

    D10 = np.interp(0.10, cdf, x_sizes)
    D50 = np.interp(0.50, cdf, x_sizes)
    D90 = np.interp(0.90, cdf, x_sizes)
    span = (D90 - D10) / D50
    return D50, span

# --- Kinetics ---
prim = (3e8, 0, 3)
sec = (4.46e10, 0, 2, 1)
growth = (5, 0, 1.32)
solub_cts = [1.45752618e+01, -9.98982300e-02, 1.72100000e-04]
kinetics = CrystKinetics(solub_cts, nucl_prim=prim, nucl_sec=sec, growth=growth)

# --- Initial Phases ---
path = 'compounds_mom.json'
temp_init = 323.15
conc_init = kinetics.get_solubility(temp_init)
conc_init = (conc_init, 0)

liquid = LiquidPhase(path, temp=temp_init, vol=0.1, mass_conc=conc_init)
x_distrib = np.geomspace(1, 1500, 35)
distrib = np.zeros_like(x_distrib)
solid = SolidPhase(path, temp=temp_init, mass_frac=(1, 0), distrib=distrib, x_distrib=x_distrib)

# --- Time Setup ---
time_final = 7200
n_steps = 24
dt = time_final / n_steps
stepwise_temps = np.linspace(temp_init, 290.0, n_steps)

results_all = []
metrics = []

for i in range(n_steps):
    print(f"\n--- Step {i + 1}/{n_steps} ---")

    current_temp = stepwise_temps[i]

    interpolator = PiecewiseLagrange(dt, [current_temp], order=1)
    controls = {'temp': interpolator.evaluate_poly}

    liquid.temp = current_temp
    solid.temp = current_temp

    CR01 = BatchCryst(target_comp='solute', method='1D-FVM', controls=controls)
    CR01.Kinetics = kinetics
    CR01.Phases = (liquid, solid)

    results = CR01.solve_unit(dt, verbose=False)
    results_all.append(results)

    D50, span = compute_d50_span(CR01.result)
    print(f"D50 = {D50:.2f} μm, Span = {span:.2f}")
    metrics.append((D50, span))

    liquid = copy.deepcopy(CR01.Phases[0])
    solid = copy.deepcopy(CR01.Phases[1])

# Final Summary
print("\n--- Summary of D50 and Span ---")
for i, (D50, span) in enumerate(metrics):
    print(f"Step {i + 1:2}: D50 = {D50:.2f} μm, Span = {span:.2f}")
