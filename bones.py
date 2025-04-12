# Standard library imports
import numpy as np

# add PharmaPy package to python path
import sys
sys.path.append(r"C:\Users\jjper\Documents\RESEARCH\takeda\PharmaPy\PharmaPy-master")

##### PharmaPy imports #####
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Interpolation import PiecewiseLagrange



#--------------------------------
# Kinetics
#--------------------------------

# Primary nucleation kinetic parameters (k_p, Ea_p, p)
prim = (3e8, 0, 3)  # kP in #/m3/s
# Secondary nucleation kinetic parameters (k_s, Ea_s, s_1, s_2)
sec = (4.46e10, 0, 2, 1)
# Crystal growth kinetic parameters (k_g, Ea_g, g)
growth = (5, 0, 1.32)  # kG in um/s



# Solubility constants for the solution in polynomial form (A + BT + CT^2)
solub_cts = [1.45752618e+01, -9.98982300e-02,  1.72100000e-04]

# Creating a kinetics object using the kinetic and solubility data
kinetics = CrystKinetics(solub_cts, nucl_prim=prim, nucl_sec=sec,
                         growth=growth)


#--------------------------------
# Crystallizer Setup
#--------------------------------

# Always must define the path to the physical properties of
# the species in the system.
path = 'compounds_mom.json'

# Temperature endpoints
temp_init = 323.15  # K
temp_final = 290.0  # K

# Intermediate temperature values for a
# linear cooling profile
temp_vals = [320, 310, 310, 305, 290]

# Let's specify a temps array to pass
# to the PiecewiseLagrange function.

# Here, the format will be a matrix
# with each row representing the two
# points for each linear segment. See
# below:
temps = np.array([[temp_init, temp_vals[0]],  # We go from the initial temperature to the first int temp
                  [temp_vals[0], temp_vals[1]],  # Next from int temp 1 to int temp 2
                  [temp_vals[1], temp_vals[2]],  # Etc...
                  [temp_vals[2], temp_vals[3]],
                  [temp_vals[3], temp_vals[4]],
                  [temp_vals[4], temp_final]], dtype=np.float64)  # Finish at final temp from the last int temp

# Initial Liquid characteristics
conc_init = kinetics.get_solubility(temp_init)  # kg/m**3
conc_init = (conc_init, 0)  # API conc and 0 for the solvent.

# Define the liquid phase for the crystallizer
liquid = LiquidPhase(path, temp=temp_init, vol=0.1, mass_conc=conc_init,)
                    # ind_solv=-1)

# Solid Characteristics (all API, 0 for the solvent)
massfrac_solid = (1, 0)

x_distrib = np.arange(1, 501)
distrib = np.zeros_like(x_distrib)
solid = SolidPhase(path, temp_init, mass_frac=massfrac_solid,
                   distrib=distrib, x_distrib=x_distrib)

# ---------- Control Setup
time_final = 3600

# Specify the linear temperature function (order 2)
interpolator = PiecewiseLagrange(time_final, y_vals=temps,
                                 order=2)

controls = {'temp': interpolator.evaluate_poly}

# ---------- Unit operation definition
CR01 = BatchCryst(target_comp='solute', method='moments',
                  controls=controls)

# Connect kinetics and phases to the unit operation
CR01.Kinetics = kinetics
CR01.Phases = (liquid, solid)

# ---------- Solve model
results = CR01.solve_unit(time_final, verbose=True)
