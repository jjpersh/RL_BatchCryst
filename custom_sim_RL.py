# Standard Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r"C:\Users\jjper\Documents\RESEARCH\takeda\PharmaPy\PharmaPy-master")

##### PharmaPy imports #####
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Interpolation import PiecewiseLagrange



class ProcessSimulation:
    """
    Class to simulate and analyze crystallization.
    """
    
    def __init__(self, trialname="TEST"):
        """
        Initialize simulation with given trial name.
        """
        # Physical properties path
        self.path_phys = r'compounds_mom.json'
        
        # Create unique directory for this instance
        self.trialname = trialname
        self.results_dir = os.path.join("data", f"sim_{self.trialname}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # # Process parameters
        # self.vol_liq = 0.010
        # self.tau_R01 = 1800
        # self.runtime_reactor = 3600 * 2
        
    def setup_run(self, c_in=np.array([0.33, 0.33, 0, 0, 0]), temp_program=None, runtime_cryst=None):
        if temp_program is None:
            temp_init = 323.15  # K
            temp_final = 290.0  # K
            temp_program = np.array([temp_init, temp_final])
        else:
            temp_init = np.max(temp_program)
            temp_final = np.min(temp_program)
        if runtime_cryst is None:
            runtime_cryst = 3600
        
        # Crystallizer Kinetics

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

        # Setup Crystallizer
        
        # Initial Liquid characteristics
        conc_init = kinetics.get_solubility(temp_init)  # kg/m**3
        conc_init = (conc_init, 0)  # API conc and 0 for the solvent.
        
        # Define the liquid phase for the crystallizer
        liquid = LiquidPhase(self.path_phys, temp=temp_init, vol=0.1, mass_conc=conc_init,)
                            # ind_solv=-1)

        # Solid Characteristics (all API, 0 for the solvent)
        massfrac_solid = (1, 0)
        x_distrib = np.geomspace(1, 1500, num=35)
        distrib = np.zeros_like(x_distrib)
        solid = SolidPhase(self.path_phys, temp_init, mass_frac=massfrac_solid,
                           distrib=distrib, x_distrib=x_distrib)
        
        lagrange_fn = PiecewiseLagrange(runtime_cryst, temp_program)
        
        # ---------- Unit operation definition ----------
        self.CR01 = BatchCryst(target_comp='solute', method='1D-FVM',
                                   controls={'temp': lagrange_fn.evaluate_poly})
        
        # Connect kinetics and phases to the unit operation
        self.CR01.Kinetics = kinetics
        self.CR01.Phases = (liquid, solid)
        
        # Run simulation / Solve Model
        self.CR01.solve_unit(runtime_cryst, verbose=True)

    def output(self):
        results = self.CR01.result
        final_distrib =results.distrib[-1,:]
        x_sizes = results.x_cryst
        # Calculate PDF and CDF
        pdf = final_distrib / np.sum(final_distrib)
        cdf = np.cumsum(pdf)
        # Calculate D values
        D10 = np.interp(0.10, cdf, x_sizes)
        D50 = np.interp(0.50, cdf, x_sizes)
        D90 = np.interp(0.90, cdf, x_sizes)
        span = (D90 - D10) / D50
        return D50, span

        
    # def save_results(self, filename='results.csv'):
    #     results = self.CR01.result
        
    #     results_dict = {
    #         'time': results.time,
    #         'vol': results.vol,
    #         'supersat': results.supersat,
    #         'solubility': results.solubility,
    #         'temperature': results.temp
    #     }
            
    #     # Add mass concentrations
    #     for i in range(len(results.mass_conc[0])):
    #         results_dict[f'mass_conc_{i}'] = results.mass_conc[:,i]
            
    #     # Add moments
    #     for i in range(len(results.mu_n[0])):
    #         results_dict[f'moment_{i}'] = results.mu_n[:,i]
            
    #     # Add distributions
    #     for i in range(results.distrib.shape[1]):
    #         results_dict[f'distrib_{i}'] = results.distrib[:,i]
            
    #     for i in range(results.vol_distrib.shape[1]):
    #         results_dict[f'vol_distrib_{i}'] = results.vol_distrib[:,i]

    #     # Repeat x_cryst for each timestep
    #     for i in range(len(results.x_cryst)):
    #         results_dict[f'x_cryst_{i}'] = np.full_like(results.time, results.x_cryst[i])
            
    #     filepath = os.path.join(self.results_dir, filename)
    #     pd.DataFrame(results_dict).to_csv(filepath, index=False)
    #     print(f"Results saved to {filepath}")

    # def plots(self):
    #     moments = self.CR01.result.mu_n[-1]
    #     mean_size = moments[1] / moments[0] * 1e6
    #     print(f'Mean crystal size: {mean_size:.2f} μm')

    #     # Create and save profiles plot
    #     fig, ax = self.CR01.plot_profiles(figsize=(10, 10))
    #     plt.savefig(os.path.join(self.results_dir, 'profiles.png'), 
    #                bbox_inches='tight', dpi=300)
    #     plt.close(fig)

    #     # Create and save crystal size distribution plot
    #     fig_csd, ax_csd = self.CR01.plot_csd(times=np.linspace(6000, 9000), figsize=(5, 4))
    #     ax_csd.axvline(x=mean_size, color='r', linestyle='--', alpha=0.5)
    #     ax_csd.text(0.02, 0.98, f'Mean crystal size: {mean_size:.2f} μm', 
    #                transform=ax_csd.transAxes, verticalalignment='top', color='red', fontsize=8)
    #     plt.savefig(os.path.join(self.results_dir, 'crystal_size_distribution.png'),
    #                bbox_inches='tight', dpi=300)
    #     plt.close(fig_csd)

    #     # Create and save CSD heatmap plot
    #     fig_heatmap, ax_heatmap = self.CR01.plot_csd_heatmap(figsize=(5, 4))
    #     ax_heatmap.axvline(x=mean_size, color='r', linestyle='--', alpha=0.5)
    #     ax_heatmap.text(0.02, 0.98, f'Mean crystal size: {mean_size:.2f} μm',
    #                transform=ax_heatmap.transAxes, verticalalignment='top', color='red', fontsize=8)
    #     plt.savefig(os.path.join(self.results_dir, 'csd_heatmap.png'),
    #                bbox_inches='tight', dpi=300)
    #     plt.close(fig_heatmap)
