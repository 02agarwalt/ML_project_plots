import numpy as np
from ndmg.nuis import nuis
import nibabel as nb
import matplotlib
import matplotlib.pyplot as plt
import unittest
import sim_gen
        
class TestNuisanceCorrection(unittest.TestCase):
    def test_everything(self):
        R2_vals = []
        x_vals = []
        for i in range(1, 51):
            err_var = i*0.01
            path = "sim" + str(i) + "/"
            sim = sim_gen.simulation(3,1,[1],0.1,path,0.004,-0.25,7,100,err_var)
            R2_vals.append(sim.R_squared)
            x_vals.append(err_var)

        matplotlib.rcParams.update({'font.size': 15})
        plt.figure(figsize=(7,5))
        plt.plot(x_vals, R2_vals)
        axes = plt.gca()
        axes.set_ylim([0,1])
        yticks = plt.gca().yaxis.get_major_ticks()
        yticks[0].set_visible(False)
        plt.title('Impact of ROI Error on GM Fit')
        plt.xlabel('ROI Error Variance')
        plt.ylabel('R^2')
        plt.savefig("r2.png")
        print(R2_vals)

if __name__ == '__main__':
    unittest.main()
