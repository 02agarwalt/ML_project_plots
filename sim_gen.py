import numpy as np
from ndmg.nuis import nuis
import nibabel as nb
import matplotlib
import matplotlib.pyplot as plt
import os

class simulation():
    
    def __init__(self, period, amp, wm_vars, csf_var, dirname, a, b, c, ts, err_var):
        self.R_squared = self.check_brains(period, amp, wm_vars, csf_var, dirname, a, b, c, ts, err_var)

    def create_brains(self, truth, period, amp, wm_vars, csf_var, dirname, a, b, c, ts, err_var):
        timesteps = ts

        wmmask = np.zeros((3,1,1))
        wmmask[0] = 1
        img = nb.Nifti1Image(wmmask, np.eye(4))
        img.to_filename(dirname + "wm_mask.nii.gz")

        lvmask = np.zeros((3,1,1))
        lvmask[1] = 1
        img = nb.Nifti1Image(lvmask, np.eye(4))
        img.to_filename(dirname + "lv_mask.nii.gz")
        
        samp_wmroi = np.random.normal(0, wm_vars[0], (1,1,1,timesteps))
        for i in wm_vars[1:]:
            samp_wmroi = samp_wmroi + np.random.normal(0, i, (1,1,1,timesteps))

        #samp_wmroi = np.random.normal(0, wm_vars[0], (1,1,1,timesteps))
        #yo = []
        #for i in wm_vars[1:]:
        #    hello = np.random.normal(0, i, (1,1,1,timesteps))
        #    yo.append(hello)
        #for hello in yo:
        #    samp_wmroi = samp_wmroi + hello

        samp_csfroi = np.random.normal(0.5, csf_var, (1,1,1,timesteps))

        temp = np.ndarray((1,1,1,timesteps))
        counter = 1
        for i in range(0,timesteps):
            temp[0,0,0,i] = counter
            counter = counter + 1
        drift = a*temp*temp + b*temp + c

        # temp sine stuff
        sineroitemp = 0.2*amp * np.sin(np.arange(0,timesteps*period/10,0.1*period))
        x = np.zeros((1,1,1,timesteps))
        x[0] = sineroitemp

        sineroitemp = 0.2*amp * np.sin(np.arange(0,timesteps*2*period/10,0.1*2*period))
        y = np.zeros((1,1,1,timesteps))
        y[0] = sineroitemp


        obs_wmroi = samp_wmroi + x + y + drift
        obs_csfroi = samp_csfroi + drift
        obs_gmroi = samp_wmroi + samp_csfroi + truth + drift

        err_wmroi = np.random.normal(0, err_var, (1,1,1,timesteps))
        err_csfroi = np.random.normal(0, err_var, (1,1,1,timesteps))
        err_gmroi = np.random.normal(0, err_var, (1,1,1,timesteps))

        obs_wmroi = obs_wmroi + err_wmroi
        obs_csfroi = obs_csfroi + err_csfroi
        obs_gmroi = obs_gmroi + err_gmroi
        
        brain1 = np.zeros((3,1,1,timesteps))
        brain1[0] = obs_wmroi
        brain1[1] = obs_csfroi
        brain1[2] = obs_gmroi
        
        img = nb.Nifti1Image(brain1, np.eye(4))
        img.to_filename(dirname + "sim_fmri.nii.gz")
            
    def check_brains(self, period, amp, wm_vars, csf_var, dirname, a, b, c, ts, err_var):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        timesteps = ts
        sineroitemp = amp * np.sin(np.arange(0,timesteps*period/10,0.1*period))
        sineroi = np.zeros((1,1,1,timesteps))
        sineroi[0] = sineroitemp
        
        truth = sineroi
        self.create_brains(truth, period, amp, wm_vars, csf_var, dirname, a, b, c, ts, err_var)

        obj = nuis()
        obj.regress_nuisance(dirname + "sim_fmri.nii.gz", dirname + "result.nii.gz", dirname + "wm_mask.nii.gz", dirname + "lv_mask.nii.gz", 1)

        # plot stuff
        sim_im = nb.load(dirname + "sim_fmri.nii.gz")
        sim = sim_im.get_data()
        
        result_im = nb.load(dirname + "result.nii.gz")
        result = result_im.get_data()

        plt.figure(figsize=(7,6))
        plt.subplot(2, 1, 1)
        plt.plot(result[2,0,0], label='post-correction', color='r')
        plt.plot(truth[0,0,0], label='latent')
        legend = plt.legend(loc='upper left')
        axes = plt.gca()
        axes.xaxis.set_major_locator(plt.NullLocator())
        axes.set_ylim([-2,4])
        plt.ylabel('Intensity')
        plt.title('GM Signal Comparison')

        plt.subplot(2, 1, 2)
        plt.plot(sim[0,0,0], label='WM')
        plt.plot(sim[1,0,0], label='CSF')
        plt.plot(sim[2,0,0], label='GM')
        legend = plt.legend(loc='upper left')
        axes = plt.gca()
        axes.set_ylim([-2,25])
        plt.ylabel('Intensity')
        plt.xlabel('Timestep')
        plt.title('Pre-correction ROIs')

        #plt.subplot(1, 3, 3)
        #plt.plot(result[0,0,0], label='WM')
        #plt.plot(result[1,0,0], label='CSF')
        #plt.plot(result[2,0,0], label='GM')
        #legend = plt.legend(loc='upper right')
        #axes = plt.gca()
        #axes.set_ylim([-6,60])
        #plt.title('All ROIs post-correction')
        
        plt.savefig(dirname + "result.png")
        plt.clf()

        # R^2 calculation
        ymean = np.mean(truth)
        ss_tot = np.sum((truth-ymean)*(truth-ymean))
        ss_res = np.sum((truth-result[2])*(truth-result[2]))
        return 1 - (float(ss_res)/float(ss_tot))
