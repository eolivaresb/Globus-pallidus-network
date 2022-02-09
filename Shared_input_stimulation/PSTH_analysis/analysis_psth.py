import numpy as np
##########################################################
N = 1000
dtbin = 0.0005  # 4 ms
lb, ub = -0.025, 0.175
hrange = [lb, ub]
tbin = np.arange(lb, ub, dtbin)
bins = len(tbin)
ppre = int(np.abs(lb) / dtbin)  # time bins before pulse
############### load population histograms################
sims = ['net', 'con', 'ran', 'gcl', 'gcl_ran']
sims = ['gcl_ran']
stimulis = [400, 600, 800, 1000, 1200, 1400]

###########################################################
##############         PSTH analysis          #############
###########################################################
'''
Data structure
N = 1000 rows, one per neuron
--- 8 columns with psth analysis:
0: 'baseline',  Baseline rate
1: 'peak',      Peak primary response
2: 'area',      Area of primary response
3: 'Tprim',     Width of primary response (time crossing above baseline)
4: 'tpeak',     Time for peak in the primary response
5: 'hwidth',    Half width rpimary response
6: 'postpeak'   Peak secondary response
7: 'tpostpeak'  Time for peak in secondary response
'''
for k, stim in enumerate(stimulis):
    for sim in sims:
        data = np.zeros((N, 8))
        hists = np.load('data/hist_%s_%d.npy' %
                        (sim, k), allow_pickle=True)  # 1 nS (3)
        for n in range(N):
            def interp_hist(x): return np.interp(
                x, 1000 * (tbin + dtbin / 2), hists[n])
            # reduce time tbin to x 0.1
            time = np.linspace(-25, 175, bins * 10)
            hf = interp_hist(time)  # histogram interpolated bin 0.1 ms
            ppre = np.where(time <= 0)[0][-1]
            baseline = np.mean(hf[:ppre])
            if baseline > 1:  # more than 1 Hz in average
                # data points in the average isis
                cycleindx = np.min([ppre + int(1 / baseline / (0.1 * dtbin)), 6*ppre])
                cs = 0.1 * dtbin * np.cumsum(hf - baseline)  # cummulative PSTH
                # constrain peak to first cycle
                ########################################
                ############  Primary response #########
                peak = np.min(hf[ppre + 1:cycleindx])  # minimum prim response
                endfirstresp = ppre + 39 + \
                    np.where(hf[ppre + 40:cycleindx] >
                             baseline)[0][0]  # position of crossing baseline up
                area = cs[endfirstresp]  # area from cummulative PTSH
                Tprim = time[endfirstresp]  # Time of crossing baseline
                # time of primary response
                tpeak = time[ppre + 1 + np.argmin(hf[ppre + 1:cycleindx])]
                # half width time
                mp = (baseline + peak) / 2  # midpoint between peak and baseline
                hp0 = np.where(hf[:cycleindx] < mp)[
                    0][0]  # first time below mp
                hp1 = np.where(hf[:cycleindx] < mp)[
                    0][-1]  # last time below mp
                hwidth = (time[hp1] - time[hp0])
                ########################################
                ############  Secondary response #######
                # peak secondary response
                postpeak = np.max(hf[ppre + 1:cycleindx])
                # seondary response time
                tpostpeak = time[ppre + 1 + np.argmax(hf[ppre + 1:cycleindx])]
                ########################################
                ############     save data       #######
                data[n] = [baseline, peak-baseline, area, Tprim,
                           tpeak, hwidth, postpeak-baseline, tpostpeak]
            else:
                #add zeros if baseline < 1Hz
                data[n] = [baseline, 0, 0, 0, 0, 0, 0, 0]
        np.save('data/hist_analysis_%s_%d.npy' % (sim, stim), data)
        # np.savetxt('data_text_files/hist_analysis_%s_%d.dat' % (sim, stim), data)
