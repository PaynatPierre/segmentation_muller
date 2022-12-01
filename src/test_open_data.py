import scipy.io

f = scipy.io.loadmat('./../data/L2_D18_S2_PSG_conf_20um_10avg_1000P_5x_a6_22-Apr-2020_PSG6procZ_MM_R_C.mat')
print(f['Fin_MM_avgZ'].shape)
print('execution end')