import csv
import numpy as np
import matplotlib.pyplot as plt

# CL
# CL + KD
# KD 
# base 

def to_plot(path, ind):
    arr = []
    with open(path) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                print(row[ind])
                continue
            arr.append(float(row[ind]))
    
    return arr, np.argmax(np.array(arr))

res_kd, max_ind_kd = to_plot('./results_distill.csv', 7)
res_baseline, max_ind_baseline = to_plot('./results.csv', 7)
res_kd_cl, max_ind_kd_cl = to_plot('./results_cl_kd_updated.csv', 7)
res_cl, max_ind_cl = to_plot('./results_cl.csv', 7)

plt.figure(figsize=(16, 8))
plt.plot(res_baseline, label='baseline')
plt.plot(res_kd, label='KD')
plt.plot(res_cl, label='CL')
plt.plot(res_kd_cl, label='KD+CL')
plt.xlabel('Epoch')
plt.ylabel('mAP50')
plt.legend()
plt.title('mAP50 vs. Epoch')
plt.savefig('./ablation_all.jpg')
plt.show()

res_kd_95, max_ind_kd_95 = to_plot('./results_distill.csv', 8)
res_baseline_95, max_ind_baseline_95 = to_plot('./results.csv', 8)
res_kd_cl_95, max_ind_kd_cl_95 = to_plot('./results_cl_kd_updated.csv', 8)
res_cl_95, max_ind_cl_95 = to_plot('./results_cl.csv', 8)

plt.figure(figsize=(16, 8))
plt.plot(res_baseline_95, label='baseline')
plt.plot(res_kd_95, label='KD')
plt.plot(res_cl_95, label='CL')
plt.plot(res_kd_cl_95, label='KD+CL')
plt.xlabel('Epoch')
plt.ylabel('mAP50-95')
plt.legend()
plt.title('mAP50-95 vs. Epoch')
plt.savefig('./ablation_mAP5095.jpg')
plt.show()
