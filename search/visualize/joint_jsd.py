import csv
import json
import numpy as np
import matplotlib.pyplot as plt
# from utils import get_net_info

output_file = '/NAS/SJ/actquant/search/visualize/fig/joint_jsd_v2.png'

colors = [
    '#FF6663',
    '#939393',
    '#C83C04',
    '#378375',
    '#6699FF',
    '#FACA00',
    '#2351AB',
    '#736363',
]

# colors = [
#     '#535353',
#     '#2351AB',
#     '#C83C04',
#     '#FACA00',
#     '#378375',
#     '#6699FF'
# ]
nrows = 1
ncols = 1
size = 6
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows)) 
# fig.subplots_adjust(hspace=0.5, wspace=0.1)
marker_size = 5
scatter_size = 5

font_size = 15
plt.rc('font', size=font_size)

# result_file = '/NAS/SJ/actquant/search/save/result/2509060835_Llama-3.1-8B-Instruct_random_sample__1000_sample_seed_wikitext2/results.csv'
result_file = '/NAS/SJ/actquant/search/save/result/2509071826_Llama-3.1-8B-Instruct_random_sample_hqq_kivi_1000_sample_seed_wikitext2/results.csv'
import csv
with open(result_file, 'r') as f:
    result_list = list(csv.reader(f))
jsd = list(map(float, result_list[5]))
joint_jsd = list(map(float, result_list[6]))

import scipy.stats as stats
rho, _ = stats.spearmanr(joint_jsd, jsd)
tau, _ = stats.kendalltau(joint_jsd, jsd)
print(f'rho: {rho}, tau: {tau}')
axes.scatter(joint_jsd, jsd, s=scatter_size, c=colors[0])
axes.set_title('Llama 3.1 8B Instruct')
axes.set_xlabel('Estimated JSD')
axes.set_ylabel('JSD')
# axes.set_xlim([None, 5.5])
axes.grid(c='0.8')
# axes.legend()

plt.tight_layout()
plt.savefig(output_file, dpi=300)