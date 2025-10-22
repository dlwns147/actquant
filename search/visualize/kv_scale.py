import csv
import json
import numpy as np
import matplotlib.pyplot as plt
# from utils import get_net_info

# ppl_arch_figure = '/NAS/SJ/actquant/search/visualize/fig/joint_results.png'
ppl_arch_figure = '/NAS/SJ/actquant/search/visualize/fig/kv_scale.png'

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
nrows = 3
ncols = 3
size = 6
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows)) 
# fig.subplots_adjust(hspace=0.5, wspace=0.1)
marker_size = 5
scatter_size = 20

font_size = 15
plt.rc('font', size=font_size)

mixed_precision_llama_31_8b_inst_131072_mem = [10194001920, 9534185472, 9321586688, 8046518272, 7386701824, 7174103040]
mixed_precision_llama_31_8b_inst_131072_longbench = [53.8925, 53.4475, 51.35625, 52.0425, 51.73375, 47.9825]

llama_31_8b_inst_kv_scale_07_131072_mem = [8265408512, 10202652672, 8053596160, 9527369728, 7390896128, 9326043136, 7180656640, 8665964544]
llama_31_8b_inst_kv_scale_07_131072_longbench = [52.46625, 53.415, 51.57125, 53.27, 49.7975, 53.34625, 44.7375, 53.94]
idx = np.argsort(llama_31_8b_inst_kv_scale_07_131072_mem)
llama_31_8b_inst_kv_scale_07_131072_mem = np.array(llama_31_8b_inst_kv_scale_07_131072_mem)[idx]
llama_31_8b_inst_kv_scale_07_131072_longbench = np.array(llama_31_8b_inst_kv_scale_07_131072_longbench)[idx]


llama_31_8b_inst_kv_scale_01_131072_mem = [8264884224, 10202652672, 8053596160, 9527369728, 7392993280, 9330237440, 7180132352, 8667537408]
llama_31_8b_inst_kv_scale_01_131072_longbench = [52.615, 53.415, 52.98875, 53.27, 51.545, 53.23, 48.50375, 52.76125]
idx = np.argsort(llama_31_8b_inst_kv_scale_01_131072_mem)
llama_31_8b_inst_kv_scale_01_131072_mem = np.array(llama_31_8b_inst_kv_scale_01_131072_mem)[idx]
llama_31_8b_inst_kv_scale_01_131072_longbench = np.array(llama_31_8b_inst_kv_scale_01_131072_longbench)[idx]


llama_31_8b_inst_kv_scale_001_131072_mem = [8264884224, 10198982656, 8050450432, 9536282624, 7390896128, 9330761728, 7179608064, 8667537408]
llama_31_8b_inst_kv_scale_001_131072_longbench = [52.0425, 53.05, 52.32, 53.05625, 51.79375, 52.9575, 47.78625, 52.89875]
idx = np.argsort(llama_31_8b_inst_kv_scale_001_131072_mem)
llama_31_8b_inst_kv_scale_001_131072_mem = np.array(llama_31_8b_inst_kv_scale_001_131072_mem)[idx]
llama_31_8b_inst_kv_scale_001_131072_longbench = np.array(llama_31_8b_inst_kv_scale_001_131072_longbench)[idx]


llama_31_8b_inst_kv_scale_0001_131072_mem = [8264884224, 10198982656, 8050450432, 9536282624, 7390896128, 9330761728, 7179608064, 8668061696]
llama_31_8b_inst_kv_scale_0001_131072_longbench = [52.0425, 53.05, 52.32, 53.05625, 51.79375, 52.9575, 47.78625, 52.4925]
idx = np.argsort(llama_31_8b_inst_kv_scale_0001_131072_mem)
llama_31_8b_inst_kv_scale_0001_131072_mem = np.array(llama_31_8b_inst_kv_scale_0001_131072_mem)[idx]
llama_31_8b_inst_kv_scale_0001_131072_longbench = np.array(llama_31_8b_inst_kv_scale_0001_131072_longbench)[idx]


llama_31_8b_inst_kv_scale_00001_131072_mem = [8265408512, 10198982656, 8050450432, 9536282624, 7390896128, 9330761728, 7179608064, 8668061696]
llama_31_8b_inst_kv_scale_00001_131072_longbench = [51.26375, 53.05, 52.32, 53.05625, 51.79375, 52.9575, 47.78625, 52.4925]
idx = np.argsort(llama_31_8b_inst_kv_scale_00001_131072_mem)
llama_31_8b_inst_kv_scale_00001_131072_mem = np.array(llama_31_8b_inst_kv_scale_00001_131072_mem)[idx]
llama_31_8b_inst_kv_scale_00001_131072_longbench = np.array(llama_31_8b_inst_kv_scale_00001_131072_longbench)[idx]



# axes[0, 0].plot(our_comb_jsd_llama_31_8b_inst_1024_mem, our_comb_jsd_llama_31_8b_inst_1024_longbench, 'o-', label='Our JSD Appr.', ms=marker_size, c=colors[-1])
# axes[0, 0].plot(our_llama_31_8b_inst_1024_mem, our_llama_31_8b_inst_1024_longbench, 'o-', label='Our Mixed-Precision', ms=marker_size, c=colors[0])
axes[0, 0].scatter(5666250752, 50.39125, label='w4g-1 / kv4g128', s=scatter_size, c=colors[1])
axes[0, 0].scatter(5649473536, 49.09, label='w4g-1 / kv2g128', s=scatter_size, c=colors[2])
axes[0, 0].scatter(5006434304, 52.24125, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
axes[0, 0].scatter(4989657088, 50.07625, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
axes[0, 0].scatter(4793835520, 39.7775, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
axes[0, 0].scatter(4777058304, 35.9175, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
axes[0, 0].set_ylabel('Long Bench Avg.')

# axes[0, 0].plot(our_llama_31_8b_inst_1024_mem, our_llama_31_8b_inst_1024_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
# axes[0, 0].scatter(5878849536, 12.26306725, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
# axes[0, 0].scatter(5862072320, 34.28988266, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
# axes[0, 0].scatter(5006434304, 14.90471745, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
# axes[0, 0].scatter(4989657088, 74.07248688, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
# axes[0, 0].scatter(4793835522, 08.66698265, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# # axes[0, 0].scatter(4777058304, 395.6902771, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
# axes[0, 0].set_ylabel('C4 PPL')

axes[0, 0].set_title('Llama 3.1 8B Inst')
axes[0, 0].set_xlabel('Memory')
# axes[0, 0].set_xlim([None, 5.5])
axes[0, 0].grid(c='0.8')
axes[0, 0].legend()

axes[0, 1].plot(mixed_precision_llama_31_8b_inst_131072_mem, mixed_precision_llama_31_8b_inst_131072_longbench, 'o-', label='Mixed-Precision+KIVI Baseline', ms=marker_size, c='black')
axes[0, 1].plot(llama_31_8b_inst_kv_scale_07_131072_mem, llama_31_8b_inst_kv_scale_07_131072_longbench, 'o-', label='JSD Appr. KV scale 0.7', ms=marker_size, c=colors[0])
axes[0, 1].plot(llama_31_8b_inst_kv_scale_01_131072_mem, llama_31_8b_inst_kv_scale_01_131072_longbench, 'o-', label='JSD Appr. KV scale 0.1', ms=marker_size, c=colors[1])
axes[0, 1].plot(llama_31_8b_inst_kv_scale_001_131072_mem, llama_31_8b_inst_kv_scale_001_131072_longbench, 'o-', label='JSD Appr. KV scale 0.01', ms=marker_size, c=colors[2])
axes[0, 1].plot(llama_31_8b_inst_kv_scale_0001_131072_mem, llama_31_8b_inst_kv_scale_0001_131072_longbench, 'o-', label='JSD Appr. KV scale 0.001', ms=marker_size, c=colors[3])
axes[0, 1].plot(llama_31_8b_inst_kv_scale_00001_131072_mem, llama_31_8b_inst_kv_scale_00001_131072_longbench, 'o-', label='JSD Appr. KV scale 0.0001', ms=marker_size, c=colors[4])

axes[0, 1].scatter(10194001920, 50.39125, label='AWQ+KIVI Baseline', s=scatter_size, c='gray')
axes[0, 1].scatter(8046518272, 49.09, s=scatter_size, c=colors[-1])
axes[0, 1].scatter(9534185472, 52.24125, s=scatter_size, c=colors[-1])
axes[0, 1].scatter(7386701824, 50.07625, s=scatter_size, c=colors[-1])
axes[0, 1].scatter(9321586688, 39.7775, s=scatter_size, c=colors[-1])
axes[0, 1].scatter(7174103040, 35.9175, s=scatter_size, c=colors[-1])
# axes[0, 1].scatter(10194001920, 50.39125, label='w4g-1 / kv4g128', s=scatter_size, c=colors[-1])
# axes[0, 1].scatter(8046518272, 49.09, label='w4g-1 / kv2g128', s=scatter_size, c=colors[-1])
# axes[0, 1].scatter(9534185472, 52.24125, label='w3g128 / kv4g128', s=scatter_size, c=colors[-1])
# axes[0, 1].scatter(7386701824, 50.07625, label='w3g128 / kv2g128', s=scatter_size, c=colors[-1])
# axes[0, 1].scatter(9321586688, 39.7775, label='w3g-1 / kv4g128', s=scatter_size, c=colors[-1])
# axes[0, 1].scatter(7174103040, 35.9175, label='w3g-1 / kv2g128', s=scatter_size, c=colors[-1])
axes[0, 1].set_ylabel('Long Bench Avg.')
axes[0, 1].set_ylim([40, None])


# axes[0, 1].plot(our_llama_31_8b_inst_1048576_mem2, our_llama_31_8b_inst_1048576_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
# axes[0, 1].scatter(42350419968, 12.26306725, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
# axes[0, 1].scatter(25170550784, 34.28988266, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
# axes[0, 1].scatter(41478004736, 14.90471745, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
# axes[0, 1].scatter(24298135552, 74.07248688, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
# axes[0, 1].scatter(41265405952, 28.66698265, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# # axes[0, 1].scatter(24085536768, 395.6902771, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
# axes[0, 1].set_ylabel('C4 PPL')

axes[0, 1].set_title('Llama 3.1 8B Inst')
axes[0, 1].set_xlabel('Memory')
# axes[0, 1].set_xlim([None, 5.5])
axes[0, 1].grid(c='0.8')
axes[0, 1].legend()

# axes[0, 2].plot(our_comb_jsd_llama_31_8b_inst_1048576_mem, our_comb_jsd_llama_31_8b_inst_1048576_longbench, 'o-', label='Our JSD Appr.', ms=marker_size, c=colors[-1])
# axes[0, 2].plot(our_llama_31_8b_inst_1048576_mem, our_llama_31_8b_inst_1048576_longbench, 'o-', label='Our JSD Appr.', ms=marker_size, c=colors[0])
# axes[0, 2].plot(our_llama_31_8b_inst_1048576_mem, our_llama_31_8b_inst_1048576_longbench, 'o-', label='Our Mixed-Precision', ms=marker_size, c=colors[0])
axes[0, 2].scatter(42137821184, 50.39125, label='w4g-1 / kv4g128', s=scatter_size, c=colors[1])
axes[0, 2].scatter(24957952000, 49.09, label='w4g-1 / kv2g128', s=scatter_size, c=colors[2])
axes[0, 2].scatter(41478004736, 52.24125, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
axes[0, 2].scatter(24298135552, 50.07625, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
axes[0, 2].scatter(41265405952, 39.7775, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
axes[0, 2].scatter(24085536768, 35.9175, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
axes[0, 2].set_ylabel('Long Bench Avg.')

# axes[0, 2].plot(our_llama_31_8b_inst_1048576_mem2, our_llama_31_8b_inst_1048576_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
# axes[0, 2].scatter(42350419968, 12.26306725, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
# axes[0, 2].scatter(25170550784, 34.28988266, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
# axes[0, 2].scatter(41478004736, 14.90471745, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
# axes[0, 2].scatter(24298135552, 74.07248688, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
# axes[0, 2].scatter(41265405952, 28.66698265, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# # axes[0, 2].scatter(24085536768, 395.6902771, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
# axes[0, 2].set_ylabel('C4 PPL')

axes[0, 2].set_title('Llama 3.1 8B Inst')
axes[0, 2].set_xlabel('Memory')
# axes[0, 2].set_xlim([None, 5.5])
axes[0, 2].grid(c='0.8')
axes[0, 2].legend()


# axes[1, 0].plot(our_qwen25_14b_inst_1024_mem, our_qwen25_14b_inst_1024_longbench, 'o-', label='Our', ms=marker_size, c=colors[0])
# axes[1, 0].scatter(10196035584, 52.245, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
# axes[1, 0].scatter(10170869760, 50.535, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
# axes[1, 0].scatter(8544528384, 50.85375, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
# axes[1, 0].scatter(8519362560, 48.92125, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
# axes[1, 0].scatter(8140302336, 45.705, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# axes[1, 0].scatter(8115136512, 42.2175, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
# axes[1, 0].set_ylabel('Long Bench Avg.')

# axes[1, 0].plot(our_qwen25_14b_inst_1024_mem, our_qwen25_14b_inst_1024_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
axes[1, 0].scatter(10196035584, 11.07064342, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
axes[1, 0].scatter(10170869761, 05.28642178, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
axes[1, 0].scatter(8544528384, 12.19672394, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
axes[1, 0].scatter(8519362562, 01.54403305, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
axes[1, 0].scatter(8140302336, 14.65903187, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
axes[1, 0].scatter(8115136512, 34.87562943, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
axes[1, 0].set_ylabel('C4 PPL')

axes[1, 0].set_title('Qwen2.5 14B Inst')
axes[1, 0].set_xlabel('Memory')
# axes[1, 0].set_xlim([None, 5.5])
axes[1, 0].grid(c='0.8')
axes[1, 0].legend()

# axes[2, 0].plot(our_mistral_7b_inst_v03_1024_mem, our_mistral_7b_inst_v03_1024_longbench, 'o-', label='Our', ms=marker_size, c=colors[0])
# axes[2, 0].scatter(4289208320, 52.19875, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
# axes[2, 0].scatter(4272431104, 50.515, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
# axes[2, 0].scatter(3416793088, 49.14625, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
# axes[2, 0].scatter(3400015872, 47.07125, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
# axes[2, 0].scatter(3204194304, 33.96125, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# axes[2, 0].scatter(3187417088, 31.93125, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
# axes[2, 0].set_ylabel('Long Bench Avg.')

# axes[2, 0].plot(our_mistral_7b_inst_v03_1024_mem, our_mistral_7b_inst_v03_1024_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
axes[2, 0].scatter(4289208320, 9.066065788, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
axes[2, 0].scatter(4272431104, 11.58375931, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
axes[2, 0].scatter(3416793088, 9.78512001, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
axes[2, 0].scatter(3400015872, 14.18944263, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
axes[2, 0].scatter(3204194304, 16.95111656, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
axes[2, 0].scatter(3187417088, 44.03146362, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
axes[2, 0].set_ylabel('C4 PPL')

axes[2, 0].set_title('Mistral 7B Inst v0.3')
axes[2, 0].set_xlabel('Memory')
# axes[2, 0].set_xlim([None, 5.5])
axes[2, 0].grid(c='0.8')
axes[2, 0].legend()

# axes[1, 2].plot(our_qwen25_14b_inst_1048576_mem, our_qwen25_14b_inst_1048576_longbench, 'o-', label='Our', ms=marker_size, c=colors[0])
axes[1, 2].scatter(64903391232, 52.245, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
axes[1, 2].scatter(39133587456, 50.535, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
axes[1, 2].scatter(63251884032, 50.85375, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
axes[1, 2].scatter(37482080256, 48.92125, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
axes[1, 2].scatter(62847657984, 45.705, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
axes[1, 2].scatter(37077854208, 42.2175, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
axes[1, 2].set_ylabel('Long Bench Avg.')

# axes[1, 2].plot(our_qwen25_14b_inst_1048576_mem, our_qwen25_14b_inst_1048576_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
# axes[1, 2].scatter(64903391232, 11.12086391, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
# axes[1, 2].scatter(39133587456, 15.28642178, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
# axes[1, 2].scatter(63251884032, 12.19672394, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
# axes[1, 2].scatter(37482080256, 21.54403305, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
# axes[1, 2].scatter(62847657984, 14.65903187, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# axes[1, 2].scatter(37077854208, 34.87562943, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
# axes[1, 2].set_ylabel('C4 PPL')

axes[1, 2].set_title('Qwen2.5 14B Inst')
axes[1, 2].set_xlabel('Memory')
# axes[1, 2].set_xlim([None, 5.5])
axes[1, 2].grid(c='0.8')
axes[1, 2].legend()

# axes[2, 2].plot(our_mistral_7b_inst_v03_1048576_mem, our_mistral_7b_inst_v03_1048576_longbench, 'o-', label='Our', ms=marker_size, c=colors[0])
axes[2, 2].scatter(40760778752, 52.19875, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
axes[2, 2].scatter(23580909568, 50.515, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
axes[2, 2].scatter(39888363520, 49.14625, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
axes[2, 2].scatter(22708494336, 47.07125, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
axes[2, 2].scatter(39675764736, 33.96125, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
axes[2, 2].scatter(22495895552, 31.93125, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
axes[2, 2].set_ylabel('Long Bench Avg.')

# axes[2, 2].plot(our_mistral_7b_inst_v03_1048576_mem, our_mistral_7b_inst_v03_1048576_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
# axes[2, 2].scatter(40760778752, 9.066065788, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
# axes[2, 2].scatter(23580909568, 11.58375931, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
# axes[2, 2].scatter(39888363520, 9.78512001, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
# axes[2, 2].scatter(22708494336, 14.18944263, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
# axes[2, 2].scatter(39675764736, 16.95111656, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# axes[2, 2].scatter(22495895552, 44.03146362, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
# axes[2, 2].set_ylabel('C4 PPL')

axes[2, 2].set_title('Mistral 7B Inst v0.3')
axes[2, 2].set_xlabel('Memory')
# axes[2, 2].set_xlim([None, 5.5])
axes[2, 2].grid(c='0.8')
axes[2, 2].legend()



plt.tight_layout()
plt.savefig(ppl_arch_figure, dpi=300)