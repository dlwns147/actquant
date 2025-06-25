import csv
import json
import numpy as np
import matplotlib.pyplot as plt
# from utils import get_net_info

ppl_arch_figure = '/NAS/SJ/actquant/search/visualize/fig/kivi_result.png'

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
ncols = 3
size = 6
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows)) 
# fig.subplots_adjust(hspace=0.5, wspace=0.1)
marker_size = 5
scatter_size = 20

font_size = 15
plt.rc('font', size=font_size)


kivi_llama_31_8b_inst_bits = [9, 8.5, 8.25, 5, 4.5, 4.25, 3, 2.5, 2.25]
kivi_llama_31_8b_inst_gsm8k = [0.754359363, 0.754359363, 0.756633813, 0.763457165, 0.764215315, 0.737680061, 0.739196361, 0.696739955, 0.636087945]
kivi_llama_31_8b_inst_longbench = [53.3075, 53.2175, 53.29625, 53.22125, 53.245, 53.35, 52.6025, 51.4125, 51.4275]
kivi_llama_31_8b_inst_c4 = [11.38768578, 11.38815784, 11.38791084, 11.4600563, 11.47313786, 11.5138855, 14.308918, 17.41479683, 26.93492889]

our_llama_31_8b_inst_bits = [4.50387168, 4.252083333, 3.99767562, 3.5, 3.00387931, 2.500336022, 2.354482323, 2.3047619]
our_llama_31_8b_inst_gsm8k = [0.757391964, 0.758908264, 0.764215315, 0.758150114, 0.736921911, 0.699014405, 0.676269901, 0.671721001]
our_llama_31_8b_inst_longbench = [53.25375, 53.26625, 53.20375, 53.15, 52.95125, 52.60375, 52.25375, 52.3175]
our_llama_31_8b_inst_c4 = [11.48899174, 11.59070683, 11.73498726, 12.13742638, 12.89850044, 14.92379189, 17.19935036, 19.52639198]

kvtuner_llama_31_8b_inst_bits = [3.0625, 3.59375, 4, 4.5, 5]
kvtuner_llama_31_8b_inst_longbench = [48.41, 49.79875, 50.3925, 53.22375, 52.87625]

kivi_qwen25_14b_inst_bits = [9, 8.5, 8.25, 5, 4.5, 4.25, 3, 2.5, 2.25]
kivi_qwen25_14b_inst_gsm8k = [0.793783169, 0.79757392, 0.79833207, 0.810462472, 0.806671721, 0.80136467, 0.783927218, 0.771796816, 0.727824109]
kivi_qwen25_14b_inst_longbench = [47.00625, 46.93125, 47.00125, 47.03375, 46.9775, 47.05, 46.90125, 46.3075, 46.13]
kivi_qwen25_14b_inst_c4 = [10.72802544, 10.7280302, 10.7282505, 10.7498703, 10.76640415, 10.77990437, 11.61621475, 12.32918644, 14.30337906]

our_qwen25_14b_inst_bits = [4.498484848, 4.250706215, 4.000706215, 3.498903509, 3.003756831, 2.50438596, 2.35301326, 2.304857002]
our_qwen25_14b_inst_gsm8k = [0.80136467, 0.811220622, 0.805913571, 0.809704321, 0.812736922, 0.769522365, 0.777103867, 0.752084913]
our_qwen25_14b_inst_longbench = [47.09625, 47.085, 46.98375, 47.17125, 47.02875, 46.82, 46.6325, 46.6875]
our_qwen25_14b_inst_c4 = [10.76792622, 10.81063557, 10.8629055, 11.04475403, 11.3263588, 11.89705849, 12.41814709, 12.80280495]

kivi_mistral_7b_inst_v03_bits = [9, 8.5, 8.25, 5, 4.5, 4.25, 3, 2.5, 2.25]
kivi_mistral_7b_inst_v03_gsm8k = [0.501137225, 0.499620925, 0.500379075, 0.488248673, 0.483699773, 0.482941622, 0.451857468, 0.439727066, 0.39878696]
kivi_mistral_7b_inst_v03_longbench = [53.105, 53.14375, 53.12875, 53.19625, 52.985, 52.91125, 52.52875, 52.09875, 52.05125]
kivi_mistral_7b_inst_v03_c4 = [8.85295105, 8.853123665, 8.853259087, 8.870660782, 8.877309799, 8.8834095, 9.465094566, 9.959600449, 10.93749809]

our_mistral_7b_inst_v03_bits = [4.496359223, 4.248842593, 3.998903509, 3.50208333, 3.003937008, 2.504076087, 2.30475427]
our_mistral_7b_inst_v03_gsm8k = [0.482941622, 0.494313874, 0.488248673, 0.473843821, 0.463229719, 0.46626232, 0.416224412]
our_mistral_7b_inst_v03_longbench = [53.1025, 53.08625, 53.0725, 52.90375, 52.76125, 52.08875, 52.28875]
our_mistral_7b_inst_v03_c4 = [8.887125969, 8.906814575, 8.962604523, 9.082873344, 9.23197937, 9.667222977, 10.34807587]

# model_name = 'Llama 3.1 8B Inst'
# model_name = 'Qwen2.5 14B Inst'
model_name = 'Mistral 7B Inst v0.3'

# axes[0].scatter(kivi_llama_31_8b_inst_bits, kivi_llama_31_8b_inst_gsm8k, label='KIVI', s=scatter_size, c=colors[0])
# axes[0].plot(our_llama_31_8b_inst_bits, our_llama_31_8b_inst_gsm8k, 'o-', label='Our', ms=marker_size, c=colors[1])
# axes[0].scatter(kivi_qwen25_14b_inst_bits, kivi_qwen25_14b_inst_gsm8k, label='KIVI', s=scatter_size, c=colors[0])
# axes[0].plot(our_qwen25_14b_inst_bits, our_qwen25_14b_inst_gsm8k, 'o-', label='Our', ms=marker_size, c=colors[1])
axes[0].scatter(kivi_mistral_7b_inst_v03_bits, kivi_mistral_7b_inst_v03_gsm8k, label='KIVI', s=scatter_size, c=colors[0])
axes[0].plot(our_mistral_7b_inst_v03_bits, our_mistral_7b_inst_v03_gsm8k, 'o-', label='Our', ms=marker_size, c=colors[1])
axes[0].set_title('GSM8K strict-match')
axes[0].set_xlabel('Bits')
axes[0].set_ylabel('strict-match')
axes[0].set_xlim([None, 5.5])
axes[0].grid(c='0.8')
axes[0].legend()

# axes[1].scatter(kivi_llama_31_8b_inst_bits, kivi_llama_31_8b_inst_longbench, s=scatter_size, label='KIVI', c=colors[0])
# axes[1].plot(our_llama_31_8b_inst_bits, our_llama_31_8b_inst_longbench, 'o-', ms=marker_size, label='Our', c=colors[1])
# axes[1].plot(kvtuner_llama_31_8b_inst_bits, kvtuner_llama_31_8b_inst_longbench, 'o-', ms=marker_size, label='KVTuner', c=colors[2])
# axes[1].scatter(kivi_qwen25_14b_inst_bits, kivi_qwen25_14b_inst_longbench, label='KIVI', s=scatter_size, c=colors[0])
# axes[1].plot(our_qwen25_14b_inst_bits, our_qwen25_14b_inst_longbench, 'o-', label='Our', ms=marker_size, c=colors[1])
axes[1].scatter(kivi_mistral_7b_inst_v03_bits, kivi_mistral_7b_inst_v03_longbench, label='KIVI', s=scatter_size, c=colors[0])
axes[1].plot(our_mistral_7b_inst_v03_bits, our_mistral_7b_inst_v03_longbench, 'o-', label='Our', ms=marker_size, c=colors[1])
axes[1].set_title('8 Long Bench Task Avg Acc')
axes[1].set_xlabel('Bits')
axes[1].set_ylabel('Avg Acc.')
axes[1].set_xlim([None, 5.5])
axes[1].grid(c='0.8')
axes[1].legend()

# axes[2].scatter(kivi_llama_31_8b_inst_bits, kivi_llama_31_8b_inst_c4, s=scatter_size, label='KIVI', c=colors[0])
# axes[2].plot(our_llama_31_8b_inst_bits, our_llama_31_8b_inst_c4, 'o-', ms=marker_size, label='Our', c=colors[1])
# axes[2].scatter(kivi_qwen25_14b_inst_bits, kivi_qwen25_14b_inst_c4, label='KIVI', s=scatter_size, c=colors[0])
# axes[2].plot(our_qwen25_14b_inst_bits, our_qwen25_14b_inst_c4, 'o-', label='Our', ms=marker_size, c=colors[1])
axes[2].scatter(kivi_mistral_7b_inst_v03_bits, kivi_mistral_7b_inst_v03_c4, label='KIVI', s=scatter_size, c=colors[0])
axes[2].plot(our_mistral_7b_inst_v03_bits, our_mistral_7b_inst_v03_c4, 'o-', label='Our', ms=marker_size, c=colors[1])
axes[2].set_title('C4 Perpleixty')
axes[2].set_xlabel('Bits')
axes[2].set_ylabel('C4 PPL')
axes[2].set_xlim([None, 5.5])
axes[2].grid(c='0.8')
axes[2].legend()

# kivi_mistral_7b_inst_v03_bits = [8.25, 5, 4.5, 4.25, 3, 2.5, 2.25]
# kivi_mistral_7b_inst_v03_gsm8k = [0.500379075, 0.488248673, 0.483699773, 0.482941622, 0.451857468, 0.439727066, 0.39878696]
# kivi_mistral_7b_inst_v03_longbench = [53.12875, 53.19625, 52.985, 52.91125, 52.52875, 52.09875, 52.05125]

# our_llama_31_8b_inst_bits = [8, 5.002770936, 3.999695122, 3.501470588, 3.00400641, 2.504076087, 2.354482323, 2.30331754]
# our_llama_31_8b_inst_gsm8k = [0.495830174, 0.486732373, 0.46398787, 0.46929492, 0.46626232, 0.445034117, 0.454131918, 0.434420015]
# our_llama_31_8b_inst_longbench = [53.08375, 52.87625, 52.36375, 52.8, 52.89375, 52.66625, 52.21875, 52.01625]

# axes[0].scatter(kivi_llama_31_8b_inst_bits, kivi_llama_31_8b_inst_gsm8k, label='KIVI', s=scatter_size, c=colors[0])
# axes[0].plot(our_llama_31_8b_inst_bits, our_llama_31_8b_inst_gsm8k, 'o-', label='Our', ms=marker_size, c=colors[1])
# axes[0].set_title('Llama 3.1 8B Inst')
# axes[0].set_xlabel('Bits')
# axes[0].set_ylabel('GSM8K strict-match')
# axes[0].grid(c='0.8')
# axes[0].legend(loc="upper left")

# axes[1].scatter(kivi_llama_31_8b_inst_bits, kivi_llama_31_8b_inst_longbench, s=scatter_size, label='KIVI', c=colors[0])
# axes[1].plot(our_llama_31_8b_inst_bits, our_llama_31_8b_inst_longbench, 'o-', ms=marker_size, label='Our', c=colors[1])
# axes[1].set_title('Llama 3.1 8B Inst')
# axes[1].set_xlabel('Bits')
# axes[1].set_ylabel('8 Long Bench Avg Acc.')
# axes[1].grid(c='0.8')
# # axes[1].set_axisbelow(True)
# axes[1].legend(loc="upper left")



plt.tight_layout()
plt.savefig(ppl_arch_figure, dpi=300)