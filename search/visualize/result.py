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
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6)) 
# fig.subplots_adjust(hspace=0.5, wspace=0.1)
marker_size = 5
scatter_size = 20


kivi_llama_31_8b_inst_bits = [8.25, 5, 4.5, 4.25, 3, 2.5, 2.25]
kivi_llama_31_8b_inst_gsm8k = [0.756633813, 0.763457165, 0.764215315, 0.737680061, 0.739196361, 0.696739955, 0.636087945]
kivi_llama_31_8b_inst_long_bench = [53.29625, 53.22125, 53.245, 53.35, 52.6025, 51.4125, 51.4275]

our_llama_31_8b_inst_bits = [4.99767562, 4.498903509, 4.248903509, 4.00393701, 3.500856164, 2.996265723, 2.504331683, 2.35209276]
our_llama_31_8b_inst_gsm8k = [0.771038666, 0.761940864, 0.758908264, 0.742228961, 0.752084913, 0.73540561, 0.690674754, 0.679302502]
our_llama_31_8b_inst_long_bench = [53.34125, 53.35875, 53.31375, 53.17375, 53.00375, 52.835, 52.51, 52.4975]


kivi_qwen25_14b_inst_bits = [8.25, 5, 4.5, 4.25, 3, 2.5, 2.25]
kivi_qwen25_14b_inst_gsm8k = [0.79833207, 0.810462472, 0.806671721, 0.80136467, 0.783927218, 0.771796816, 0.727824109]
kivi_qwen25_14b_inst_long_bench = [47.00125, 47.03375, 46.9775, 47.05, 46.90125, 46.3075, 46.13]

our_qwen25_14b_inst_bits = [4.998903509, 4.50342827, 4.248903509, 4.000706215, 3.49664297, 3.004166667, 2.503289474, 2.350788288, 2.304857002]
our_qwen25_14b_inst_gsm8k = [0.803639121, 0.808946171, 0.79833207, 0.79984837, 0.789992418, 0.777862017, 0.759666414, 0.746777862, 0.764973465]
our_qwen25_14b_inst_long_bench = [46.97125, 46.9625, 47.0325, 47.08875, 47.09, 47.07, 46.68875, 46.5575, 46.775]

model_name = 'Llama 3.1 8B Inst'
model_name = 'Qwen2.5 14B Inst'

# axes[0].scatter(kivi_llama_31_8b_inst_bits, kivi_llama_31_8b_inst_gsm8k, label='KIVI', s=scatter_size, c=colors[0])
# axes[0].plot(our_llama_31_8b_inst_bits, our_llama_31_8b_inst_gsm8k, 'o-', label='Our', ms=marker_size, c=colors[1])
axes[0].scatter(kivi_qwen25_14b_inst_bits, kivi_qwen25_14b_inst_gsm8k, label='KIVI', s=scatter_size, c=colors[0])
axes[0].plot(our_qwen25_14b_inst_bits, our_qwen25_14b_inst_gsm8k, 'o-', label='Our', ms=marker_size, c=colors[1])
axes[0].set_title(model_name)
axes[0].set_xlabel('Bits')
axes[0].set_ylabel('GSM8K strict-match')
axes[0].set_xlim([None, 5.5])
axes[0].grid(c='0.8')
axes[0].legend(loc="upper left")

# axes[1].scatter(kivi_llama_31_8b_inst_bits, kivi_llama_31_8b_inst_long_bench, s=scatter_size, label='KIVI', c=colors[0])
# axes[1].plot(our_llama_31_8b_inst_bits, our_llama_31_8b_inst_long_bench, 'o-', ms=marker_size, label='Our', c=colors[1])
axes[1].scatter(kivi_qwen25_14b_inst_bits, kivi_qwen25_14b_inst_long_bench, label='KIVI', s=scatter_size, c=colors[0])
axes[1].plot(our_qwen25_14b_inst_bits, our_qwen25_14b_inst_long_bench, 'o-', label='Our', ms=marker_size, c=colors[1])
axes[1].set_title(model_name)
axes[1].set_xlabel('Bits')
axes[1].set_ylabel('8 Long Bench Avg Acc.')
axes[1].set_xlim([None, 5.5])
axes[1].grid(c='0.8')
axes[1].legend(loc="upper left")

# kivi_mistral_7b_inst_v03_bits = [8.25, 5, 4.5, 4.25, 3, 2.5, 2.25]
# kivi_mistral_7b_inst_v03_gsm8k = [0.500379075, 0.488248673, 0.483699773, 0.482941622, 0.451857468, 0.439727066, 0.39878696]
# kivi_mistral_7b_inst_v03_long_bench = [53.12875, 53.19625, 52.985, 52.91125, 52.52875, 52.09875, 52.05125]

# our_llama_31_8b_inst_bits = [8, 5.002770936, 3.999695122, 3.501470588, 3.00400641, 2.504076087, 2.354482323, 2.30331754]
# our_llama_31_8b_inst_gsm8k = [0.495830174, 0.486732373, 0.46398787, 0.46929492, 0.46626232, 0.445034117, 0.454131918, 0.434420015]
# our_llama_31_8b_inst_long_bench = [53.08375, 52.87625, 52.36375, 52.8, 52.89375, 52.66625, 52.21875, 52.01625]

# axes[0].scatter(kivi_llama_31_8b_inst_bits, kivi_llama_31_8b_inst_gsm8k, label='KIVI', s=scatter_size, c=colors[0])
# axes[0].plot(our_llama_31_8b_inst_bits, our_llama_31_8b_inst_gsm8k, 'o-', label='Our', ms=marker_size, c=colors[1])
# axes[0].set_title('Llama 3.1 8B Inst')
# axes[0].set_xlabel('Bits')
# axes[0].set_ylabel('GSM8K strict-match')
# axes[0].grid(c='0.8')
# axes[0].legend(loc="upper left")

# axes[1].scatter(kivi_llama_31_8b_inst_bits, kivi_llama_31_8b_inst_long_bench, s=scatter_size, label='KIVI', c=colors[0])
# axes[1].plot(our_llama_31_8b_inst_bits, our_llama_31_8b_inst_long_bench, 'o-', ms=marker_size, label='Our', c=colors[1])
# axes[1].set_title('Llama 3.1 8B Inst')
# axes[1].set_xlabel('Bits')
# axes[1].set_ylabel('8 Long Bench Avg Acc.')
# axes[1].grid(c='0.8')
# # axes[1].set_axisbelow(True)
# axes[1].legend(loc="upper left")



plt.tight_layout()
plt.savefig(ppl_arch_figure, dpi=300)