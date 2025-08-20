import csv
import json
import numpy as np
import matplotlib.pyplot as plt
# from utils import get_net_info

ppl_arch_figure = '/NAS/SJ/actquant/search/visualize/fig/longppl_results.png'

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


longjsd_llama_31_8b_inst_bits = [4.25390625, 4.00390625, 3.49609375, 3.25390625, 3.00390625, 2.50390625, 2.3515625, 2.3046875]
longjsd_llama_31_8b_inst_gsm8k = [0.758908264, 0.752843063, 0.758908264, 0.741470811, 0.73161486, 0.678544352, 0.652767248, 0.646702047]
longjsd_llama_31_8b_inst_longbench = [53.1325, 53.00375, 52.96625, 53.02375, 52.6975, 52.0825, 51.74375, 51.8075]
# kivi_llama_31_8b_inst_c4 = [11.38768578, 11.38815784, 11.38791084, 11.4600563, 11.47313786, 11.5138855, 14.308918, 17.41479683, 26.93492889]

jsd_llama_31_8b_inst_bits = [4.24609375, 3.99609375, 3.50390625, 3.25, 3.00390625, 2.50390625, 2.3515625, 2.3046875]
jsd_llama_31_8b_inst_gsm8k = [0.758908264, 0.758150114, 0.759666414, 0.754359363, 0.746019712, 0.680818802, 0.676269901, 0.644427597]
jsd_llama_31_8b_inst_longbench = [53.095, 52.9825, 53.07, 52.98625, 53.0125, 52.24375, 52.1475, 51.7025]
# our_llama_31_8b_inst_c4 = [11.48899174, 11.59070683, 11.73498726, 12.13742638, 12.89850044, 14.92379189, 17.19935036, 19.52639198]

longjsd_qwen25_14b_inst_bits = [4.25, 3.997395833, 3.497395833, 3.252604167, 3.002604167, 2.497395833, 2.3515625, 2.302083333]
longjsd_qwen25_14b_inst_gsm8k = [0.793783169, 0.792266869, 0.783927218, 0.774071266, 0.760424564, 0.773313116, 0.73616376, 0.73388931]
longjsd_qwen25_14b_inst_longbench = [52.095, 52.07875, 52.15375, 52.1575, 52.08, 51.4, 50.9875, 50.6325]
# kivi_qwen25_14b_inst_c4 = [10.72802544, 10.7280302, 10.7282505, 10.7498703, 10.76640415, 10.77990437, 11.61621475, 12.32918644, 14.30337906]

jsd_qwen25_14b_inst_bits = [4.25, 4.002604167, 3.502604167, 3.252604167, 3, 2.497395833, 2.348958333, 2.3046875]
jsd_qwen25_14b_inst_gsm8k = [0.805155421, 0.805155421, 0.803639121, 0.792266869, 0.786201668, 0.771038666, 0.73237301, 0.718726308]
jsd_qwen25_14b_inst_longbench = [52.20375, 52.17875, 52.26, 52.355, 52.19, 51.21875, 50.915, 50.905]
# our_qwen25_14b_inst_c4 = [10.76792622, 10.81063557, 10.8629055, 11.04475403, 11.3263588, 11.89705849, 12.41814709, 12.80280495]

longjsd_mistral_7b_inst_v03_bits = [4.24609375, 3.99609375, 3.50390625, 3.24609375, 3, 2.5, 2.3515625, 2.296875]
longjsd_mistral_7b_inst_v03_gsm8k = [0.489006823, 0.476876422, 0.46398787, 0.458680819, 0.484457923, 0.449583017, 0.423805914, 0.418498863]
longjsd_mistral_7b_inst_v03_longbench = [52.31125, 52.45, 52.465, 52.485, 52.50125, 51.9825, 52.12625, 51.8925]
# kivi_mistral_7b_inst_v03_c4 = [8.85295105, 8.853123665, 8.853259087, 8.870660782, 8.877309799, 8.8834095, 9.465094566, 9.959600449, 10.93749809]

jsd_mistral_7b_inst_v03_bits = [4.25390625, 3.99609375, 3.50390625, 3.25390625, 3, 2.50390625, 2.3515625, 2.3046875]
jsd_mistral_7b_inst_v03_gsm8k = [0.489006823, 0.492039424, 0.479150872, 0.46777862, 0.457164519, 0.426838514, 0.423805914, 0.394996209]
jsd_mistral_7b_inst_v03_longbench = [52.44625, 52.56125, 52.52875, 52.29125, 52.41625, 51.6925, 51.73875, 51.53875]
# our_mistral_7b_inst_v03_c4 = [8.887125969, 8.906814575, 8.962604523, 9.082873344, 9.23197937, 9.667222977, 10.34807587]

# model_name = 'Llama 3.1 8B Inst'
model_name = 'Qwen2.5 14B Inst'
# model_name = 'Mistral 7B Inst v0.3'

axes[0].plot(jsd_llama_31_8b_inst_bits, jsd_llama_31_8b_inst_longbench, 'o-', label='JSD', ms=marker_size, c=colors[1])
axes[0].plot(longjsd_llama_31_8b_inst_bits, longjsd_llama_31_8b_inst_longbench, 'o-', label='LongJSD', ms=marker_size, c=colors[2])
# axes[0].plot(jsd_qwen25_14b_inst_bits, jsd_qwen25_14b_inst_gsm8k, 'o-', label='JSD', ms=marker_size, c=colors[1])
# axes[0].plot(longjsd_qwen25_14b_inst_bits, longjsd_qwen25_14b_inst_gsm8k, 'o-', label='LongJSD', ms=marker_size, c=colors[2])
# axes[0].scatter(kivi_qwen25_14b_inst_bits, kivi_qwen25_14b_inst_gsm8k, label='KIVI', s=scatter_size, c=colors[0])
# axes[0].plot(our_qwen25_14b_inst_bits, our_qwen25_14b_inst_gsm8k, 'o-', label='Our', ms=marker_size, c=colors[1])
# axes[0].scatter(kivi_mistral_7b_inst_v03_bits, kivi_mistral_7b_inst_v03_gsm8k, label='KIVI', s=scatter_size, c=colors[0])
# axes[0].plot(our_mistral_7b_inst_v03_bits, our_mistral_7b_inst_v03_gsm8k, 'o-', label='Our', ms=marker_size, c=colors[1])
# axes[0].set_title('GSM8K strict-match')
axes[0].set_title('Llama 3.1 8B Inst')
axes[0].set_xlabel('Bits')
axes[0].set_ylabel('strict-match')
axes[0].set_ylabel('Avg. Acc.')
# axes[0].set_xlim([None, 5.5])
axes[0].grid(c='0.8')
axes[0].legend()

# axes[1].scatter(kivi_llama_31_8b_inst_bits, kivi_llama_31_8b_inst_longbench, s=scatter_size, label='KIVI', c=colors[0])
# axes[1].plot(jsd_llama_31_8b_inst_bits, jsd_llama_31_8b_inst_longbench, 'o-', ms=marker_size, label='JSD', c=colors[1])
# axes[1].plot(longjsd_llama_31_8b_inst_bits, longjsd_llama_31_8b_inst_longbench, 'o-', ms=marker_size, label='LongJSD', c=colors[2])
axes[1].plot(jsd_qwen25_14b_inst_bits, jsd_qwen25_14b_inst_longbench, 'o-', ms=marker_size, label='JSD', c=colors[1])
axes[1].plot(longjsd_qwen25_14b_inst_bits, longjsd_qwen25_14b_inst_longbench, 'o-', ms=marker_size, label='LongJSD', c=colors[2])
# axes[1].scatter(kivi_qwen25_14b_inst_bits, kivi_qwen25_14b_inst_longbench, label='KIVI', s=scatter_size, c=colors[0])
# axes[1].plot(our_qwen25_14b_inst_bits, our_qwen25_14b_inst_longbench, 'o-', label='Our', ms=marker_size, c=colors[1])
# axes[1].scatter(kivi_mistral_7b_inst_v03_bits, kivi_mistral_7b_inst_v03_longbench, label='KIVI', s=scatter_size, c=colors[0])
# axes[1].plot(our_mistral_7b_inst_v03_bits, our_mistral_7b_inst_v03_longbench, 'o-', label='Our', ms=marker_size, c=colors[1])
# axes[1].set_title('8 Long Bench Task Avg. Acc.')
axes[1].set_title('Qwen2.5 14B Inst')
axes[1].set_xlabel('Bits')
axes[1].set_ylabel('Avg. Acc.')
# axes[1].set_xlim([None, 5.5])
axes[1].grid(c='0.8')
axes[1].legend()

# axes[2].scatter(jsd_mistral_7b_inst_v03_bits, jsd_mistral_7b_inst_v03_longbench, s=scatter_size, label='JSD', c=colors[0])
axes[2].plot(jsd_mistral_7b_inst_v03_bits, jsd_mistral_7b_inst_v03_longbench, 'o-', ms=marker_size, label='JSD', c=colors[1])
axes[2].plot(longjsd_mistral_7b_inst_v03_bits, longjsd_mistral_7b_inst_v03_longbench, 'o-', ms=marker_size, label='LongJSD', c=colors[2])
# axes[2].scatter(kivi_qwen25_14b_inst_bits, kivi_qwen25_14b_inst_c4, label='KIVI', s=scatter_size, c=colors[0])
# axes[2].plot(our_qwen25_14b_inst_bits, our_qwen25_14b_inst_c4, 'o-', label='Our', ms=marker_size, c=colors[1])
# axes[2].scatter(kivi_mistral_7b_inst_v03_bits, kivi_mistral_7b_inst_v03_c4, label='KIVI', s=scatter_size, c=colors[0])
# axes[2].plot(our_mistral_7b_inst_v03_bits, our_mistral_7b_inst_v03_c4, 'o-', label='Our', ms=marker_size, c=colors[1])
# axes[2].set_title('C4 Perpleixty')
axes[2].set_title('Mistral 7B Inst v0.3')
axes[2].set_xlabel('Bits')
# axes[2].set_ylabel('C4 PPL')
axes[2].set_ylabel('Avg. Acc.')
# axes[2].set_xlim([None, 5.5])
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