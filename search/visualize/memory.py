import csv
import json
import numpy as np
import matplotlib.pyplot as plt
# from utils import get_net_info

ppl_arch_figure = '/NAS/SJ/actquant/search/visualize/fig/memory_results.png'

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
nrows = 2
ncols = 3
size = 6
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows)) 
# fig.subplots_adjust(hspace=0.5, wspace=0.1)
marker_size = 5
scatter_size = 20

font_size = 15
plt.rc('font', size=font_size)


our_llama_31_8b_inst_1024_mem = [5876686848, 5862006784, 5008596992, 4992770048, 4794392576, 4780892160]
our_llama_31_8b_inst_1024_longbench = [53.1225, 52.79, 50.74625, 50.65, 48.63625, 48.2725]
our_llama_31_8b_inst_1024_c4 = [12.2087965, 12.24480724, 15.16447544, 15.46710396, 17.71403122, 18.25587463]

# our_llama_31_8b_inst_1048576_mem = [42312146944, 25181036544, 41338019840, 41204326400, 24103624704]
our_llama_31_8b_inst_1048576_mem = [42312146944, 41338019840, 41204326400, 25181036544, 24103624704]
our_llama_31_8b_inst_1048576_mem2 = [42312146944, 41338019840, 41204326400, 25181036544, 24293416960, 24103624704]
our_llama_31_8b_inst_1048576_longbench = [52.9225, 52.81375, 52.8125, 48.6075, 44.4075]
our_llama_31_8b_inst_1048576_c4 = [12.30204964, 12.39396954, 12.39116287, 32.80329895, 80.79725647, 109.1700058]

# our_qwen25_7b_inst_1024_mem = [5653232640, 5653232640, 4857920512, 4850613248, 4659690496, 4651793408]
# our_qwen25_7b_inst_1024_longbench = [51.63875, 51.63875, 46.5425, 46.82625, 41.9175, 41.55]

# our_qwen25_7b_inst_1048576_mem = [21591440384, 14117583872, 20752579584, 13307853824, 20601584640, 13108755456]
# our_qwen25_7b_inst_1048576_longbench = [51.57375, 48.08, 51.43625, 39.72625, 51.44875, 33.37875]

our_qwen25_14b_inst_1024_mem = [10199181312, 10172606464, 8546723840, 8517232640, 8144824320, 8105469952]
our_qwen25_14b_inst_1024_longbench = [52.25875, 52.3725, 50.87, 50.62625, 50.1125, 49.14375]
our_qwen25_14b_inst_1024_c4 = [11.07064342, 11.07767773, 12.21304131, 12.27282238, 13.23994827, 13.2931118]

# our_qwen25_14b_inst_1048576_mem = [64761964544, 39148201984, 63219116032, 37516814336, 62749353984, 37110491136]
# our_qwen25_14b_inst_1048576_longbench = [51.48125, 50.305, 52.22625, 46.14625, 52.16125, 42.8575]
our_qwen25_14b_inst_1048576_mem = [64761964544, 63219116032, 62749353984, 39148201984, 37516814336, 37110491136]
our_qwen25_14b_inst_1048576_longbench = [51.48125, 52.22625, 52.16125, 50.305, 46.14625, 42.8575]
our_qwen25_14b_inst_1048576_c4 = [11.12086391, 11.16060066, 11.17476559, 15.09971619, 25.46052551, 32.95238876]

our_mistral_7b_inst_v03_1024_mem = [4286586880, 4271775744, 3417088000, 3394641920, 3205013504, 3189121024]
our_mistral_7b_inst_v03_1024_longbench = [52.14375, 52.50125, 50.02875, 49.945, 49.25625, 48.51125]
our_mistral_7b_inst_v03_1024_c4 = [9.057579041, 9.069731712, 9.813754082, 9.852708817, 10.61759567, 10.6838007]

# our_mistral_7b_inst_v03_1048576_mem = [40691572736, 23583006720, 39919820800, 39651385344, 22516604928]
# our_mistral_7b_inst_v03_1048576_longbench = [52.52, 50.25, 52.4, 52.45875, 46.94125]
our_mistral_7b_inst_v03_1048576_mem = [40691572736, 39919820800, 39651385344, 23583006720, 22516604928]
our_mistral_7b_inst_v03_1048576_mem2 = [40691572736, 39919820800, 39651385344, 23583006720, 22728417280, 22516604928]
our_mistral_7b_inst_v03_1048576_longbench = [52.52, 52.4, 52.45875, 50.25, 46.94125]
our_mistral_7b_inst_v03_1048576_c4 = [9.082056999, 9.093878746, 9.101392746, 11.4829216, 15.13122749, 18.23580742]

# axes[0, 0].plot(our_llama_31_8b_inst_1024_mem, our_llama_31_8b_inst_1024_longbench, 'o-', label='Our', ms=marker_size, c=colors[0])
# axes[0, 0].scatter(5878849536, 53.0625, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
# axes[0, 0].scatter(5862072320, 51.61375, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
# axes[0, 0].scatter(5006434304, 52.24125, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
# axes[0, 0].scatter(4989657088, 50.07625, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
# axes[0, 0].scatter(4793835520, 39.7775, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# axes[0, 0].scatter(4777058304, 35.9175, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
# axes[0, 0].set_ylabel('Long Bench Avg.')

axes[0, 0].plot(our_llama_31_8b_inst_1024_mem, our_llama_31_8b_inst_1024_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
axes[0, 0].scatter(5878849536, 12.26306725, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
axes[0, 0].scatter(5862072320, 34.28988266, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
axes[0, 0].scatter(5006434304, 14.90471745, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
axes[0, 0].scatter(4989657088, 74.07248688, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
axes[0, 0].scatter(4793835520, 28.66698265, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# axes[0, 0].scatter(4777058304, 395.6902771, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
axes[0, 0].set_ylabel('C4 PPL')

axes[0, 0].set_title('Llama 3.1 8B Inst')
axes[0, 0].set_xlabel('Memory')
# axes[0, 0].set_xlim([None, 5.5])
axes[0, 0].grid(c='0.8')
axes[0, 0].legend()

# axes[0, 1].plot(our_qwen25_14b_inst_1024_mem, our_qwen25_14b_inst_1024_longbench, 'o-', label='Our', ms=marker_size, c=colors[0])
# axes[0, 1].scatter(10196035584, 52.245, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
# axes[0, 1].scatter(10170869760, 50.535, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
# axes[0, 1].scatter(8544528384, 50.85375, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
# axes[0, 1].scatter(8519362560, 48.92125, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
# axes[0, 1].scatter(8140302336, 45.705, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# axes[0, 1].scatter(8115136512, 42.2175, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
# axes[0, 1].set_ylabel('Long Bench Avg.')

axes[0, 1].plot(our_qwen25_14b_inst_1024_mem, our_qwen25_14b_inst_1024_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
axes[0, 1].scatter(10196035584, 11.07064342, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
axes[0, 1].scatter(10170869760, 15.28642178, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
axes[0, 1].scatter(8544528384, 12.19672394, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
axes[0, 1].scatter(8519362560, 21.54403305, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
axes[0, 1].scatter(8140302336, 14.65903187, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
axes[0, 1].scatter(8115136512, 34.87562943, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
axes[0, 1].set_ylabel('C4 PPL')

axes[0, 1].set_title('Qwen2.5 14B Inst')
axes[0, 1].set_xlabel('Memory')
# axes[0, 1].set_xlim([None, 5.5])
axes[0, 1].grid(c='0.8')
axes[0, 1].legend()

# axes[0, 2].plot(our_mistral_7b_inst_v03_1024_mem, our_mistral_7b_inst_v03_1024_longbench, 'o-', label='Our', ms=marker_size, c=colors[0])
# axes[0, 2].scatter(4289208320, 52.19875, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
# axes[0, 2].scatter(4272431104, 50.515, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
# axes[0, 2].scatter(3416793088, 49.14625, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
# axes[0, 2].scatter(3400015872, 47.07125, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
# axes[0, 2].scatter(3204194304, 33.96125, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# axes[0, 2].scatter(3187417088, 31.93125, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
# axes[0, 2].set_ylabel('Long Bench Avg.')

axes[0, 2].plot(our_mistral_7b_inst_v03_1024_mem, our_mistral_7b_inst_v03_1024_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
axes[0, 2].scatter(4289208320, 9.066065788, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
axes[0, 2].scatter(4272431104, 11.58375931, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
axes[0, 2].scatter(3416793088, 9.78512001, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
axes[0, 2].scatter(3400015872, 14.18944263, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
axes[0, 2].scatter(3204194304, 16.95111656, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
axes[0, 2].scatter(3187417088, 44.03146362, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
axes[0, 2].set_ylabel('C4 PPL')

axes[0, 2].set_title('Mistral 7B Inst v0.3')
axes[0, 2].set_xlabel('Memory')
# axes[0, 2].set_xlim([None, 5.5])
axes[0, 2].grid(c='0.8')
axes[0, 2].legend()

# axes[1, 0].plot(our_llama_31_8b_inst_1048576_mem, our_llama_31_8b_inst_1048576_longbench, 'o-', label='Our', ms=marker_size, c=colors[0])
# axes[1, 0].scatter(42350419968, 53.0625, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
# axes[1, 0].scatter(25170550784, 51.61375, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
# axes[1, 0].scatter(41478004736, 52.24125, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
# axes[1, 0].scatter(24298135552, 50.07625, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
# axes[1, 0].scatter(41265405952, 39.7775, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# axes[1, 0].scatter(24085536768, 35.9175, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
# axes[1, 0].set_ylabel('Long Bench Avg.')

axes[1, 0].plot(our_llama_31_8b_inst_1048576_mem2, our_llama_31_8b_inst_1048576_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
axes[1, 0].scatter(42350419968, 12.26306725, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
axes[1, 0].scatter(25170550784, 34.28988266, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
axes[1, 0].scatter(41478004736, 14.90471745, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
axes[1, 0].scatter(24298135552, 74.07248688, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
axes[1, 0].scatter(41265405952, 28.66698265, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# axes[1, 0].scatter(24085536768, 395.6902771, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
axes[1, 0].set_ylabel('C4 PPL')

axes[1, 0].set_title('Llama 3.1 8B Inst')
axes[1, 0].set_xlabel('Memory')
# axes[1, 0].set_xlim([None, 5.5])
axes[1, 0].grid(c='0.8')
axes[1, 0].legend()

# axes[1, 1].plot(our_qwen25_14b_inst_1048576_mem, our_qwen25_14b_inst_1048576_longbench, 'o-', label='Our', ms=marker_size, c=colors[0])
# axes[1, 1].scatter(64903391232, 52.245, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
# axes[1, 1].scatter(39133587456, 50.535, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
# axes[1, 1].scatter(63251884032, 50.85375, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
# axes[1, 1].scatter(37482080256, 48.92125, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
# axes[1, 1].scatter(62847657984, 45.705, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# axes[1, 1].scatter(37077854208, 42.2175, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
# axes[1, 1].set_ylabel('Long Bench Avg.')

axes[1, 1].plot(our_qwen25_14b_inst_1048576_mem, our_qwen25_14b_inst_1048576_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
axes[1, 1].scatter(64903391232, 11.12086391, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
axes[1, 1].scatter(39133587456, 15.28642178, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
axes[1, 1].scatter(63251884032, 12.19672394, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
axes[1, 1].scatter(37482080256, 21.54403305, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
axes[1, 1].scatter(62847657984, 14.65903187, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
axes[1, 1].scatter(37077854208, 34.87562943, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
axes[1, 1].set_ylabel('C4 PPL')

axes[1, 1].set_title('Qwen2.5 14B Inst')
axes[1, 1].set_xlabel('Memory')
# axes[1, 1].set_xlim([None, 5.5])
axes[1, 1].grid(c='0.8')
axes[1, 1].legend()

# axes[1, 2].plot(our_mistral_7b_inst_v03_1048576_mem, our_mistral_7b_inst_v03_1048576_longbench, 'o-', label='Our', ms=marker_size, c=colors[0])
# axes[1, 2].scatter(40760778752, 52.19875, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
# axes[1, 2].scatter(23580909568, 50.515, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
# axes[1, 2].scatter(39888363520, 49.14625, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
# axes[1, 2].scatter(22708494336, 47.07125, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
# axes[1, 2].scatter(39675764736, 33.96125, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
# axes[1, 2].scatter(22495895552, 31.93125, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
# axes[1, 2].set_ylabel('Long Bench Avg.')

axes[1, 2].plot(our_mistral_7b_inst_v03_1048576_mem2, our_mistral_7b_inst_v03_1048576_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
axes[1, 2].scatter(40760778752, 9.066065788, label='w4g128 / kv4g128', s=scatter_size, c=colors[1])
axes[1, 2].scatter(23580909568, 11.58375931, label='w4g128 / kv2g128', s=scatter_size, c=colors[2])
axes[1, 2].scatter(39888363520, 9.78512001, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
axes[1, 2].scatter(22708494336, 14.18944263, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
axes[1, 2].scatter(39675764736, 16.95111656, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
axes[1, 2].scatter(22495895552, 44.03146362, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
axes[1, 2].set_ylabel('C4 PPL')

axes[1, 2].set_title('Mistral 7B Inst v0.3')
axes[1, 2].set_xlabel('Memory')
# axes[1, 2].set_xlim([None, 5.5])
axes[1, 2].grid(c='0.8')
axes[1, 2].legend()



plt.tight_layout()
plt.savefig(ppl_arch_figure, dpi=300)