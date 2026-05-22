import csv
import json
import numpy as np
import matplotlib.pyplot as plt
# from utils import get_net_info

# ppl_arch_figure = '/NAS/SJ/actquant/search/visualize/fig/joint_results.png'
ppl_arch_figure = '/NAS/SJ/actquant/search/visualize/fig/jsd_appr_results.png'

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

our_comb_jsd_llama_31_8b_inst_1024_mem = [5867839488, 5670412288, 5654159360, 5011349504, 4994572288, 4798554112, 4781744128]
our_comb_jsd_llama_31_8b_inst_1024_longbench = [53.04625, 53.9, 53.24875, 53.3925, 53.005, 51.4325, 48.94875]

our_comb_jsd_llama_31_8b_inst_131072_mem = [10202652672, 9543098368, 9330237440, 8669110272, 7390896128, 7180656640]
our_comb_jsd_llama_31_8b_inst_131072_longbench = [53.415, 52.9225, 52.735, 53.5675, 49.7975, 44.7375]

our_comb_jsd_llama_31_8b_inst_1048576_mem = [42171637760, 41496879104, 40624463872, 25193095168, 24982331392, 24319107072, 24108867584]
our_comb_jsd_llama_31_8b_inst_1048576_longbench = [53.5125, 52.99625, 52.96, 51.605, 51.925, 50.16375, 47.6175]

our_llama_31_8b_inst_1024_mem = [5666250752, 5649473536, 5006434304, 4989657088, 4793835520, 4777058304]
our_llama_31_8b_inst_1024_longbench = [53.8925, 52.0425, 53.4475, 51.73375, 51.35625, 47.9825]
out_llama_31_8b_inst_1024_gsm8k = [0.698256255, 0.547384382, 0.526914329, 0.33055345, 0.351781653, 0.182714177]

our_llama_31_8b_inst_131072_mem = [10194001920, 9534185472, 9321586688, 8046518272, 7386701824, 7174103040]
our_llama_31_8b_inst_131072_longbench = [53.8925, 53.4475, 51.35625, 52.0425, 51.73375, 47.9825]

# our_llama_31_8b_inst_1048576_mem = [42137821184, 41478004736, 41265405952, 24957952002, 04298135552, 24085536768]
# our_llama_31_8b_inst_1048576_longbench = [53.8925, 53.4475, 51.35625, 52.0425, 51.73375, 47.9825]
# our_llama_31_8b_inst_1048576_gsm8k = [0.698256255, 0.526914329, 0.351781653, 0.547384382, 0.33055345, 0.182714177]
our_llama_31_8b_inst_1048576_mem = [42137821184, 41478004736, 41265405952, 24957952000, 24298135552, 24085536768]
our_llama_31_8b_inst_1048576_longbench = [53.8925, 53.4475, 51.35625, 52.0425, 51.73375, 47.9825]

# our_qwen25_7b_inst_1024_mem = [5653232640, 5653232640, 4857920512, 4850613248, 4659690496, 4651793408]
# our_qwen25_7b_inst_1024_longbench = [51.63875, 51.63875, 46.5425, 46.82625, 41.9175, 41.55]

# our_qwen25_7b_inst_1048576_mem = [21428902912, 20811594752, 20613241856, 13912710144, 13295401984, 13097049088]
# our_qwen25_7b_ins_1048576_longbench = [49.88, 49.12, 42.19375, 47.8075, 46.85, 39.85]
# our_qwen25_7b_inst_1048576_gsm8k = [0.714935557, 0.635329795, 0.529946929, 0.721758908, 0.561789234, 0.387414708]

our_qwen25_14b_inst_1024_mem = [10199181312, 10172606464, 8546723840, 8517232640, 8144824320, 8105469952]
our_qwen25_14b_inst_1024_longbench = [52.25875, 52.3725, 50.87, 50.62625, 50.1125, 49.14375]
our_qwen25_14b_inst_1024_c4 = [11.07064342, 11.07767773, 12.21304131, 22.27282238, 13.23994827, 13.2931118]

# our_qwen25_14b_inst_1048576_mem = [64761964544, 39148201984, 63219116032, 37516814336, 62749353984, 37110491136]
# our_qwen25_14b_inst_1048576_longbench = [51.48125, 50.305, 52.22625, 46.14625, 52.16125, 42.8575]
our_qwen25_14b_inst_1048576_mem = [64761964544, 63219116032, 62749353984, 39148201984, 37516814336, 37110491136]
our_qwen25_14b_inst_1048576_longbench = [51.48125, 52.22625, 52.16125, 50.305, 46.14625, 42.8575]
our_qwen25_14b_inst_1048576_c4 = [11.12086391, 21.16060066, 11.17476559, 15.09971619, 25.46052551, 32.95238876]

our_mistral_7b_inst_v03_1024_mem = [4286586880, 4271775744, 3417088000, 3394641920, 3205013504, 3189121024]
our_mistral_7b_inst_v03_1024_longbench = [52.14375, 52.50125, 50.02875, 49.945, 49.25625, 48.51125]
our_mistral_7b_inst_v03_1024_c4 = [9.057579041, 9.069731712, 9.813754082, 9.852708817, 10.61759567, 10.6838007]

# our_mistral_7b_inst_v03_1048576_mem = [40691572736, 23583006720, 39919820800, 39651385344, 22516604928]
# our_mistral_7b_inst_v03_1048576_longbench = [52.52, 50.25, 52.4, 52.45875, 46.94125]
# our_mistral_7b_inst_v03_1048576_mem = [40691572736, 39919820800, 39651385344, 23583006722, 02516604928]
# our_mistral_7b_inst_v03_1048576_mem2 = [40691572736, 39919820800, 39651385344, 23583006722, 02728417282, 02516604928]
# our_mistral_7b_inst_v03_1048576_longbench = [52.52, 52.4, 52.45875, 50.25, 46.94125]
# our_mistral_7b_inst_v03_1048576_c4 = [9.082056999, 9.093878746, 9.101392746, 11.4829216, 15.13122749, 18.23580742]
our_mistral_7b_inst_v03_1048576_mem = [40548179968, 39888363520, 39675764736, 23368310784, 22708494336, 22495895552]
our_mistral_7b_inst_v03_1048576_longbench = [51.5375, 49.8325, 48.90375, 50.31125, 48.50875, 47.53375]

axes[0, 0].plot(our_comb_jsd_llama_31_8b_inst_1024_mem, our_comb_jsd_llama_31_8b_inst_1024_longbench, 'o-', label='Our JSD Appr.', ms=marker_size, c=colors[-1])
axes[0, 0].plot(our_llama_31_8b_inst_1024_mem, our_llama_31_8b_inst_1024_longbench, 'o-', label='Our Mixed-Precision', ms=marker_size, c=colors[0])
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

axes[0, 1].plot(our_comb_jsd_llama_31_8b_inst_131072_mem, our_comb_jsd_llama_31_8b_inst_131072_longbench, 'o-', label='Our JSD Appr.', ms=marker_size, c=colors[-1])
axes[0, 1].plot(our_llama_31_8b_inst_131072_mem, our_llama_31_8b_inst_131072_longbench, 'o-', label='Our Mixed-Precision', ms=marker_size, c=colors[0])
axes[0, 1].scatter(10194001920, 50.39125, label='w4g-1 / kv4g128', s=scatter_size, c=colors[1])
axes[0, 1].scatter(8046518272, 49.09, label='w4g-1 / kv2g128', s=scatter_size, c=colors[2])
axes[0, 1].scatter(9534185472, 52.24125, label='w3g128 / kv4g128', s=scatter_size, c=colors[3])
axes[0, 1].scatter(7386701824, 50.07625, label='w3g128 / kv2g128', s=scatter_size, c=colors[4])
axes[0, 1].scatter(9321586688, 39.7775, label='w3g-1 / kv4g128', s=scatter_size, c=colors[5])
axes[0, 1].scatter(7174103040, 35.9175, label='w3g-1 / kv2g128', s=scatter_size, c=colors[6])
axes[0, 1].set_ylabel('Long Bench Avg.')


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

axes[0, 2].plot(our_comb_jsd_llama_31_8b_inst_1048576_mem, our_comb_jsd_llama_31_8b_inst_1048576_longbench, 'o-', label='Our JSD Appr.', ms=marker_size, c=colors[-1])
axes[0, 2].plot(our_llama_31_8b_inst_1048576_mem, our_llama_31_8b_inst_1048576_longbench, 'o-', label='Our JSD Appr.', ms=marker_size, c=colors[0])
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

axes[1, 0].plot(our_qwen25_14b_inst_1024_mem, our_qwen25_14b_inst_1024_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
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

axes[2, 0].plot(our_mistral_7b_inst_v03_1024_mem, our_mistral_7b_inst_v03_1024_c4, 'o-', label='Our', ms=marker_size, c=colors[0])
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

axes[1, 2].plot(our_qwen25_14b_inst_1048576_mem, our_qwen25_14b_inst_1048576_longbench, 'o-', label='Our', ms=marker_size, c=colors[0])
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

axes[2, 2].plot(our_mistral_7b_inst_v03_1048576_mem, our_mistral_7b_inst_v03_1048576_longbench, 'o-', label='Our', ms=marker_size, c=colors[0])
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