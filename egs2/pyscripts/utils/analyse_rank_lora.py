import torch
import os

from espnet2.torch_utils.model_summary import (
    get_human_readable_count,
    get_trainable_params,
)


lang=93

model_dirs = {
        #'svdlora-s1em3-minr1': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em3_minr1_raw_cgnnl_bpe50000',
        #'svdlora-s1em3-minr2': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em3_minr2_raw_cgnnl_bpe50000',
        #'svdlora-s1em3-minr4': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em3_minr4_raw_cgnnl_bpe50000',
        #'svdlora-s1em2-minr1': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em2_minr1_raw_cgnnl_bpe50000', 
        'svdlora-s1em2-minr2': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em2_minr2_raw_cgnnl_bpe50000',
        'svdlora-s1em2-minr4': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em2_minr4_raw_cgnnl_bpe50000',
        'svdlora-s1em2-minr4 (R-VL)': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em2_minr4_rao_reg2f_verylow_raw_cgnvl_from_cgnnl_bpe50000',
        'svdlora-s1em2-minr4 (R-L)': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em2_minr4_rao_reg2f_low_raw_cgnvl_from_cgnnl_bpe50000',
        'svdlora-s1em2-minr4 (R-M)': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em2_minr4_rao_reg2f_med_raw_cgnvl_from_cgnnl_bpe50000',
        'svdlora-s1em2-minr4 (R-H)': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em2_minr4_rao_reg2f_high_raw_cgnvl_from_cgnnl_bpe50000',
        'svdlora-s1em2-minr4 (R-VH)': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em2_minr4_rao_reg2f_veryhigh_raw_cgnvl_from_cgnnl_bpe50000',
        'svdlora-s1em3-minr4': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em3_minr4_raw_cgnnl_bpe50000',
        #'svdlora-s1em3-minr4(2)': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlorab_svd090_s1em3_minr4_raw_cgnvl_from_cgnnl_bpe50000',
        #'svdlora-s1em3-minr4-R': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlorab_svd090_s1em3_minr4_rao_raw_cgnnl_bpe50000',
        #'finetuning-weight matrices': 'exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd100_s0_minr0_raw_cgnnl_bpe50000',
        'increlora-i4-a1': 'exp/ml/s2t_train_asr_owsmv31_base_clincrelora2_r8_highlr_i4a1_raw_cgnnl_bpe50000', 
        #'infincrelora-i4-a1': 'exp/ml/s2t_train_asr_owsmv31_base_clinfincrelora2_r8_highlr_i4a1_raw_cgnnl_bpe50000',
        #'increlora-i4-a1(2)': 'exp/ml/s2t_train_asr_owsmv31_base_clincrelora2_r8_highlr_i4a1_raw_cgnvl_from_cgnnl_bpe50000',
        #'increlora-i4-a1(2P)': 'exp/ml/s2t_train_asr_owsmv31_base_clincrelora2_r8_highlr_i4a1_pruning_raw_cgnvl_from_cgnnl_bpe50000',
        #'increlora[1e0]-i4-a1': 'exp/ml/s2t_train_asr_owsmv31_base_clincrelora2_r8_highlr_i4a1_ort1e0_raw_cgnnl_bpe50000',
        #'increlora-i4-a1-scale': 'exp/ml/s2t_train_asr_owsmv31_base_clincrelora2_r8_highlr_i4a1_scale_raw_cgnnl_bpe50000',
        #'increlora-i4-a1-pruning': 'exp/ml/s2t_train_asr_owsmv31_base_clincrelora2_r8_highlr_i4a1_scale_pruning_raw_cgnnl_bpe50000',
}

def process_task_mapping(file_name):
    result_dict = {}
    tm = torch.load(file_name)
    for layer, tm_dict in tm.items():
        assert any([k[0]==lang for k in tm_dict.keys()])
        tasks = [k[1] for k in tm_dict.keys() if k[0] == lang]
        result_dict[layer] = 0
        for task in tasks:
            result_dict[layer] += len(tm_dict[(lang,task)])
    return result_dict

def process_rank_pattern(file_name):
    result_dict = {}
    tm = torch.load(file_name)
    for layer, tm_dict in tm.items():
        assert any([k[0]==lang for k in tm_dict.keys()])
        tasks = [k[1] for k in tm_dict.keys() if k[0] == lang]
        if len(tasks) > 1:
            print("tasks", tasks)
        result_dict[layer] = 0
        for task in tasks:
            result_dict[layer] += tm_dict[(lang,task)]
    return result_dict

def print_stats(method, file_name, rank_pattern):
    ts_params = 0
    rank = 0
    state_dict = torch.load(file_name, map_location='cpu')
    for name, param in state_dict.items():
        if 'ranknum' in name:
            layer_name = ".".join(name.split(".")[:-1])
            n1, n2 = state_dict[f"{layer_name}.weight"].size()
            rank += param.item()
            #assert param.item() == rank_pattern[layer_name], f"{layer_name}: Rank from state dict {param.item()} and rank pattern {rank_pattern[layer_name]} are not equal."
            ts_params += n1 * param.item() + param.item() * n2 + param.item()
    print(f"Method {method.upper()} adds {int(rank)} rank and {get_human_readable_count(ts_params)} task specific parameters..")

final_dict = {}

for model, model_dir in model_dirs.items():
    print(f"Processing {model} - {model_dir}..")
    if os.path.exists(f"{model_dir}/task_mapping.pth"):
        rank_pattern = process_task_mapping(f"{model_dir}/task_mapping.pth")
        model_name = "latest.pth"
    else: 
        assert os.path.exists(f"{model_dir}/B_rank_pattern.pth"), f"File {model_dir}/rank_pattern.pth does not exist."
        rank_pattern = process_rank_pattern(f"{model_dir}/B_rank_pattern.pth")
        model_name = "B_pruned_model.pth"
    final_dict[model] = rank_pattern
    print_stats(model, f"{model_dir}/{model_name}", rank_pattern)

torch.save(final_dict, 'lora_rank_comparison.pth')
