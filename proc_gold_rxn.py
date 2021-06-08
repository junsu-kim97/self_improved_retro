import argparse
import pickle
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool

from mlp_retrosyn.mlp_inference import MLPModel


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_gold_rxns(args, plan_dict, cost_thr):
    device = 0 if args.gpu > -1 else -1

    if args.single_iter:
        pre_trained = None
    else:
        pre_trained = MLPModel(args.pre_trained_backward_model, args.mlp_templates, device=device)

    cnt_succ_routes = 0
    cnt_fail_routes = 0
    cnt_cutoff = 0
    cnt_forward_cutoff = 0
    gold_rxns = []
    for succ, routes, time in tqdm(zip(plan_dict['succ'], plan_dict['routes'], plan_dict['cumulated_time'])):
        # if args.debug and time == 0:
        #    continue
        # if args.forward_only and time != 0:
        #     continue

        if succ:
            cnt_succ_routes += 1
            for par_idx, ch_idxes in enumerate(routes.children):
                if ch_idxes is not None:
                    ###  Calculate confidence score ###
                    if args.single_iter:
                        cost = routes.costs[par_idx]
                    else:
                        pre_trained_result = pre_trained.run(routes.mols[par_idx], topk=args.pre_trained_topk)
                        if pre_trained_result is None or routes.templates[par_idx] not in pre_trained_result['template']:
                            cnt_cutoff += 1
                            continue
                        match_idx = pre_trained_result['template'].index(routes.templates[par_idx])
                        score = pre_trained_result['scores'][match_idx]
                        cost = 0.0 - np.log(np.clip(np.array(score), 1e-3, 1.0))
                    ####################################

                    # Filter out reactions under confidence score
                    if args.cut_off and cost_thr < cost:
                        cnt_cutoff += 1
                        continue

                    # Filter augmentation by forward one-step model
                    if time == 0:
                        backward_result = pre_trained.run(routes.mols[par_idx], topk=args.pre_trained_topk)
                        if backward_result is None or routes.templates[par_idx] not in backward_result['template']:
                            cnt_forward_cutoff += 1
                            continue
                        match_idx = backward_result['template'].index(routes.templates[par_idx])
                        score = backward_result['scores'][match_idx]
                        cost = 0.0 - np.log(np.clip(np.array(score), 1e-3, 1.0))

                        if args.cut_off and cost_thr < cost:
                            cnt_forward_cutoff += 1
                            continue

                    par_smi = routes.mols[par_idx]
                    tpl = routes.templates[par_idx]
                    ch_smi_list = [routes.mols[ch_idx] for ch_idx in ch_idxes]
                    ch_smi = ".".join(ch_smi_list)
                    rxn = ch_smi + ">>" + par_smi
                    gold_rxns.append((rxn, tpl))

        else:
            cnt_fail_routes += 1

    return gold_rxns


def save_gold_rxns(gold_rxns_list, save_path):
    df = pd.DataFrame(gold_rxns_list, columns=["rxn_smiles", "retro_templates"])

    df.to_csv(save_path, index=False)
    print("save done @ ", save_path)


def save_gold_rxns_template(gold_rxns_list, save_path):
    out_list = []
    for rxn, tpl in gold_rxns_list:
        prod = rxn.split(">>")[-1]
        out_list.append(tpl + '\t' + prod)
    with open(save_path, "w") as f:
        f.write("\n".join(out_list))


def aug_gold_rxn(args, gold_rxn_list, aug_thr):
    device = 0 if args.gpu > -1 else -1

    forward_model = MLPModel(args.forward_model, args.mlp_templates, device=device)

    if args.fw_backward_validate:
        backward_model = MLPModel(args.fw_backward_model, args.mlp_templates, device=device)

    aug_rxn_list = []
    cnt_none = 0
    for gold_rxn in tqdm(gold_rxn_list):
        rxn, tpl = gold_rxn
        react, _, prod = rxn.split(">")
        forward_result = forward_model.run(react, topk=args.aug_topk, backward=False)

        if forward_result is None:
            cnt_none += 1
            continue
        else:
            predictions_list = forward_result['reactants']  # Actually, this it predicted product
            scores_list = forward_result['scores']
            template_list = forward_result['template']

            costs_list = [0.0 - np.log(np.clip(np.array(score), 1e-3, 1.0)) for score in scores_list]

            for pred, cost, pred_tpl in zip(predictions_list, costs_list, template_list):
                if cost > aug_thr:
                    break
                if args.fw_backward_validate:
                    backward_result = backward_model.run(pred, topk=1)

                    if backward_result is None or pred_tpl not in backward_result['template']:
                        continue

                aug_rxn_list.append((react + '>>' + pred, pred_tpl))

    gold_rxn_list.extend(aug_rxn_list)

    return gold_rxn_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=""" build gold training reactions from successful trajectories from 
                                                     multi-step retro-synthesis planning """)
    parser.add_argument("--plan", default="./retro_star/results/retro_star_zero/x1/multi-step/succ_traj/shard_0/plan.pkl",
                        type=str)

    # Cut off options
    parser.add_argument("--cut_off", action="store_true")
    parser.add_argument("--threshold", "-thr", default=0.8,
                        type=float, help="""  """)

    parser.add_argument("--mlp_templates", default="""./retro_star/one_step_model/template_rules_1.dat""")
    # parser.add_argument("--valid_topk", default=5, type=int)

    parser.add_argument("--pre_trained_backward_model",
                        default="./retro_star/one_step_model/saved_rollout_state_1_2048.ckpt")
    parser.add_argument("--single_iter", action='store_true')
    parser.add_argument("--pre_trained_topk", default=5, type=int)
    parser.add_argument("--gpu", default=-1, type=int)

    # Augment using forward model single-step reaction-wise
    parser.add_argument("--aug_forward", action="store_true")
    parser.add_argument("--forward_model", default="./retro_star/one_step_model/forward/forward_model.ckpt")
    parser.add_argument("--aug_topk", default=5, type=int)
    parser.add_argument("--aug_thr", default=0.8, type=float)
    parser.add_argument("--fw_backward_validate", action="store_true")
    parser.add_argument("--fw_backward_model",
                           default="./retro_star/one_step_model/saved_rollout_state_1_2048.ckpt")
    parser.add_argument('--num_proc', default=1, type=int)

    # Save paths
    parser.add_argument("--save_path", default='./retro_star/results/retro_star_zero/x1/multi-step/succ_traj/shard_0/gold.csv',
                        type=str)
    parser.add_argument("--tpl2prod_save_path", default='./retro_star/results/retro_star_zero/x1/multi-step/succ_traj/shard_0/templates.dat')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    plan = None
    with open(args.plan, "rb") as f:
        plan = pickle.load(f)

    cost_thr = 0.0 - np.log(np.clip(np.array(args.threshold), 1e-3, 1.0))
    gold_rxns_list = get_gold_rxns(args, plan, cost_thr)
    if args.aug_forward:
        num_process = args.num_proc
        aug_thr = 0.0 - np.log(np.clip(np.array(args.aug_thr), 1e-3, 1.0))
        if num_process > 1:
            aug_gold_rxns_list = []
            with Pool(processes=num_process) as pool:
                results = pool.starmap(aug_gold_rxn,
                                       [(args, sub_list, aug_thr)
                                        for sub_list in chunks(gold_rxns_list, int(len(gold_rxns_list)/num_process))])
                for result in results:
                    aug_gold_rxns_list.extend(result)
            gold_rxns_list = aug_gold_rxns_list
        else:
            gold_rxns_list = aug_gold_rxn(args, gold_rxns_list, aug_thr)

    save_gold_rxns(gold_rxns_list, args.save_path)
    save_gold_rxns_template(gold_rxns_list, args.tpl2prod_save_path)

    print("save ", len(gold_rxns_list))
