import argparse
import pickle
from retro_star.packages.mlp_retrosyn.mlp_retrosyn.mlp_inference import MLPModel
from retro_star.packages.rdchiral.rdchiral.main import rdchiralRunText
import numpy as np
from tqdm import tqdm
from rdkit import Chem

def cano_smiles(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
        if tmp is None:
            return None, smiles
        tmp = Chem.RemoveHs(tmp)
        if tmp is None:
            return None, smiles
        [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
        return tmp, Chem.MolToSmiles(tmp)
    except:
        return None, smiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=str,
                        default='', required=True)
    parser.add_argument('--iter_step', type=int,
                        default=50)
    parser.add_argument('--ref_back_model', type=str,
                        default='./one_step_model/saved_rollout_state_1_2048.ckpt')
    parser.add_argument('--template_rule_path', default='./one_step_model/template_rules_1.dat',
                        type=str, help='Specify the path of all template rules.')
    parser.add_argument("--target_mols", default='./dataset/routes_possible_test_hard.pkl',
                        type=str, help="target molecule dataset")
    parser.add_argument("--topk", default=50, type=int)
    parser.add_argument('--gpu', default=-1, type=int)

    args = parser.parse_args()

    with open(args.target_mols, "rb") as f:
        gt_list = pickle.load(f)

    len_list = [len(gt) for gt in gt_list]
    max_len = max(len_list)

    ref_back_model = MLPModel(args.ref_back_model, args.template_rule_path, device=args.gpu)

    route_cost_list = []
    topk = 50
    for gt in tqdm(gt_list):
        route_cost = 0
        for rxn in gt:
            tpl_ans = None
            prod, _, react = rxn.split(">")
            cano_prod = cano_smiles(prod)[1]
            cano_react = cano_smiles(react)[1]
            result = ref_back_model.run(cano_prod, topk)

            if 'templates' in result.keys():
                templates = result['templates']
            else:
                templates = result['template']

            found_ans = False
            for tpl in templates:
                out = rdchiralRunText(tpl, cano_prod)
                cano_out = cano_smiles(out)[1]
                for cano_out_elem in cano_out:
                    if cano_out_elem == cano_react:
                        tpl_ans = tpl
                        break

                if tpl_ans is not None:
                    break

            if tpl_ans is not None:
                match_idx = templates.index(tpl_ans)
                score = result['scores'][match_idx]
            else:
                score = 0

            cost = 0.0 - np.log(np.clip(np.array(score), 1e-3, 1.0))
            route_cost += cost

        # print(route_cost)
        route_cost_list.append(route_cost)

    max_len = max(len_list)
    max_cost = max(route_cost_list)

    with open(args.plan, "rb") as f:
        plan_dict = pickle.load(f)

    max_iter = max(plan_dict['iter'])
    total = len(plan_dict['succ'])

    iter_step = args.iter_step
    succ_cnt_list = [0] * (int(max_iter/iter_step))
    cumulated_len = 0
    cumulated_time = 0
    route_costs_list = []
    route_len_list = []
    # rxn_costs_list = []

    for succ, iter, route_len, route in tqdm(zip(plan_dict['succ'], plan_dict['iter'], plan_dict['route_lens'], plan_dict['routes'])):
        # Length & Time
        if succ is False:
            cumulated_len += 2 * max_len
            cumulated_time += max_iter
            route_costs_list.append(2 * max_cost)
            # rxn_costs_list.append(None)
            continue

        else:
            route_len_list.append(route_len)
            cumulated_len += route_len
            cumulated_time += iter

        # Success rate
        for i in range(len(succ_cnt_list)):
            if iter <= (i+1) * iter_step:
                succ_cnt_list[i] += 1

        # Route cost calculation for success routes
        route_cost = 0
        rxn_costs = []
        for par_idx, ch_idxes in enumerate(route.children):
            if ch_idxes is not None:
                par_smi = route.mols[par_idx]
                result = ref_back_model.run(par_smi, args.topk)
                if 'templates' in result.keys():
                    templates = result['templates']
                else:
                    templates = result['template']

                if route.templates[par_idx] not in templates:
                    score = 0
                else:
                    match_idx = templates.index(route.templates[par_idx])
                    score = result['scores'][match_idx]

                cost = 0.0 - np.log(np.clip(np.array(score), 1e-3, 1.0))
                route_cost += cost
                rxn_costs.append(cost)
        route_costs_list.append(route_cost)
        # rxn_costs_list.append(rxn_costs)

    for i in range(len(succ_cnt_list)):
        print("%d" %(succ_cnt_list[i]))
    for i in range(len(succ_cnt_list)):
        success_ratio = float(succ_cnt_list[i])/float(total)
        print("%f" %(success_ratio))


    print("Avg. length: ", float(cumulated_len)/float(total))
    print("Avg. time: ", float(cumulated_time)/float(total))
    print("Avg. cost ", sum(route_costs_list)/len(route_costs_list))
    print("done")
