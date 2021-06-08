import argparse
from pprint import pprint
from tqdm import tqdm
import csv

from rdkit import Chem

from retro_star.packages.mlp_retrosyn.mlp_retrosyn.mlp_inference import MLPModel


def canonicalize(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
    except:
        # print('no mol')
        return smiles
    if tmp is None:
        return smiles
    try:
        tmp = Chem.RemoveHs(tmp)
    except:
        print("ERROR, smiles : ", smiles)
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    return Chem.MolToSmiles(tmp)


def load_raw_reacts(name):
    reactions = []
    with open(name, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in tqdm(reader):
            reactions.append((row[1]))
    print('num %s:' % name, len(reactions))
    return reactions


def rxn_data_gen(phase):
    list_reactions = load_raw_reacts(phase)

    eval_cnt = 0
    for rxn in list_reactions:
        _, _, raw_prod = rxn.split('>')
        eval_cnt += 1
        yield rxn, raw_prod
    assert eval_cnt == len(list_reactions)


def eval_model(test_dataset_path, model, fname_pred, topk):
    case_gen = rxn_data_gen(test_dataset_path)

    cnt = 0
    topk_scores = [0.0] * topk

    pbar = tqdm(case_gen)

    fpred = open(fname_pred, 'w')
    for rxn, raw_prod in pbar:
        pred_struct = model.run(raw_prod, args.topk)
        # pred_struct = model.run(raw_prod, beam_size, topk, rxn_type=rxn_type)
        reactants, _, prod = rxn.split('>')
        if pred_struct is not None and len(pred_struct['reactants']):
            predictions = pred_struct['reactants']
        else:
            predictions = [prod]
        s = 0.0
        reactants = canonicalize(reactants)
        for i in range(topk):
            if i < len(predictions):
                pred = predictions[i]
                pred = canonicalize(pred)
                predictions[i] = pred
                cur_s = (pred == reactants)
            else:
                cur_s = s
            s = max(cur_s, s)
            topk_scores[i] += s
        cnt += 1
        if pred_struct is None or len(pred_struct['reactants']) == 0:
            predictions = []
        fpred.write('%s %d\n' % (rxn, len(predictions)))
        for i in range(len(predictions)):
            fpred.write('%s %s\n' % (pred_struct['template'][i], predictions[i]))
        msg = 'average score'
        for k in range(0, min(topk, 10), 3):
            msg += ', t%d: %.4f' % (k + 1, topk_scores[k] / cnt)
        pbar.set_description(msg)
    fpred.close()
    h = '========%s results========' % test_dataset_path
    print(h)
    for k in range(topk):
        print('top %d: %.4f' % (k + 1, topk_scores[k] / cnt))
    print('=' * len(h))

    f_summary = '.'.join(fname_pred.split('.')[:-1]) + '.summary'
    with open(f_summary, 'w') as f:
        f.write('type overall\n')
        for k in range(topk):
            f.write('top %d: %.4f\n' % (k + 1, topk_scores[k] / cnt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate one step retro-synthesis model")
    parser.add_argument('--template_rule_path', default='./retro_star/one_step_model/template_rules_1.dat',
                        type=str, help='Specify the path of all template rules.')
    parser.add_argument('--model_path', default='./retro_star/one_step_model/saved_rollout_state_1_2048.ckpt',
                        type=str, help='specify where the trained model is')
    # parser.add_argument('--test_dataset', default='USPTO-full',
    #                    type=str, choices=['USPTO-full'])
    parser.add_argument('--result_dir', default='./tmp.summary',
                        type=str)
    parser.add_argument('--topk', default=50, type=int)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--valid', action="store_true")

    args = parser.parse_args()
    state_path = args.model_path
    template_path = args.template_rule_path
    model = MLPModel(state_path, template_path, device=args.gpu)

    x = 'O=C1Nc2ccccc2C12COc1cc3c(cc12)OCCO3'
    y = model.run(x, args.topk)
    pprint(y)

    test_dataset_path = './retro_star/dataset/raw_rxn/raw_val.csv' if args.valid else \
        './retro_star/dataset/raw_rxn/raw_test.csv'
    eval_model(test_dataset_path, model, args.result_dir, args.topk)
    # eval_one_step_model(model, args.test_dataset, args.topk)

    print("done")
