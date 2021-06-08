import argparse
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="build prod2tpl file for training MLP")
    parser.add_argument("--input", required=True,
                        type=str)
    parser.add_argument("--mol2tpl_save_path", required=True,
                        type=str)
    parser.add_argument("--tpl_save_path", required=True,
                        type=str)
    parser.add_argument("--direction", default='backward',
                        type=str, choices=['forward', 'backward'])
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    rxn_list = input_df['rxn_smiles']
    tpl_list = input_df['retro_templates']

    mol2tpl_list = []

    loc = -1 if args.direction == 'backward' else 0
    for rxn, tpl in tqdm(zip(rxn_list, tpl_list)):
        mol = rxn.split(">")[loc]
        mol2tpl_list.append(tpl + '\t' + mol)

    with open(args.mol2tpl_save_path, "w") as f:
        f.write("\n".join(mol2tpl_list))

    with open(args.tpl_save_path, "w") as f:
        f.write("\n".join(list(set(list(tpl_list)))))
