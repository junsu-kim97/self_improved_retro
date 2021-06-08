import rdkit.Chem as Chem
import pickle

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None

    if mol is None:
        return ''
    else:
        return Chem.MolToSmiles(mol)


def canonicalize(smiles=None, smiles_list=None):
    """Return the canonicalized version of the given smiles or smiles list"""
    assert (smiles is None) != (smiles_list is None)  # Only take one input

    if smiles is not None:
        return canonicalize_smiles(smiles)
    elif smiles_list is not None:
        # Convert smiles to mol and back to cannonicalize
        new_smiles_list = []

        for smiles in smiles_list:
            new_smiles_list.append(canonicalize_smiles(smiles))
        return new_smiles_list


def read_txt(file_path, detokenize=False):
    out_list = []
    with open(file_path, "r") as f:
        while True:
            line = f.readline().rstrip()
            if not line:
                break
            if detokenize:
                line = "".join(line.split(" "))
            out_list.append(line)

    return out_list


def read_file(file_path, beam_size=1, max_read=-1, parse_func=None):
    read_file = open(file_path, 'r+')
    output_list = []  # List of beams if beam_size is > 1 else list of smiles
    cur_beam = []  # Keep track of the current beam

    for line in read_file.readlines():
        if parse_func is None:
            parse = line.strip().replace(' ', '')  # default parse function
            if ',' in parse:
                # If output separated by commas, return first by default
                parse = parse.split(',')[0]
        else:
            parse = parse_func(line)

        cur_beam.append(parse)
        if len(cur_beam) == beam_size:
            if beam_size == 1:
                output_list.append(cur_beam[0])
            else:
                output_list.append(cur_beam)
            if max_read != -1 and len(output_list) >= max_read:
                break
            cur_beam = []
    read_file.close()
    return output_list


def remove_atom_mapping(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    smiles = Chem.MolToSmiles(mol)
    return smiles


def txt2pkl(txt_path, pkl_path):
    input_list = read_txt(txt_path, detokenize=True)
    input_list = [[line] for line in input_list]
    with open(pkl_path, 'wb') as f:
        pickle.dump(input_list, f)


if __name__ == '__main__':
    txt2pkl(txt_path='/data/junsu_data/ssl-rxn/retro_smiles_transformer/dataset/schneider50k/backward/src-train.txt',
            pkl_path='/home/junsu/workspace/retro_star/biased_one_step/data/cooked_schneider50k/src-train.pkl')

