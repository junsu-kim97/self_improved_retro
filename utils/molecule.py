from rdkit import Chem

def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RemoveHs(mol)
        for atom in mol.GetAtoms():
            atom.ClearProp("molAtomMapNumber")
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except:
        return ""