from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

#Convert model outputs to a PDB file! Eventually put this into a src script! 
def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def output_pdb(prot_name, filepath, pdb):
    #output pdb to file
    with open(f"{filepath}{prot_name}.pdb", "w") as f:
        f.write(f"{prot_name}\n".join(pdb))


def fasta2dict(filepath):
    protein_dict = {}
    with open(filepath, "r") as a_file:
        for line in a_file:
            if line.startswith('>'):
                #Get rid of the > and \n
                split_line = line.split('>')
                split_line = split_line[1].split('\n')
                prot_name = split_line[0]
                seq = next(a_file)
                split_seq = seq.split('\n')
                protein_dict[prot_name] = split_seq[0]
    return protein_dict


   