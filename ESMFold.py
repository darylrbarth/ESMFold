import src 
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from src.make_pdb import convert_outputs_to_pdb, output_pdb, fasta2dict
#from src.run_esmfold import run_esm_fold
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from tqdm import tqdm


tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

model = model.cuda()
model.esm = model.esm.half()
#Model optimizations
#Enable TensorFloat32 computation for a general speedup if your hardware supports it. This line has no effect if your hardware doesn't support it.
import torch

torch.backends.cuda.matmul.allow_tf32 = True

#Finally, we can reduce the 'chunk_size' used in the folding trunk. Smaller chunk sizes use less memory, but have slightly worse performance.
# Uncomment this line if your GPU memory is 16GB or less, or if you're folding longer (over 600 or so) sequences
model.trunk.set_chunk_size(64)

def run_esm_fold(aa):
    tokenized_input = tokenizer([aa], return_tensors="pt", add_special_tokens=False)['input_ids']
    tokenized_input = tokenized_input.cuda()
    with torch.no_grad():
         output = model(tokenized_input)
    pdb = convert_outputs_to_pdb(output)
    return pdb


#Put fasta file name here:
fasta_fp = '/stor/work/Marcotte/project/drbarth/plastics/data/results/PETHits_Guaymas2020_ALLBINS.fasta'
output_fp = '/stor/work/Marcotte/project/drbarth/plastics/data/results/PETHits_Guaymas2020/ESMfold_output/'

#Read in fasta file to a dictionary
protein_dict = fasta2dict(fasta_fp)

#Run esm fold on each protein sequence
for prot_name, aa in tqdm(protein_dict.items()):
    pdb = run_esm_fold(aa)
    print(f"Finished folding {prot_name}")
    output_pdb(prot_name, output_fp, pdb)

print('Done!')


#Clean up memory!
del model 
torch.cuda.empty_cache()





#outputs = []
#
#with torch.no_grad():
#    for input_ids in tqdm(phi_tokenized):
#        input_ids = torch.tensor(input_ids, device='cuda').unsqueeze(0)
#        output = model(input_ids)
#        outputs.append({key: val.cpu() for key, val in output.items()})
#
        
#Run in bulk
#Loop over tokenized data, passing each sequence into our model 












   
   
   
   
P1 = "MEKPVVFENEGQQIVGMLHAPDGAEGLSPAVVMFHGFTGTKVEPHRLFVKTARRLAEEGFYVLRFDFRGSGDSEGEFREMTLEGEISDAKASLDFILSQPGVDRGRIGVIGLSMGGGVAACLAGRDERVRAVALWAAVSEDPPDLFQELIKTFEERPDKSVDYVDMGGNLVGKGFFEDLRNVKPLQEISGFEGPVLIVHGDNDQTVSVEHAYRFYERLKGKHPLTALHIIRGADHTFNSHEWEREVIEVTVDFMKRA"
P1_141_183_del = "MEKPVVFENEGQQIVGMLHAPDGAEGLSPAVVMFHGFTGTKVEPHRLFVKTARRLAEEGFYVLRFDFRGSGDSEGEFREMTLEGEISDAKASLDFILSQPGVDRGRIGVIGLSMGGGVAACLAGRDERVRAVALWAAVSEPLQEISGFEGPVLIVHGDNDQTVSVEHAYRFYERLKGKHPLTALHIIRGADHTFNSHEWEREVIEVTVDFMKRA"
P1_140_182_del = "MEKPVVFENEGQQIVGMLHAPDGAEGLSPAVVMFHGFTGTKVEPHRLFVKTARRLAEEGFYVLRFDFRGSGDSEGEFREMTLEGEISDAKASLDFILSQPGVDRGRIGVIGLSMGGGVAACLAGRDERVRAVALWAAVSKPLQEISGFEGPVLIVHGDNDQTVSVEHAYRFYERLKGKHPLTALHIIRGADHTFNSHEWEREVIEVTVDFMKRA"
P1_139_183_del = "MEKPVVFENEGQQIVGMLHAPDGAEGLSPAVVMFHGFTGTKVEPHRLFVKTARRLAEEGFYVLRFDFRGSGDSEGEFREMTLEGEISDAKASLDFILSQPGVDRGRIGVIGLSMGGGVAACLAGRDERVRAVALWAAVPLQEISGFEGPVLIVHGDNDQTVSVEHAYRFYERLKGKHPLTALHIIRGADHTFNSHEWEREVIEVTVDFMKRA"
tokenized_input = tokenizer([P1_139_183_del], return_tensors="pt", add_special_tokens=False)['input_ids']


#On GPU, move tokenized data to GPU
tokenized_input = tokenized_input.cuda()

#Now to get model outputs, its as simple as: 

import torch

with torch.no_grad():
    output = model(tokenized_input)





outputs = []

with torch.no_grad():
    for input_ids in tqdm(input_list):
        input_ids = torch.tensor(input_ids, device='cuda').unsqueeze(0)
        output = model(input_ids)
        outputs.append({key: val.cpu() for key, val in output.items()})

#Convert outputs to PDB files
pdb_list = [convert_outputs_to_pdb(output) for output in outputs]

#Save all of the pdbs to a disc together
protein_identifiers = df.Entry.tolist()
for identifier, pdb in zip(protein_identifiers, pdb_list):
    with open(f"{identifier}.pdb", "w") as f:
        f.write("".join(pdb))
