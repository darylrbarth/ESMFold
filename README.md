# ESMFold

Small script that uses the transformers library from hugging face and enables you to fold large batches of proteins with ESMFold!

First, you want to have the right environment setup for ESMFold. Do that by following the instructions on the Transformer Page here: https://huggingface.co/docs/transformers/en/installation

Once you have the virtual environment setup, activate it using something like: 

```
source .env/bin/activate
```

Edit the ESMFold.py script to include the path to the folder of fasta files you want to fold as well as the output path which will hold all of the pdbs. 

Then run and watch your pdb files appear!

```
python -m ESMFold.py
```

And that should be it <3, happy Folding!

HUGE thanks to the authors of this notebook: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb#scrollTo=9beab944
As well as the creators and maintainers of the Hugging Face Transformers module AND all of the awesome people behind ESM and OpenFold! Y'all rock!!