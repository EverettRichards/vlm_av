from datasets import load_dataset

# download subset once
ds = load_dataset("dgural/bdd100k", split="validation[:500]")

# save locally
ds.save_to_disk("data/bdd100k_val_500")
