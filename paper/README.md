# How to train and evaluate assessors

- `train_embeddings_assessor.ipynb` is a notebook that trains the baseline assessors based on embeddings and stores the full set of results in the `results` folder (in `.pkl` files that include the full weights of each assessor)
- `compute_assessors_score.py` is a Python script that post-processes the full assessor results, compute metrics and saves the results in `.csv` files in the `results` folder
- `generate_paper_figures.ipynb` starts from the `.csv` files generated above and the full LLM results and produces the figures present in the original paper.

By default, assessors are trained on a train split of the MMLU-Pro dataset and evaluated both on that and the Big-Bench Hard.


# TODO
- [ ] make a script from train assessors notebook
  - [ ] connect it with the llm-data folder so that running it automatically computes assessors based on the newly added models
  - [ ] make it evaluate both the ID and OOD performance at the same time!
- [ ] merge all of this folder with the llm_data and the assessors folder




