# PredictaBoard

[PredictaBoard](https://predictaboard.github.io/) is a novel benchmark for measuring the predictability of Large Language Models.

You can find our datasets on huggingface [(benchmarks)](https://huggingface.co/datasets/kvoudouris/PredictaBoard_Benchmarks) [(LLM results)](https://huggingface.co/datasets/kvoudouris/PredictaBoard_LLMs).

You can contribute in two ways: 

## 1. Add results for a new LLM
Follow the instructions in the `llm_data` folder. Then, look at the instructions in the `paper` folder to train and evaluate the baseline assessors on the new model (for now, this requires manually moving the LLM results to that folder, but we're working on that). 

## 2. Add a new assessor
For now, that requires significant changes to the notebook `paper/train_embeddings_assessors.ipynb` and the associated utils. We are working on making it easier to add a new assessor.



# Cite our work
```bibtex
@misc{pacchiardi2025predictaboardbenchmarkingllmscore,
      title={{PredictaBoard}: Benchmarking {LLM} Score Predictability},
      author={Lorenzo Pacchiardi and Konstantinos Voudouris and Ben Slater and Fernando Martínez-Plumed and José Hernández-Orallo and Lexin Zhou and Wout Schellaert},
      year={2025},
      eprint={2502.14445},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14445},
}
```
