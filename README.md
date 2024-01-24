![Logo WTW](/img/wtw_pic.png)


<h1 align="center">
<span>What the Weight?! A Unified Framework for Zero-Shot Knowledge Composition</span>
</h1>

[![arxiv preprint](https://img.shields.io/badge/arXiv-2208.01575-b31b1b.svg)](https://arxiv.org/abs/2401.12756)

## Paper Abstract

The knowledge encapsulated in a model is the core factor determining its final performance on downstream tasks. 
Much research in NLP has focused on efficient methods for storing and adapting different types of knowledge, e.g., in dedicated modularized structures, and on how to effectively combine these. However, given the many possible options, a thorough understanding of the mechanisms involved is missing, and hence it remains unclear which strategies to utilize. 
To address this research gap, we propose a novel framework for zero-shot module composition, which encompasses existing and some novel variations for selecting, weighting, and combining parameter modules under a single. Focusing on the scenario of domain knowledge and adapter layers, our framework provides a systematic unification of concepts, allowing us to conduct the first comprehensive benchmarking study on various zero-shot knowledge composition strategies. In particular, we test two module combination methods and five selection and weighting strategies for their effectiveness and efficiency in an extensive experimental setup. Our results highlight the efficacy of ensembling, but also hint at the power of simple though often-ignored weighting methods. Further in-depth analysis allow us to understand the role of weighting vs. top-k selection, and show that, to a certain extent, the performance of adapter composition can even be predicted.

------------------------
## Getting Started

We conducted all our experiments with Python 3.10. Before getting started, make sure you install the requirements listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Repository Description

This repository contains all code and data needed to reproduce the experiments and results reported in our paper.

### Data 

A brief description of the files in *data* is:

- **experimental_results.xlsx**
    - Contains all results we produced throughout our experiments.

- **efficiency calculation.xlsx**
    - Contains the results of the efficiency calculations for DeBERTa and GPT-2.



Additional data sources used:
- [C4 Corpus](https://github.com/allenai/c4-documentation)
- [yelp.com Corpus](https://www.yelp.com/dataset)


**Note:** The computed domain adapters could not be uploaded to GitHub due to size constraints. Find it on https://xxxx


### Code

Includes all python files and notebooks subject to this paper.

A brief description of the files in *code* is:

- **creation_of_paper_plots.ipynb**
    - This notebook can be used to recreate all plots present in the paper, based on the experimental results.

- **run_clm_adapter.py**
    - This script is based on the *run_clm.py* script of the Adapter Hub, which can be found [here](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/language-modeling). It is used to train the domain adapters using a causal language modeling loss.
- **run_mlm_adapter.py**
    - This script is based on the *run_mlm.py* script of the Adapter Hub, which can be found [here](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/language-modeling). It is used to train the domain adapters using a masked language modeling loss.





### Shell Files

Includes example shell files to run the python code.

- **train_clm_adapter.sh**
    - Contains a shell script to run *run_clm_adapter.py* and trains a domain adapter using the specified variables. 

- **train_mlm_adapter.sh**
    - Contains a shell script to run *run_mlm_adapter.py* and trains a domain adapter using the specified variables. 

- **evaluate_clm_models.sh**
    - Contains a shell script to run *run_clm_adapter.py* and evaluates the adapters using our presented framework.

- **evaluate_mlm_models.sh**
    - Contains a shell script to run *run_mlm_adapter.py* and evaluates the adapters using our presented framework.



------------------------
## References

Please use the following bibtex entry if you use this model in your project (TBD):
 
```bib
@inproceedings{,
    title = "What the Weight?! A Unified Framework for Zero-Shot Knowledge Composition",
    author = "Holtermann, Carolin and Frohmann, Markus and Rekabsaz, Navid and Lauscher, Anne",
    editor = "",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "",
    doi = "",
    pages = "",
}
```


---
*Author contact information: carolin.holtermann@uni-hamburg.de*


## License

All source code is made available under a MIT license. See `LICENSE.md` for the full license text.


