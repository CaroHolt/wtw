
<h1 align="center">
<span>What the Weight?! A Unified Framework for Zero-Shot Knowledge Composition</span>
</h1>

## Paper Abstract
The knowledge encapsulated in a model is the core factor determining its final performance on down-stream tasks. Much research in NLP has focused on efficient methods for storing and adapting different types of knowledge, e.g., in dedicated modularized structures, and on how to effectively combine these modules, e.g., via  parameter averaging at test time. However, given the many possible options in composing knowledge, a thorough understanding of the mechanisms involved is missing, and hence it remains unclear which strategies to utilize.  
In this work, we address this research gap by proposing a novel framework for zero-shot module composition, which encompasses existing and some novel variations for selecting, weighting, and combining parameter modules under a single unified notion. Focusing on the example of domain knowledge and adapter layers, our framework provides a systematic unification of concepts, allowing us to conduct the first comprehensive benchmarking study on various zero-shot knowledge composition strategies. In particular, we test two module combination methods (parameter averaging, output ensembling), and five selection and weighting strategies (uniform, and based on entropy, domain prior, TF-IDF, and semantic similarity) for their effectiveness and efficiency on 21 training and 10 test domains across three models. Our results highlight the efficacy of ensembling, but also hint at the power of simple though often-ignored weighting methods. We further conduct various in-depth analyses, that, for instance, allow us to understand the role of weighting vs. top-k selection, and we show that, to a certain extent, the performance of an adapter composition can even be predicted.  
------------------------
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
    - This notebook can be used to evaluate the language model bias of different model architectures using the ABBA annotation corpus as a test set. This is done by calculating the models perplexity for stereotypically and anti-stereotypically biased sentences and performing a paired t-test on the results.

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

- **eval_clm_adapter.sh**
    - Contains a shell script to run *run_clm_adapter.py* and evaluates the adapters using our presented framework.

- **eval_mlm_adapter.sh**
    - Contains a shell script to run *run_mlm_adapter.py* and evaluates the adapters using our presented framework.




------------------------
## Citation

```
@inproceedings{xxx,
  title={xxx},
  author={xxx},
  booktitle={...},
  year={xxx}
}
```


---
*Author contact information:*


