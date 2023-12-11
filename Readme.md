# Playing with Hugging Face's Language Models and Transformers Library

This repository has code to check and play with Hugging Face's Language Models under the [Transformers library](hhttps://huggingface.co/docs/transformers/index) and PyTorch.

**Important Note**
There are some TODOs here which I would like to get to at some point but, as this is a learning exercise, I am not sure when I will get to them. 

## Requirements

### PIP 

There is a requirements.txt file that can be used to install the required packages. It may not be complete as I use conda for this repository.

```bash
pip install -r requirements.txt
```

### Conda (miniconda)

I personally use miniconda to manage my python environments. I have included a `transformers.yml`` file that can be used to create a conda environment.

```bash
conda env create -f environment.yml
```

I exported the environment using the following command:

```bash
conda env export > transformers.yml
```
# Scripts 

## Model Compare

model_compare.py takes an argument "input" and runs several models and saves the results, including resource usage, in a duckdb database.

The script can be run with the following command:

```bash
python model_compare.py input
```



### TODO

- [ ] get_model_size is not working


## Summarize

summarize.py takes a term or phrase as an argument, an optional language_code (defaults to `en`), looks it up in wikipedia, and summarizes the article with a 500 char limit using the [BART](https://huggingface.co/facebook/bart-large-cnn) model and the Transformers library.

The script can be run with the following command:

```bash
python summarize.py "term or phrase" lnaguage_code
```

### TODO

- [ ] If a term does not have a wikipedia page, the script fails. Need to add a check for this and return a message. The library I am using to get the page is [Wikipediaapi](https://wikipedia-api.readthedocs.io/en/latest/wikipediaapi/api.html) and it does not support searching for a term. I may need to use a different library such as https://github.com/goldsmith/Wikipedia
- [ ] Try other models and check the results and resource usage. In the model_compare script I have a function to get the resource usage.

