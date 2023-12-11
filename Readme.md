# Playing with Hugging Face's Language Models and Transformers Library

This repository has code to check and play with Hugging Face's Language Models using the [Transformers library](hhttps://huggingface.co/docs/transformers/index) and PyTorch.

**Important Note**
There are some TODOs here which I would like to get to at some point but, as this is a learning exercise, I am not sure when or if I will get to them. 

## Requirements

### PIP 

There is a requirements.txt file that can be used to install the required packages. _It may not be complete as I use conda for this repository._

```bash
pip install -r requirements.txt
```

### Conda (miniconda)

I personally use miniconda to manage this python environment. You can install it from [here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

I have included a `transformers.yml` file that can be used to create a conda environment with the required packages. You can create the environment with the following command:

```bash
conda env create -f environment.yml
```

I exported the environment using the following command, should you modify the environment and want to update the file:

```bash
conda env export > transformers.yml
```


# Scripts 

Here is a short description of the scripts in this repository.

## Model Compare

model_compare.py takes an argument "input" and runs several models and saves the results, including resource usage, in a duckdb database.

The script can be run with the following command:

```bash
python model_compare.py input
```

**Important Note**
The script will download a lot of models, use all the available cores but one, and use a lot of memory. 

### Results

Here are some logs from running the script with the input _"What is  N. Tesla recognized for?"_

```bash
2023-12-11 23:59:47,501: Testing model: google/flan-t5-small
2023-12-11 23:59:47,898: Model google/flan-t5-small tested successfully. Execution Time: 0.16420340538024902, CPU Usage: 43.0%, Memory Usage: 3.9375 MB, Model Size: 0.0 MB 
with result: 
['a pioneer in the field of physics']
2023-12-12 00:15:58,585: Testing model: google/flan-t5-xl
2023-12-12 00:15:59,797: Model google/flan-t5-xl tested successfully. Execution Time: 1.209467887878418, CPU Usage: 49.699999999999996%, Memory Usage: 4.859375 MB, Model Size: 0.0 MB 
with result: 
['electrical engineering']
2023-12-12 00:16:05,352: Testing model: google/flan-t5-large
2023-12-12 00:16:05,866: Model google/flan-t5-large tested successfully. Execution Time: 0.5079565048217773, CPU Usage: 50.5%, Memory Usage: -4.88671875 MB, Model Size: 0.0 MB 
with result: 
['electric car']
2023-12-12 00:16:14,000: Testing model: MBZUAI/LaMini-Flan-T5-783M
2023-12-12 00:16:15,831: Model MBZUAI/LaMini-Flan-T5-783M tested successfully. Execution Time: 1.829249382019043, CPU Usage: 2.5%, Memory Usage: -186.125 MB, Model Size: 0.0 MB 
with result: 
N. Tesla is recognized for his contributions to the development of the Tesla coil, the Tesla coil
```

### TODO

- [ ] get_model_size is not working
- [ ] add more models
- [ ] use and publish the results
- [ ] analyse the results


## Summarize

`summarize.py` takes a term or phrase as an argument, an optional language_code (defaults to `en`), looks it up in wikipedia, and summarizes the article with a 500 char limit using the [BART](https://huggingface.co/facebook/bart-large-cnn) model and the Transformers library.

The script can be run with the following command:

```bash
python summarize.py "term or phrase" lnaguage_code
```

### TODO

- [ ] If a term does not have a wikipedia page, the script fails. Need to add a check for this and return a message. The library I am using to get the page is [Wikipediaapi](https://wikipedia-api.readthedocs.io/en/latest/wikipediaapi/api.html) and it does not support searching for a term. I may need to use a different library such as https://github.com/goldsmith/Wikipedia
- [ ] Try other models and check the results and resource usage. In the model_compare script I have a function to get the resource usage.


# Other Information

This information may be important as I am trying to compare the resource usage of the different models.

**I am using **
- an AMD Ryzen 7 5700G with Radeon Graphics
- 96 gb ram 
- 1 TB NVME SSD

**I am running **
- Linux Mint 21 with KDE Plasma, 
- Kernel 5.15.0-79-generic
- conda 4.14.0
- Python 3.11.5