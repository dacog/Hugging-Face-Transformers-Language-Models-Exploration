# this file is used to test the performance of the models and store the results in a database
# it tests a model and then frees up the memory before testing the next model
# this is to avoid memory issues when testing multiple models
# it also logs the results to a log file
# the log file can be used to check if any errors occurred during testing
# the log file can also be used to check the performance of each model
import psutil
import time
import duckdb
import logging
import gc
import sys
import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    T5ForConditionalGeneration,
)
import transformers

MAX_NEW_TOKENS = 200  # Maximum number of new tokens to generate

# Set NUMEXPR_MAX_THREADS to the number of available cores minus one
num_cores = psutil.cpu_count(logical=False)  # Get physical core count
if num_cores and num_cores > 1:
    os.environ['NUMEXPR_MAX_THREADS'] = str(num_cores - 1)
else:
    os.environ['NUMEXPR_MAX_THREADS'] = '1'  # Fallback to 1 if single-core or unable to determine


# Initialize logging
logging.basicConfig(
    filename="model_testing.log", level=logging.INFO, format="%(asctime)s: %(message)s"
)

# Connect to DuckDB
conn = duckdb.connect("model_results.duckdb")
conn.execute(
    "CREATE TABLE IF NOT EXISTS model_performance (input TEXT, model TEXT, execution_time FLOAT, cpu_usage FLOAT, memory_usage FLOAT, model_size FLOAT, output TEXT)"
)


def get_model_size(model_name):
    cache_dir = transformers.file_utils.default_cache_path
    model_dir = os.path.join(cache_dir, model_name)
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert bytes to megabytes


def measure_performance_and_store(model_func, model_name, input_text, tokenizer, conn):
    try:
        logging.info(f"Testing model: {model_name}")
        cpu_start = psutil.cpu_percent(interval=None)
        mem_start = psutil.virtual_memory().used
        start_time = time.time()

        result = model_func(input_text, tokenizer)

        cpu_end = psutil.cpu_percent(interval=None)
        mem_end = psutil.virtual_memory().used
        end_time = time.time()

        execution_time = end_time - start_time
        cpu_usage = cpu_end - cpu_start
        memory_usage = (mem_end - mem_start) / (1024 * 1024)  # Convert to MB
        model_size = get_model_size(model_name)

        conn.execute(
            "INSERT INTO model_performance VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                input_text,
                model_name,
                execution_time,
                cpu_usage,
                memory_usage,
                model_size,
                result,
            ),
        )

        logging.info(
            f"Model {model_name} tested successfully. Execution Time: {execution_time}, CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage} MB, Model Size: {model_size} MB \nwith result: \n{result}"
        )
    except Exception as e:
        logging.error(f"Error testing model {model_name}: {str(e)}")


# List of models to test
models_to_test = [
    {"name": "google/flan-t5-small", "type": "seq2seq"},
    #{"name": "google/flan-t5-xxl", "type": "conditional_generation"},
    {"name": "google/flan-t5-large", "type": "conditional_generation"},
    {"name": "MBZUAI/LaMini-Flan-T5-783M", "type": "text2text-generation"},
    # Add other models here
]

if __name__ == "__main__":
    input_text = sys.argv[1]
    # input_text = sys.argv[1] if len(sys.argv) > 1 else "Default input text"

    for model_info in models_to_test:
        model_name = model_info["name"]
        model_type = model_info["type"]

        if model_type == "seq2seq":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model_func = lambda text, tok: tok.batch_decode(
                model.generate(**tok(text, return_tensors="pt"), max_new_tokens=MAX_NEW_TOKENS),
                skip_special_tokens=True,
            )
        elif model_type == "conditional_generation":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            model_func = lambda text, tok: tok.batch_decode(
                model.generate(**tok(text, return_tensors="pt"), max_new_tokens=MAX_NEW_TOKENS),
                skip_special_tokens=True,
            )
        elif model_type == "text2text-generation":
            tokenizer = None
            model = pipeline("text2text-generation", model=model_name)
            model_func = lambda text, tok: model(text)[0]["generated_text"]

        measure_performance_and_store(
            model_func, model_name, input_text, tokenizer, conn
        )

        # Free up memory
        del model, tokenizer
        gc.collect()

    # Close database connection
    conn.close()
