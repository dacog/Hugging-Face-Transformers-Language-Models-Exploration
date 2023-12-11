from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline


def flan(input):
    print("flan running...")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    inputs = tokenizer(input, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def flan_xxl(input):
    print("flan_xxl running...")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")

    input_ids = tokenizer(input, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])


def flan_large(input):
    print("flan_large running...")

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

    input_ids = tokenizer(input, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])


def lamini_flan_t5(input):
    print("lamini_flan_t5 running...")

    checkpoint = "MBZUAI/LaMini-Flan-T5-783M"

    model = pipeline("text2text-generation", model=checkpoint)

    # input_prompt = 'Please let me know your thoughts on the given place and why you think it deserves to be visited: \n"Barcelona, Spain"'
    generated_text = model(input, max_length=512, do_sample=True)[0]["generated_text"]

    return generated_text


if __name__ == "__main__":
    # get input from command line as a string
    input = sys.argv[1]
    flan = flan(input)
    flan_xxl = flan_xxl(input)
    flan_large = flan_large(input)
    lamini_lm_t5_783M = lamini_flan_t5(input)

    print(
        f"""
for input: {input}
flan: {flan},
flan_xxl: {flan_xxl},,
flan_large: {flan_large},,
lamini_lm_t5_783M: {lamini_lm_t5_783M}
"""
    )
