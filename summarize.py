from transformers import BartTokenizer, BartForConditionalGeneration

import wikipediaapi


def get_wikipedia_data(term: str):
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="Test (merlin@example.com)",
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )

    p_wiki = wiki_wiki.page(term)
    text = p_wiki.text
    # print(text)
    return text


def summarize(input_text):
    # use Facebook's BART model to summarize the text
    # Load model and tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    # Tokenize input text
    inputs = tokenizer([input_text], max_length=1024, return_tensors="pt")

    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"], num_beams=4, max_length=500, early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


if __name__ == "__main__":
    input_text = sys.argv[1]
    print("Summarizing Wikipedia article for " + input_text + "...")
    text = get_wikipedia_data(input_text)
    print(summarize(text))
