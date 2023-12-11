from transformers import BartTokenizer, BartForConditionalGeneration

import wikipediaapi
import sys


def get_wikipedia_data(term: str, language: str = "en") -> str:
    print("Fetching Wikipedia data for " + term + " in " + language + "...")
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="Test (merlin@example.com)",
        language=language,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )

    # Try fetching the page directly
    p_wiki = wiki_wiki.page(term)
    if p_wiki.exists():
        return p_wiki.text

    # If direct fetch fails, use search
    # search_results = wiki_wiki.search(term, results=5)
    # if not search_results:
    #     return "No relevant Wikipedia article found."

    # # Attempt to fetch the top result from search
    # likely_page = wiki_wiki.page(search_results[0])
    # if likely_page.exists():
    #     return likely_page.text

    return "No relevant Wikipedia article found."

def summarize(input_text):
    # use Facebook's BART model to summarize the text
    # Load model and tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    # Tokenize input text
    inputs = tokenizer(
        [input_text], 
        max_length=1024, 
        truncation=True,
        return_tensors="pt"
        )

    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"], 
        num_beams=4, 
        max_length=500, 
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


if __name__ == "__main__":
    input_text = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "en"

    print("Summarizing Wikipedia article for " + input_text + "...")
    text = get_wikipedia_data(input_text, language)
    print(summarize(text))
