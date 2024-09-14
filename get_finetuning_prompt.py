import os
import json
import arxiv
from openai import OpenAI
from PyPDF2 import PdfReader
import requests
from io import BytesIO
import tiktoken

client = OpenAI()


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_paper_content(arxiv_id):
    # Search for the paper
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(search.results())

    # Download the PDF
    response = requests.get(paper.pdf_url)
    pdf_content = BytesIO(response.content)

    # Extract text from PDF
    reader = PdfReader(pdf_content)
    text = ""
    token_count = 0
    for page in reader.pages:
        page_text = page.extract_text()
        page_tokens = num_tokens_from_string(page_text, "cl100k_base")
        if token_count + page_tokens > 10000:
            remaining_tokens = 10000 - token_count
            truncated_text = page_text[
                : int(remaining_tokens * 4)
            ]  # Approximate chars per token
            text += truncated_text
            break
        text += page_text
        token_count += page_tokens
        if token_count >= 10000:
            break

    return text


def summarize_paper(text):
    prompt = f"""Here is an academic paper: <paper>{text}</paper>. Please summarize this academic paper comprehensively while retaining key technical details, methodologies, and results. Your summary should include the following elements:

1. Introduction and Background
2. Research Objectives
3. Methodology
4. Results
5. Discussion
6. Conclusion

Ensure that the summary is accurate and captures the essence of the paper, making it useful for readers who may not have the time to read the full document but need a thorough understanding of the research."""

    response = client.chat.completions.create(
        model="gpt-4",  # or another suitable model
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes academic papers.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=10000,
    )

    return response.choices[0].message.content


def main():
    arxiv_id = input("Enter the arXiv ID of the paper: ")

    try:
        paper_content = get_paper_content(arxiv_id)
        summary = summarize_paper(paper_content)

        print("\nSummary of the paper:")
        print(summary)

        # Optionally, save the summary to a file
        with open(f"{arxiv_id}_summary.txt", "w") as f:
            f.write(summary)
        print(f"\nSummary saved to {arxiv_id}_summary.txt")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
