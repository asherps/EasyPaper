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
        if token_count + page_tokens > 0:
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
    prompt = f"""Here is an academic paper: <paper>{text}</paper>. Please provide a comprehensive summary of this academic paper that would allow someone to recreate the paper from scratch. Include all relevant details about how the paper is written, all technical details of the methodology, and all empirical findings. Your summary should cover the following elements in detail:

1. Introduction and Background:
   - Detailed context of the research
   - Specific research questions or hypotheses
   - Comprehensive literature review

2. Methodology:
   - Detailed experimental design
   - Precise descriptions of all techniques and procedures used
   - Exact specifications of any equipment or software employed
   - Complete details of data collection methods
   - Thorough explanation of data analysis techniques

3. Results:
   - All empirical findings, including exact numbers, statistics, and measurements
   - Detailed descriptions of any graphs, tables, or figures
   - Any unexpected or anomalous results

4. Discussion:
   - In-depth interpretation of all results
   - Comprehensive comparison with existing literature
   - Thorough analysis of the implications of the findings

5. Conclusion:
   - Precise summary of key findings
   - Detailed discussion of limitations
   - Specific suggestions for future research

Ensure that the summary is extremely detailed, technically precise, and captures all essential information from the paper. It should provide enough information for a researcher to replicate the study or build directly upon this work."""

    response = client.chat.completions.create(
        model="gpt-4o",  # or another suitable model
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes academic papers.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
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
