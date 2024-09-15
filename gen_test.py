from openai import OpenAI
import random
import arxiv
from PyPDF2 import PdfReader
import requests
from io import BytesIO
import tiktoken

client = OpenAI()
""" summarizes a random paper from arxiv """


def get_paper_content(arxiv_id):
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(search.results())
    response = requests.get(paper.pdf_url)
    pdf_content = BytesIO(response.content)
    reader = PdfReader(pdf_content)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) > 10000:
        tokens = tokens[:10000]
        text = encoding.decode(tokens)
    return text


def summarize_paper(text):
    prompt = f"""Here is an academic paper: <paper>{text}</paper>. Please provide a comprehensive summary of this machine learning paper that would allow someone to recreate the paper from scratch. Include all relevant details about how the paper is written, all technical details of the methodology, and all empirical findings. Your summary should cover the following sections in detail:

1. Abstract:
   - Concise overview of the paper's main contributions and results

2. Introduction:
   - Detailed context of the research
   - Specific research questions or hypotheses
   - Comprehensive literature review and positioning of the work

3. Related Work:
   - Thorough review of relevant prior work
   - Clear explanation of how this paper advances the state of the art

4. Method:
   - Detailed description of the proposed approach
   - Precise mathematical formulations and algorithms
   - Architectural details of any neural networks or models used
   - Explanation of any novel techniques or adaptations

5. Experimental Setup:
   - Detailed description of datasets used
   - Precise specifications of hardware and software
   - Comprehensive explanation of evaluation metrics
   - Thorough description of baselines and comparison methods

6. Results and Analysis:
   - All empirical findings, including exact numbers, statistics, and measurements
   - Detailed descriptions of any graphs, tables, or figures
   - Comprehensive analysis of results, including ablation studies
   - Discussion of any unexpected or anomalous results

7. Discussion:
   - In-depth interpretation of all results
   - Thorough analysis of the implications of the findings
   - Critical examination of limitations and potential biases

8. Conclusion:
   - Precise summary of key contributions
   - Specific suggestions for future research directions

Ensure that the summary is extremely detailed, technically precise, and captures all essential information from the paper. It should provide enough information for a researcher to replicate the study or build directly upon this work."""


# List of random ML paper arXiv IDs
paper_ids = [
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "2005.14165",  # GPT-3
    # Add more arXiv IDs here
]

# Select a random paper ID
random_paper_id = random.choice(paper_ids)

# Get the paper content
paper_content = get_paper_content(random_paper_id)

# Generate the summary
summary = summarize_paper(paper_content)

print("Original paper ID:")
print(random_paper_id)
print("\nGenerated summary:")
print(summary)

"""
Ensure the output is formatted as a LaTeX document. Include the necessary preamble, sections, and commands to render the document correctly. Use appropriate LaTeX packages for figures, tables, and references.
"""
