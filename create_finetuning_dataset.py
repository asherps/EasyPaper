import json
import arxiv
from get_finetuning_prompt import get_paper_summary
import time
import random
from tqdm import tqdm


def get_ml_papers(num_papers=1):
    search = arxiv.Search(
        query="cat:cs.LG OR cat:cs.AI OR cat:stat.ML",
        max_results=num_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    return [result.get_short_id() for result in search.results()]


def create_dataset(num_papers=50):
    paper_ids = get_ml_papers(num_papers)
    dataset = []

    system_prompt = """Objective: Given a summary of a research project, generate a high-quality academic paper in the style of reputable machine learning publications, such as those found in conferences like NeurIPS, ICML and ICLR. The total length should be 5000 to 6000 words.

    Structure: Follow the typical structure of a machine learning research paper, which includes the following:

    1. Title: Concise, informative, and reflective of the research focus (max 15 words).
    2. Abstract: A brief summary (250-400 words) outlining the problem, the proposed solution, key results, and the significance of the findings.
    3. Introduction: Introduce the research problem, its importance, existing challenges, and the motivation behind the study. Clearly state the contributions of the paper (max 800 words).
    4. Related Work: Review relevant literature, summarizing existing methods, their limitations, and how the proposed work differs or improves upon them (max 1000 words).
    5. Methodology: Provide a detailed description of the proposed method, including mathematical formulations, algorithms, and theoretical foundations. Highlight any novel techniques or modifications to existing methods (max 1200 words).
    6. Experiments: Present a comprehensive experimental setup, including datasets used, evaluation metrics, baselines, and implementation details. Provide comparative analysis, ablation studies, and visualizations like graphs, tables, and charts to validate the proposed approach (max 800 words).
    7. Results: Report key findings with quantitative and qualitative results, including performance metrics, statistical significance, and visual comparisons where applicable (max 600 words).
    8. Discussion: Analyze the results, discuss the strengths and weaknesses of the approach, and suggest potential future work or improvements (max 600 words).
    9. Conclusion: Summarize the main findings, contributions, and implications of the research (max 400 words).
    10. References: Include citations formatted according to a recognized academic style (e.g., IEEE or APA), covering relevant and recent works in the field.

    Content Requirements:

    1. Clarity and Precision: Use clear, precise, and formal academic language. Avoid ambiguous statements and ensure that technical terminology is used correctly.
    2. Technical Rigor: Ensure the content reflects a deep understanding of machine learning concepts, including mathematical rigor where appropriate. Use equations, algorithms, and technical diagrams to explain the proposed method.
    3. Novelty and Contribution: Highlight the novel aspects of the research. Clearly differentiate between existing work and the new contributions of the paper.
    4. Citations and References: Reference foundational and recent papers to provide context and validate claims. Ensure citations are accurately used to support statements and comparisons.
    5. Evaluation and Analysis: Include a thorough evaluation of the proposed method, comparing it with state-of-the-art techniques. Use statistical methods to validate performance improvements.
    6. Figures and Tables: Use high-quality figures and tables to illustrate key points, results, and comparisons. Ensure all visual elements are well-labeled, easy to interpret, and add value to the text.
    
    Formatting and Presentation:

    Use a professional and clean layout, adhering to common standards in academic publishing.
    Ensure consistency in font size, headings, and style throughout the document.
    Include captions for all figures and tables, and ensure they are referenced appropriately within the text.

    Technical Specifics:

    Include specific details about the datasets used, such as source, size, preprocessing steps, and splits.
    Provide hyperparameter settings, model architectures, and training details to ensure reproducibility.
    Discuss computational resources used, such as hardware specifications and runtime.

    Tone and Voice:

    Maintain a formal, objective, and impersonal tone throughout the paper.
    Avoid first-person language; instead, use passive or third-person constructions.
    Be persuasive and evidence-driven, backing claims with data and logical reasoning.
    Target Audience: Write for an audience of machine learning researchers, data scientists, and academics familiar with the field. Assume knowledge of basic ML concepts but explain any advanced or novel techniques in detail.

    Additional Requirements:
    Target Audience: Write for an audience of machine learning researchers, data scientists, and academics familiar with the field. Assume knowledge of basic ML concepts but explain any advanced or novel techniques in detail."""

    for i, arxiv_id in enumerate(
        tqdm(paper_ids, desc="Processing papers", total=num_papers), 1
    ):
        retries = 3
        while retries > 0:
            try:
                summary, content = get_paper_summary(arxiv_id)
                if summary and content:
                    entry = {
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {"role": "user", "content": summary},
                            {"role": "assistant", "content": content},
                        ]
                    }
                    dataset.append(entry)

                    # Save every 50 papers
                    if i % 50 == 0:
                        save_dataset(dataset, f"finetuning_dataset_{i}.jsonl")
                        dataset = []  # Clear the dataset after saving
                break
            except Exception as e:
                print(f"Error processing {arxiv_id}: {str(e)}")
                retries -= 1
                if retries > 0:
                    time.sleep(random.uniform(1, 3))  # Random delay before retry
                else:
                    print(f"Failed to process {arxiv_id} after 3 attempts")

        time.sleep(random.uniform(1, 2))  # Rate limiting

    # Save any remaining entries
    if dataset:
        save_dataset(dataset, f"finetuning_dataset_final.jsonl")

    return dataset


def save_dataset(dataset, filename="finetuning_dataset.jsonl"):
    with open(filename, "a") as f:  # Changed to "a" mode for appending
        for entry in dataset:
            json.dump(entry, f)
            f.write("\n")


def main():
    dataset = create_dataset()
    save_dataset(dataset)
    print(f"Dataset created with {len(dataset)} entries.")


if __name__ == "__main__":
    main()
