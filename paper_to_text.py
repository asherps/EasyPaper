import arxiv
from get_finetuning_prompt import get_paper_summary
import random


def get_ml_papers(num_papers=2):
    search = arxiv.Search(
        query="cat:cs.LG OR cat:cs.AI OR cat:stat.ML",
        max_results=num_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    return [result.get_short_id() for result in search.results()]


def convert_papers_to_text():
    paper_ids = get_ml_papers(2)
    paper_texts = []

    for arxiv_id in paper_ids:
        summary, content = get_paper_summary(arxiv_id)
        if summary and content:
            paper_texts.append(
                f"Paper ID: {arxiv_id}\nSummary: {summary}\nContent: {content}"
            )

    return "\n\n".join(paper_texts)


def append_to_system_prompt(paper_texts):
    with open("generate_paper.py", "r") as file:
        content = file.read()

    system_prompt_start = content.find('system_prompt = """')
    if system_prompt_start == -1:
        raise ValueError("System prompt not found in generate_paper.py")

    system_prompt_end = content.find('"""', system_prompt_start + 18)
    if system_prompt_end == -1:
        raise ValueError("System prompt end not found in generate_paper.py")

    new_content = (
        content[:system_prompt_end]
        + "\n\nExample papers:\n"
        + paper_texts
        + content[system_prompt_end:]
    )

    with open("generate_paper.py", "w") as file:
        file.write(new_content)


if __name__ == "__main__":
    paper_texts = convert_papers_to_text()
    append_to_system_prompt(paper_texts)
    print("Papers converted and appended to system prompt in generate_paper.py")
