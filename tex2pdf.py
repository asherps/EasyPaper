import requests
from io import BytesIO
from PyPDF2 import PdfReader


def get_paper_content(arxiv_id):
    # Search for the paper
    url = f"https://arxiv.org/pdf/1706.03762.pdf"
    response = requests.get(url)

    if response.status_code == 200:
        pdf_content = BytesIO(response.content)
        reader = PdfReader(pdf_content)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    else:
        return "Error: Paper not found."


if __name__ == "__main__":
    content = get_paper_content(arxiv_id)
    print(content)
