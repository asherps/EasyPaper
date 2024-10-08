import os
from flask import Flask, render_template, request
import openai
from openai import OpenAI

app = Flask(__name__)

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": """Objective: Given a summary of a research project, generate a high-quality academic paper in the style of reputable machine learning publications, such as those found in conferences like NeurIPS, ICML and ICLR. The total length should be 5000 to 6000 words.

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
                            Target Audience: Write for an audience of machine learning researchers, data scientists, and academics familiar with the field. Assume knowledge of basic ML concepts but explain any advanced or novel techniques in detail.""",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            return render_template(
                "index.html",
                prompt=prompt,
                response=response.choices[0].message.content,
            )
        except Exception as e:
            return render_template("index.html", prompt=prompt, error=str(e))
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
