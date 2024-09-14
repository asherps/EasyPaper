import os
import subprocess
import shutil

def convert_latex_to_pdf(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Copy all .tex and .sty files to the output folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.tex', '.sty')):
            shutil.copy(os.path.join(input_folder, filename), output_folder)

    # Find the main .tex file (assuming it's the only one or the first one)
    main_tex_file = next(f for f in os.listdir(output_folder) if f.endswith('.tex'))

    # Change to the output directory
    os.chdir(output_folder)

    # Run pdflatex twice to resolve references
    for _ in range(2):
        subprocess.run(['pdflatex', '-interaction=nonstopmode', main_tex_file], check=True)

    # Clean up auxiliary files
    for filename in os.listdir():
        if filename.endswith(('.aux', '.log', '.out', '.toc')):
            os.remove(filename)

    print(f"PDF generated: {os.path.splitext(main_tex_file)[0]}.pdf")

# Example usage
input_folder = '/path/to/input/folder'
output_folder = '/path/to/output/folder'
convert_latex_to_pdf(input_folder, output_folder)