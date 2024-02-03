from pathlib import Path
from typing import List, Optional, Union

import loguru
import nbformat
import tiktoken
import typer
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

PathLike = Union[str, Path]

logger = loguru.logger
client = OpenAI()

app = typer.Typer()
logger.add("julie.log", rotation="1 MB")


load_dotenv()


# Function to tokenize text and return the number of tokens
def count_tokens(text, model_name):
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))


class Notebook(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    nb_path: Path
    nb_cells: nbformat.NotebookNode
    grouped_content: Optional[List[dict]] = None


def pair_markdown_code_and_outputs(notebook_cells):
    grouped_content = []
    current_markdown_group = []
    current_code_cell = None

    def append_group():
        if current_code_cell or current_markdown_group:
            if current_code_cell:
                outputs = [element.get("text", "") for element in current_code_cell.get("outputs", [])]
            else:
                outputs = []
            grouped_content.append(
                {
                    "markdown": "\n".join(current_markdown_group),
                    "code": current_code_cell["source"] if current_code_cell else "",
                    "outputs": outputs,
                }
            )

    for cell in notebook_cells:
        if cell["cell_type"] == "markdown":
            # Check for a header in markdown
            if any(
                line.strip().startswith(("#", "##", "###", "####", "#####", "######"))
                for line in cell["source"].splitlines()
            ):
                append_group()  # End the current group at a header
                current_markdown_group = []  # Start a new markdown group
                current_code_cell = None

            current_markdown_group.append(cell["source"])

        elif cell["cell_type"] == "code":
            # If there's an existing code cell waiting to be processed, process it first
            if current_code_cell:
                append_group()  # End the current group before starting a new code cell
                current_markdown_group = []  # Reset the markdown group

            # Update current_code_cell to the latest code cell
            current_code_cell = cell

    # Process any remaining content after the loop
    append_group()

    return grouped_content


# Assuming we have a sample notebook 'sample_notebook.ipynb' in the current directory
def read_notebooks(directory: PathLike):
    notebooks: List[Notebook] = []
    directory = Path(directory)
    for file in directory.rglob("*.ipynb"):
        notebooks.append(Notebook(nb_path=file.resolve(), nb_cells=nbformat.read(file, as_version=4)))
    assert len(notebooks) > 0, f"No notebooks found in {directory}"
    return notebooks


def explain_group(group: dict, idx: int, total: int, model_name: str):
    markdown = group["markdown"]
    code = group["code"]
    outputs = group["outputs"]
    if outputs:
        input_content = f"Markdown: {markdown}\n\nCode: {code}\n\nOutputs: {outputs}"
    else:
        input_content = f"Markdown: {markdown}\n\nCode: {code}"
    prompt = f"""Given the following sections with headings, content, code, and outputs, generate an explanation for each section that preserves the headings where relevant:

{input_content}

"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": """Generate a blog from a Jupyter notebook. Each section is demarcated by markdown headers and may include explanatory text, code snippets, and their outputs. The blog should:
- Preserve the original structure indicated by markdown headers.
- Incorporate key insights from both the explanatory text and the associated code snippets.
- Reflect the significance of any code outputs, highlighting how they demonstrate or impact the discussed concepts.
- Remain concise, focusing on the most important points that convey the essence of each section.
- Never write a conclusion or ask for follow up questions.

Note: Readers who may have Python expertise but are not familiar with Qdrant.
Never introduce yourself or greet the reader.
""",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

@app.command()
def list_notebooks(
    notebook_directory: str,
):
    """
    List the notebooks in the specified directory.

    Args:
        notebook_directory (str): The directory containing the notebooks to list.
    """
    notebooks = read_notebooks(notebook_directory)
    for notebook in notebooks:
        logger.info(notebook.nb_path.name)

@app.command()
def generate(
    notebook_directory: str,
    model_name: str = "gpt-4-turbo-preview",
):
    """
    Generate a blog from notebooks in the specified directory using the given model.

    Args:
        model_name (str): The name of the model to use for generating the blog. Defaults to "gpt-4-turbo-preview".
        notebook_directory (str): The directory containing the notebooks to generate the blog from.
    """
    logger.info(f"Generating blog from notebooks in {notebook_directory} using model {model_name}")
    notebooks = read_notebooks(notebook_directory)
    logger.info(f"Read {len(notebooks)} notebooks")
    markdown_string = ""
    for notebook in notebooks:
        cells = notebook.nb_cells.cells
        grouped_content = pair_markdown_code_and_outputs(cells)
        logger.info(f"Notebook {notebook.nb_path.name} has {len(grouped_content)} groups of markdown and code")
        notebook.grouped_content = grouped_content
        with open(Path(notebook_directory) / "BLOG-DRAFT.md", "w+") as f:
            for idx, group in tqdm(enumerate(grouped_content)):
                explanation = explain_group(group=group, idx=idx, total=len(grouped_content), model_name=model_name)
                # logger.info(f"Explanation: {explanation}")
                cell_string = ""
                cell_string += explanation.strip() + "\n\n"
                if group["code"]:
                    cell_string += f"```python\n{group['code']}\n```\n\n"
                if group["outputs"]:
                    cell_string += f"Outputs: {group['outputs']}\n\n"
                f.write(cell_string)
                markdown_string += cell_string


if __name__ == "__main__":
    app()
