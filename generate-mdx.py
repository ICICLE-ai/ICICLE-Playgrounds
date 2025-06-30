import inspect
from docstring_to_markdown import convert
from icicle_playgrounds.pydantic.plug_n_play import Image
from icicle_playgrounds.pydantic.patra_model_cards import PatraModelCard, PatraBiasAnalysis,PatraXAIAnalysis,PatraAIModel

def generate_mdx(cls, output_file):
    # Get the docstring
    doc = inspect.getdoc(cls)

    # Convert to markdown
    markdown = convert(doc) if doc else ""

    # Create MDX content
    mdx_content = f"""---
title: {cls.__name__}
---

import {{Callout}} from 'nextra/components'

{markdown}

## Properties

"""

    # Add properties from the model if it's a Pydantic model
    if hasattr(cls, "model_fields"):
        for field_name, field in cls.model_fields.items():
            mdx_content += f"### {field_name}\n"
            mdx_content += f"Type: `{field.annotation}`\n\n"

    # Write to file
    with open(f"docs/{output_file}.mdx", "w") as f:
        f.write(mdx_content)

# Generate MDX files for each class
generate_mdx(Image, "pydantic/image")
generate_mdx(PatraModelCard, "pydantic/patra-model-card")
generate_mdx(PatraBiasAnalysis, "pydantic/patra-bias-analysis")
generate_mdx(PatraXAIAnalysis, "pydantic/patra-xai-analysis")
generate_mdx(PatraAIModel, "pydantic/patra-ai-model")