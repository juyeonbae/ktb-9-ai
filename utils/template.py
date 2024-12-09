from langchain import PromptTemplate


def image_editor_template(prompt):         
    prompt_template = PromptTemplate.from_template(
        """
            prompt: {prompt}
            
            Make sure prompt must be in English when using the tool.
        """
    )
    
    prompt = prompt_template.format(prompt=prompt)
    return prompt