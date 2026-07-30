from prompts.templates import SYSTEM_TEMPLATE, USER_TEMPLATE


def render_prompt(context: str, question: str):
    return (
        SYSTEM_TEMPLATE.format(context=context),
        USER_TEMPLATE.format(question=question),
    )