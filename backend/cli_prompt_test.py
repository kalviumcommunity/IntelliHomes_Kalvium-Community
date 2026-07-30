from prompts.renderer import render_prompt

def main():
    system_prompt, user_prompt = render_prompt(
        context="Property registration",
        question="What is a Title Deed?"
    )

    print("===== SYSTEM PROMPT =====")
    print(system_prompt)

    print("\n===== USER PROMPT =====")
    print(user_prompt)


if __name__ == "__main__":
    main()