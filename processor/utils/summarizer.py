"""
Summary generator using Ollama LLM API.
"""

from .llm_client import ollama_generate

def generate_summary(transcript_path: str, model: str, max_tokens: int = 2048, mode: str = "dense") -> str:
    """
    Summarize the transcript contained at relative path <transcript_path>.

    == Preconditions ==
    'mode' == 'dense' (single paragraph summary) or bullet (5-7 bullet point summary)
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read()

    if mode == "bullet":
        prompt = (
            "You are an assistant that summarizes transcripts.\n"
            "Transcript:\n"
            f"{text}\n\n"
            "Provide 5–7 bullet points summarizing the main ideas discussed."
        )
    else:
        prompt = (
            "You are an assistant that summarizes transcripts into a concise paragraph.\n"
            "Answer in plaintext only. Do not use bold, italics, markdown, or any special formatting.\n"
            "Transcript:\n"
            f"{text}\n\n"
            "Provide a concise paragraph summary."
        )

    output = ollama_generate(prompt=prompt, model=model, max_tokens=max_tokens)
    return output
