"""
processor.py
Open-source LLM post-processing of audio transcripts (perfect to be used in tandem with audiotranscriber.py)

This is the API for the processor API.
The processor takes a finished transcript of audio or video in either plaintext or .srt
(possibly from audiotranscriber.py) and provides 1.) a summary and 2.) automatic chapter/topic segmentation.

This file also provides the CLI wrapper for post-processing transcripts with open-source LLMs (via Ollama).

Note: Ollama must be downloaded on the user's machine.
"""

from pathlib import Path
import argparse
from utils.summarizer import generate_summary
from utils.segmenter import generate_chapters

# Default open-source models.
AVAILABLE_MODELS = ["llama3", "mistral", "phi3", "llama2"]

def _seconds_to_timestamp(secs: float) -> str:
    """Parse <secs> to a hh:mm:ss string."""

    hours = int(secs // 3600)
    mins = int((secs % 3600) // 60)
    new_secs = int(secs % 60)
    return f"{hours:02d}:{mins:02d}:{new_secs:02d}"


def processor() -> None:
    """Processor API."""

    parser = argparse.ArgumentParser(description="Transcript LLM (Ollama) post-processor")
    parser.add_argument("transcript", help="Path to transcript, .txt or .srt")
    parser.add_argument("--summary", action="store_true", help="Generate summary")
    parser.add_argument("--chapters", action="store_true", help="Generate LLM-created topic chapters")
    parser.add_argument("--model", default="llama3",
                        help=f"Ollama model name (default=llama3). Possible values: {AVAILABLE_MODELS}")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens permitted for LLM calls")
    parser.add_argument("--chunk_seconds", type=int, default=120,
                        help="Estimated min. length per clustered chapter during segmentation")
    args = parser.parse_args()

    if not args.summary and not args.chapters:
        raise SystemExit("Please specify --summary and/or --chapters")

    if args.summary:
        print("\nprocessor: === Generating Summary... ===")
        result = generate_summary(args.transcript, model=args.model, max_tokens=args.max_tokens)
        output = "\n".join([line.strip() + "." for line in result.split(".") if line.strip()])
        output.join('.')
        print(output)
        print("\n")

    if args.chapters:
        print("processor: === Generating Chapters... ===")
        chapters = generate_chapters(
            args.transcript,
            model=args.model,
            max_tokens=args.max_tokens,
            min_chunk_seconds=args.chunk_seconds
        )
        for chapter in chapters:
            start  = _seconds_to_timestamp(chapter["start"])
            end = _seconds_to_timestamp(chapter["end"])
            print(f"[{start} – {end}] {chapter['title']}:")
            summary_lines = "\n".join([s.strip() + "." for s in chapter['summary'].split(".") if s.strip()])
            print(summary_lines)
            print("---")


def main():
    """
    Main API.
    """
    import sys

    # This allows the code to be run within an IDE (non-CLI environment).
    if len(sys.argv) == 1:
        sys.argv += [
            "YOUR_FILE_PATH_HERE",  # Change this to your file path.
            "--summary",
            "--chapters"
        ]
        print("processor: Running with default settings inside IDE...")

    # Convert path to be relative to project root (i.e, one level above the processor/ folder).
    project_root = Path(__file__).resolve().parents[1]
    transcript_arg = sys.argv[1]
    transcript_path = (project_root / transcript_arg).resolve()
    sys.argv[1] = str(transcript_path)

    processor()


if __name__ == "__main__":
    main()
