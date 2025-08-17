"""
audiotranscriber.py

Transcription of audio and video files using OpenAI Whisper (Windows-compatible).

Features:
- transcribe(): handles all audio/video transcription, optional timestamped and cleaned output
- clean_transcript(): allows post-processing of known mishearings (e.g., 'excess' → 'XS')
- Generates .txt and .srt output as needed
"""
import math
import os

import whisper
from whisper.model import Whisper
import sys
from typing import Union
import re
import torch

# == UTILITIES ==
def _split_text_sentences(text: str) -> list[str]:
    """Split <text> into newlines after ., ?, ! followed by space or end of string."""
    return re.findall(r'[^.!?]+[.!?]', text)


def clean_transcript(text: str, replacements: dict[str, str], case_sensitive: bool = False) -> str:
    """
    Given the transcript contained at <text>, make the replacements indicated
    in <replacements> such that if the key in replacements appears in <text> then
    it is replaced by the corresponding value.

    This can be used in tandem with transcribe to remove commonly occuring mishearings, such as
    the shirt size 'XS' for 'excess.'

    Set <case_sensitive> to True for case sensitive replacement.

    Example:
    r = {'blue': 'red'}
    t = 'I have a blue car.'
    print(clean_transcript(t, r))
    I have a red car.
    """

    if case_sensitive:
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
    else:
        for wrong, right in replacements.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            text = pattern.sub(right, text)
    return text


def _format_time(seconds: float) -> str:
    """Format the time in <seconds> and return an understandable string."""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02}:{secs:02}"


def _format_srt_time(seconds: float) -> str:
    """Format <seconds> and return a .srt compatible string."""
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"


def _write_timestamped_txt(segments: list[dict], output_path: str,
                          replacements: dict[str, str] = None, case_sensitive: bool = False) -> None:
    """
    Write the <segments> to <output_path.>
    Make all replacements in <replacements> with <case_sensitive> sensitivity to case if provided.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()
            if replacements:
                text = clean_transcript(text, replacements, case_sensitive)
            f.write(f"[{_format_time(start)} - {_format_time(end)}] {text}\n")


def _write_srt(segments: list[dict], output_path: str,
              replacements: dict[str, str] = None, case_sensitive: bool = False) -> None:
    """
    Write the <segments> to <output_path.> as an .srt file.
    Make all replacements in <replacements> with <case_sensitive> sensitivity to case if provided.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = _format_srt_time(seg["start"])
            end = _format_srt_time(seg["end"])
            text = seg["text"].strip()
            if replacements:
                text = clean_transcript(text, replacements, case_sensitive)
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


# == TRANSCRIBER ==
def transcribe(
        relative_path: str,
        transcriber: Whisper,
        detect_language: bool = False,
        replacements: dict[str, str] | None = None,
        write_to_file: bool = False,
        chunk_duration: float = 30.0,
        chunk_overlap: float = 5.0
) -> str:
    """
    Transcribe the audio contained at <relative_path> using OpenAI whisper.
    Return the full, raw, unprocessed result of the model for debugging if needed.
    Note: FFmpeg must be installed and available in your system PATH.
    - <model_level> represents the type of model used: 'tiny,' 'base', 'small', 'medium', 'large', 'turbo'.
    - <detect_language> should be set to True if the model needs to identify the language first; default False.
    - <replacements> contains a dictionary of {'wrong': 'right'} strings for which the key is the erroneous
    transcription and the corresponding value is the correct word for which it should be replaced.
    This uses the clean_transcript function; default None for no replacements
    needed. Note that the replacement applies to the output to the transcription.txt file and not the return value
    of this function.
    - <timestamps>: whether timestamps should be included in the .txt file.
    - <write_to_file> specifies whether the output text should be written to a file. If True, writes to a file
    with the directory <relative_path>_transcription.txt.

    == Preconditions ==
    - <model_level> == 'tiny,' 'base', 'small', 'medium', 'large', 'turbo'.
        'turbo.' is not suited for translation from [lang1] to [lang2].
    - FFmpeg must be installed
    """

    print("\n" * 5 + f"audiotranscriber: Transcribing {relative_path} ...")
    print("Device:", torch.cuda.get_device_name(0))

    # Load full audio.
    audio = whisper.load_audio(relative_path)
    sr = whisper.audio.SAMPLE_RATE

    if detect_language:
        audio_chunk = audio[:int(chunk_duration * sr)]
        mel = whisper.log_mel_spectrogram(audio_chunk).to(transcriber.device)
        _, probs = transcriber.detect_language(mel)
        print(f"audiotranscriber: Detected language: {max(probs, key=probs.get)}")

    # Chunk audio into segments.
    chunk_size = int(chunk_duration * sr)
    overlap_size = int(chunk_overlap * sr)
    step_size = chunk_size - overlap_size
    total_samples = len(audio)

    all_segments = []  # Store time-adjusted stamped segments.

    num_chunks = math.ceil((total_samples - overlap_size) / step_size)

    # Track end time of last kept segment to avoid overlap duplicates.
    last_global_end = 0.0

    for i in range(num_chunks):
        start = i * step_size
        end = min(start + chunk_size, total_samples)
        audio_chunk = audio[start:end]

        if len(audio_chunk) < chunk_size:
            audio_chunk = whisper.pad_or_trim(audio_chunk)

        # Transcribe audio.
        result = transcriber.transcribe(audio_chunk, fp16=True)
        segments = result.get("segments", [])

        # Add offset to timestamps to prevent rollover to 0 on new chunks.
        offset_sec = start / sr

        # Apply timestamps.
        for seg in segments:
            seg_start = seg.get("start", 0.0) + offset_sec
            seg_end = seg.get("end", 0.0) + offset_sec

            # Skip if this is entirely before what we've already kept due to overlap.
            if seg_end <= last_global_end:
                continue

            # Adjust word-level timestamps.
            if "words" in seg and isinstance(seg["words"], list):
                for w in seg["words"]:
                    if "start" in w:
                        w["start"] = w["start"] + offset_sec
                    if "end" in w:
                        w["end"] = w["end"] + offset_sec

            # Write back adjusted starts/ends.
            seg["start"] = seg_start
            seg["end"] = seg_end

            all_segments.append(seg)
            last_global_end = max(last_global_end, seg_end)

        print(f"Chunk {i + 1}/{num_chunks} transcribed.")

    # Combine segments into full text.
    full_text = " ".join(seg["text"].strip() for seg in all_segments)

    # Make replacements.
    if replacements:
        full_text = clean_transcript(full_text, replacements, case_sensitive=False)

    sentences = _split_text_sentences(full_text)
    full_text_with_breaks = "\n".join(sentence.strip() for sentence in sentences)

    # Write timestamped text and SRT using segment data.
    if write_to_file:
        txt_output = f"{relative_path}_transcription.txt"
        srt_output = f"{relative_path}_transcription.srt"

        _write_timestamped_txt(all_segments, txt_output, replacements)
        _write_srt(all_segments, srt_output, replacements)

        print(f"audiotranscriber: transcription written to {txt_output} and {srt_output}")

    return full_text_with_breaks


# == INTERFACE ==
def main() -> None:
    """
    Main execution loop.
    """

    replacements = {'':''}

    transcriber = whisper.load_model('medium', device='cuda')  # Change model settings here.

    files = ['YOUR_FILE_DIRECTORY_HERE.mp4']  # Change this to whatever you need.

    # Change transcriber parameters here.
    for path in files:
        transcribe(
            relative_path=path,
            transcriber=transcriber,
            detect_language=False,
            replacements=replacements,
            write_to_file=True
        )


if __name__ == '__main__':
    main()
