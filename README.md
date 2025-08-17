# ai-tooling-project
An open-source repository containing a variety of QOL AI tools that interface with well-known AI APIs.
Provides a full pipeline to transcribe audio/video media and generate summaries and segment into timestamped chapters for ease of access. 

# Motivation 
The contents of this project were developed to enable business-end users in any field to improve the efficiency of their work. These tools interface with well-known AI APIs and provide an easy-to-use user interface for non-technical users to support their work through a simple and minimalistic user interface. 

# Contents 
- audiotranscriber.py: transcribes audio/video using OpenAI's Whisper API. Helpful for generating things such as meeting notes, summarizing videos for processing with LLMs, or preparing media for word processing locally without infosec concerns. REQUIREMENTS: openAI whisper + ffmpeg (for video decoding, must be in system PATH)
- processor.py (contained in the folder 'processor'): summarizes and auto-chapters those transcripts via open-source LLMs (using Ollama in this case). Customizable Ollama model and prompts. REQUIREMENTS: Ollama installed, requests >= 2.30.0. 

No cloud APIs or uploads required — everything runs locally. Good for sensitive work. 

# General Requirements

- Python ≥ 3.9
audiotranscriber:
- An OpenAI Whisper installation
- ffmpeg for video decoding, must be avaliable in system PATH
processor: 
- Ollama installed 
- requests>=2.30.0
  
