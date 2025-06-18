# PDF to MP3 — Audiobook Converter

## Showcase
![showcase](resources/showcase.gif)

## Setup environment

**Disclaimer: tested with Python 3.10.16**

* `apt-get install build-essential python-tk python3-tk tk-dev  zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev libncurses-dev`
* `pyenv install 3.10.16`
* `pyenv global 3.10.16`

## Steps to run it

Check [VSCode Tasks](.vscode/tasks.json) for up-to-date examples.

* Setup Python virtual env with required dependencies: `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
* Convert multiple PDF files (e.g. chapters of a book) from a folder: `python3 pdf_to_mp3.py -d pdf/`
* Convert a single PDF file: `python3 pdf_to_mp3.py -f pdf/file.pdf`

## Licensing Notice

This project is licensed under the MIT License.

However, it uses [TTS](https://github.com/coqui-ai/TTS) and [XTTSv2](https://huggingface.co/coqui/XTTS-v2), which is subject to a different license:
- XTTSv2 is free for personal, non-commercial use.
- For commercial use, a commercial license from Coqui is required.

You are responsible for ensuring your use of XTTSv2 complies with its licensing terms.