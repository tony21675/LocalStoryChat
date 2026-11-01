Local Story Chat v7

A local browser UI for Dolphin 3.0 Llama 3.1 8B Q4_K_M using the existing llama-cli Vulkan backend.

v7 uses normal stdin/stdout pipes with llama-cli --simple-io instead of a pseudo-terminal. This avoids PTY I/O errors and prompt-detection hangs.

Requirements:
- Linux
- Python 3
- Existing Uncensored Local Studio install and Dolphin GGUF at the paths in server.py
- Story files in ~/StoryFiles/

Start:
  ./start.sh
Then the default browser opens automatically to http://127.0.0.1:8787

Desktop launcher:
  ./install_desktop.sh

Configuration:
- Vulkan
- 10 GPU layers
- 4096 context
- Dolphin 3.0 Llama 3.1 8B Q4_K_M
- Persistent llama-cli conversation

## Future independent applications

The project plans two separate local applications, currently on hold until StoryBuilder Writer and Story Bible are finished. See [FUTURE_APPS_ON_HOLD.md](FUTURE_APPS_ON_HOLD.md) for the shared scope and restart checklist. The existing LocalStoryChat v7 app is unchanged; it is reference material for the future generic local chat app, not a decision to retrofit this story-oriented app.
