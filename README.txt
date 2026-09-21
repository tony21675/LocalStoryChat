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
