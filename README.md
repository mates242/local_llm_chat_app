# Chat App Simple

A straightforward application for chatting with locally running LLMs (Large Language Models), with support for context from documents and web pages.

## Features

- 🤖 Connect to any OpenAI API-compatible local LLM server
- 🌐 Access your local LLM from anywhere using ngrok tunneling
- 📄 Upload various document types for context-aware conversations
- 🔗 Add web page content as context through URL scraping
- 📱 Responsive design for desktop and mobile use
- 🧠 Automatic model discovery and fallback

## Supported Document Formats

- PDF documents
- Text files (.txt)
- Word documents (.docx)
- JSON files
- SQL files
- CSV data files
- Excel spreadsheets (.xls, .xlsx)

## Requirements

- Python 3.8+
- A locally running LLM server with OpenAI API compatibility
- ngrok account (for remote access)

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or just run the application script which will install dependencies automatically:

```bash
python run_streamlit.py
```

## Usage

### Running the Streamlit App

```bash
python run_streamlit.py
```

or

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501` by default.

### Setting up Remote Access with ngrok

1. Create a free account at [ngrok.com](https://ngrok.com) and get your auth token
2. Run the remote access script to expose your local LLM server:

```bash
python remote_access.py --token YOUR_NGROK_TOKEN --port YOUR_LLM_SERVER_PORT
```

For example, if your LLM server is running on port 1234:

```bash
python remote_access.py --token abc123def456 --port 1234
```

3. Copy the generated ngrok URL and use it in the Chat App's LLM Server URL field

### Using the Chat App

1. **Enter your LLM Server URL** in the sidebar (your local LLM server URL or ngrok URL)
2. **Select a model** from the available models detected
3. **Add context** (optional):
   - Upload documents in the Context tab
   - Add web pages by entering URLs
4. Start chatting with your LLM!

## Example Setups

### Local LLM Server Options

- [Ollama](https://ollama.com/)
- [LocalAI](https://localai.io/)
- [llama.cpp server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)
- [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui) with OpenAI API extension

### Example Server URL Formats

- Local Ollama: `http://localhost:11434/v1`
- Local LM Studio: `http://localhost:1234/v1`
- ngrok URL: `https://abcd-123-45-678-90.ngrok-free.app`

## Troubleshooting

- **No models found**: Ensure your LLM server is running and the URL is correct
- **Connection error**: Check if your LLM server supports the OpenAI API format
- **Document upload fails**: Ensure you have all dependencies installed
- **ngrok tunnel fails**: Verify your auth token and ensure you have an active internet connection

## Remote Access Details

The `remote_access.py` script creates a secure tunnel to your local server using ngrok:

```
python remote_access.py --help
```

Options:
- `--token`: Your ngrok authentication token (required)
- `--port`: Local port to expose (default: 1234)
- `--name`: Optional name for your tunnel

Once running, the script will display the public URL that can be used to access your local LLM server from anywhere.

## License

MIT

## Credits

- Built with [Streamlit](https://streamlit.io/)
- Tunneling by [pyngrok](https://pyngrok.readthedocs.io/)
- Document processing with PyPDF2, PyMuPDF, and python-docx
- Web scraping with BeautifulSoup