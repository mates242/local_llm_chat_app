import streamlit as st
import requests
import os
import socket
import time
import json
import re
import io
import base64
import tempfile
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Document processing imports
import PyPDF2
import fitz  # PyMuPDF
import docx
import magic  # For file type detection

# Default LLM server URL (now empty to encourage user to input ngrok URL)
DEFAULT_LLM_SERVER_URL = ""

# Configure page
st.set_page_config(
    page_title="Chat with Local LLM",
    page_icon="💬",
    layout="wide"
)

# Function to get local IP address
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# Function to fetch available models from the LLM server
def fetch_available_models(server_url):
    if not server_url:
        return {}
        
    try:
        response = requests.get(
            f"{server_url}/v1/models",
            timeout=5
        )
        
        if response.status_code == 200:
            models_data = response.json()
            # Extract model IDs and create a dictionary
            models_dict = {}
            for model in models_data.get('data', []):
                model_id = model.get('id')
                if model_id:
                    # Use the model ID as both key and display name initially
                    models_dict[model_id] = model_id
                    
                    # Try to create a more friendly name from the model ID
                    friendly_name = model_id.replace('-', ' ').title()
                    # Apply some common replacements for better display
                    friendly_name = friendly_name.replace("Instruct", "Instruct")
                    friendly_name = friendly_name.replace("Chat", "Chat")
                    friendly_name = friendly_name.replace("Llama", "Llama")
                    friendly_name = friendly_name.replace("Llm", "LLM")
                    friendly_name = friendly_name.replace("Gpt", "GPT")
                    friendly_name = friendly_name.replace("Bert", "BERT")
                    
                    models_dict[model_id] = friendly_name
            
            return models_dict
        return {}
    except Exception as e:
        st.sidebar.error(f"Error fetching models: {str(e)}")
        return {}

# Define fallback models in case server doesn't support model listing
FALLBACK_MODELS = {
    "gemma-3-4b-it": "Gemma 3 4B Instruct",
    "qwen2.5-7b-instruct-1m": "Qwen 2.5 7B Instruct"
}

# Function to extract text from PDF file
def extract_text_from_pdf(file_content):
    try:
        # Try using PyMuPDF first (better text extraction)
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception:
        # Fall back to PyPDF2 if PyMuPDF fails
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"

# Function to extract text from DOCX file
def extract_text_from_docx(file_content):
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        return f"Error extracting text from DOCX: {str(e)}"

# Function to extract text from TXT file
def extract_text_from_txt(file_content):
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return file_content.decode('latin-1')
        except Exception as e:
            return f"Error decoding text file: {str(e)}"

# Function to extract text from JSON file
def extract_text_from_json(file_content):
    try:
        json_data = json.loads(file_content.decode('utf-8'))
        # Convert the JSON to a formatted string for better readability
        return json.dumps(json_data, indent=2)
    except Exception as e:
        return f"Error parsing JSON file: {str(e)}"

# Function to extract text from SQL file
def extract_text_from_sql(file_content):
    return extract_text_from_txt(file_content)  # SQL files are just text

# Function to extract text from Excel files
def extract_text_from_excel(file_content):
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        # Convert dataframe to string representation
        return df.to_string()
    except Exception as e:
        return f"Error parsing Excel file: {str(e)}"

# Function to extract text from CSV files
def extract_text_from_csv(file_content):
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        # Convert dataframe to string representation
        return df.to_string()
    except Exception as e:
        return f"Error parsing CSV file: {str(e)}"

# Function to process uploaded file
def process_uploaded_file(uploaded_file):
    try:
        # Read file content
        file_content = uploaded_file.getvalue()
        
        # Detect file type using python-magic
        file_type = magic.from_buffer(file_content, mime=True)
        
        # Process based on detected file type
        if 'pdf' in file_type:
            text = extract_text_from_pdf(file_content)
            file_type = "PDF"
        elif 'word' in file_type or 'docx' in file_type:
            text = extract_text_from_docx(file_content)
            file_type = "DOCX"
        elif 'text/plain' in file_type:
            text = extract_text_from_txt(file_content)
            file_type = "TXT"
        elif 'json' in file_type or uploaded_file.name.endswith('.json'):
            text = extract_text_from_json(file_content)
            file_type = "JSON"
        elif 'sql' in file_type or uploaded_file.name.endswith('.sql'):
            text = extract_text_from_sql(file_content)
            file_type = "SQL"
        elif 'spreadsheet' in file_type or 'excel' in file_type:
            text = extract_text_from_excel(file_content)
            file_type = "Excel"
        elif 'csv' in file_type or uploaded_file.name.endswith('.csv'):
            text = extract_text_from_csv(file_content)
            file_type = "CSV"
        else:
            # Try as text for unknown types
            text = extract_text_from_txt(file_content)
            file_type = "Unknown"
        
        # Limit text length if too large
        max_chars = 100000  # 100K characters max
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[Content truncated. Original size: {len(text)} characters]"
            
        return {
            'filename': uploaded_file.name,
            'type': file_type,
            'content': text,
            'size': len(text)
        }
    except Exception as e:
        return {
            'filename': uploaded_file.name,
            'type': "Error",
            'content': f"Error processing file: {str(e)}",
            'size': 0
        }

# Function to scrape content from a URL
def scrape_url_content(url):
    try:
        # Add http if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Get the domain for display purposes
        domain = urlparse(url).netloc
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the content
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        
        # Get the page title
        title = soup.title.string if soup.title else "No title"
        
        # Extract text from the page
        text = soup.get_text(separator='\n')
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit the text length to avoid excessive content
        max_length = 10000  # Reasonable limit for context
        if len(text) > max_length:
            text = text[:max_length] + "...\n[Content truncated due to length]"
        
        return {
            'url': url,
            'title': title, 
            'domain': domain,
            'content': text,
            'size': len(text)
        }
    except Exception as e:
        raise Exception(f"Failed to scrape URL: {str(e)}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

if "url_context" not in st.session_state:
    st.session_state.url_context = []

if "document_context" not in st.session_state:
    st.session_state.document_context = []

if "available_models" not in st.session_state:
    # Start with an empty dict - will be populated during initialization
    st.session_state.available_models = {}

if "active_model" not in st.session_state:
    st.session_state.active_model = None

if "model_status" not in st.session_state:
    st.session_state.model_status = {}

if "llm_server_url" not in st.session_state:
    st.session_state.llm_server_url = DEFAULT_LLM_SERVER_URL

# Function to update the system message with context information
def update_system_message():
    system_message = "You are a helpful assistant. "
    
    # Add URL context information
    if st.session_state.url_context:
        num_urls = len(st.session_state.url_context)
        system_message += f"You have access to content from {num_urls} web pages. "
        
        for i, url_data in enumerate(st.session_state.url_context):
            system_message += f"[{url_data['title']}({url_data['url']})], "
    
    # Add document context information
    if st.session_state.document_context:
        num_docs = len(st.session_state.document_context)
        system_message += f"You have access to content from {num_docs} documents. "
        
        for i, doc_data in enumerate(st.session_state.document_context):
            system_message += f"[{doc_data['filename']}({doc_data['type']})], "
    
    if st.session_state.url_context or st.session_state.document_context:
        system_message += "\nUse these resources to answer questions. You can ask for specific content if needed."
    
    # Update the system message
    st.session_state.messages[0]["content"] = system_message

# Function to test if a model is available
def test_model_availability(model_name):
    if not st.session_state.llm_server_url:
        return False
        
    try:
        response = requests.post(
            f"{st.session_state.llm_server_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            },
            timeout=3
        )
        return response.status_code == 200
    except:
        return False

# Function to find relevant content from context
def find_relevant_content(query):
    relevant_content = []
    
    # Find relevant URLs
    if st.session_state.url_context:
        query_lower = query.lower()
        keywords = [word for word in query_lower.split() if len(word) > 3]
        keywords = [word.strip(".,?!;:'\"\(\){}[]") for word in keywords]
        
        scored_urls = []
        for url_data in st.session_state.url_context:
            score = 0
            title = url_data["title"].lower()
            content = url_data["content"].lower()
            domain = url_data["domain"].lower()
            
            for keyword in keywords:
                if keyword in title:
                    score += 10
                if keyword in domain:
                    score += 5
                if keyword in content:
                    score += 1
            
            if score > 0:
                scored_urls.append({"data": url_data, "score": score, "type": "url"})
    
    # Find relevant documents
    if st.session_state.document_context:
        query_lower = query.lower()
        keywords = [word for word in query_lower.split() if len(word) > 3]
        keywords = [word.strip(".,?!;:'\"\(\){}[]") for word in keywords]
        
        scored_docs = []
        for doc_data in st.session_state.document_context:
            score = 0
            filename = doc_data["filename"].lower()
            content = doc_data["content"].lower()
            doc_type = doc_data["type"].lower()
            
            for keyword in keywords:
                if keyword in filename:
                    score += 10
                if keyword in content:
                    score += 1
            
            if score > 0:
                scored_docs.append({"data": doc_data, "score": score, "type": "doc"})
    
    # Combine and sort both types of content by relevance score
    all_content = scored_urls + scored_docs if 'scored_urls' in locals() and 'scored_docs' in locals() else \
                 scored_urls if 'scored_urls' in locals() else \
                 scored_docs if 'scored_docs' in locals() else []
                 
    all_content.sort(key=lambda x: x["score"], reverse=True)
    
    # Return the top 3 most relevant pieces of content
    return all_content[:3]

# Function to send message to LLM with model fallback logic
def send_to_llm(messages, model_name):
    # Try the requested model first
    try:
        response = requests.post(
            f"{st.session_state.llm_server_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": False
            },
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            st.session_state.model_status[model_name] = True
            return response.json(), model_name
            
        # If the model failed, mark it as unavailable
        st.session_state.model_status[model_name] = False
    except:
        st.session_state.model_status[model_name] = False
    
    # Try all other models
    for alt_model in st.session_state.available_models.keys():
        if alt_model != model_name:
            try:
                response = requests.post(
                    f"{st.session_state.llm_server_url}/v1/chat/completions",
                    json={
                        "model": alt_model,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": -1,
                        "stream": False
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                
                if response.status_code == 200:
                    st.session_state.model_status[alt_model] = True
                    return response.json(), alt_model
                
                st.session_state.model_status[alt_model] = False
            except:
                st.session_state.model_status[alt_model] = False
    
    # If all models failed, raise error
    raise Exception("No available LLM models found")

# Title and description
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🤖 Chat with Local LLM")
    st.caption("Connect to your locally running LLM through this interface")

with col2:
    st.image("https://github.com/charm-tex/streamlit-jupyterchat/raw/master/assets/app-banner.png", width=130)

# Sidebar Configuration
with st.sidebar:
    # Create tabs for different settings
    tab1, tab2, tab3 = st.tabs(["🔌 Connection", "📄 Context", "ℹ️ Info"])
    
    # Connection Settings Tab
    with tab1:
        st.header("LLM Server Settings")
        
        # LLM Server URL input
        st.info("Enter your local LLM server URL or ngrok URL")
        llm_url = st.text_input(
            "LLM Server URL", 
            value=st.session_state.llm_server_url,
            placeholder="http://your-ngrok-url or http://localhost:1234"
        )
        
        # Update URL if changed
        if llm_url != st.session_state.llm_server_url:
            st.session_state.llm_server_url = llm_url
            st.session_state.model_status = {}  # Reset model status when URL changes
            
            # Fetch available models when URL changes
            if llm_url:  # Only try to fetch if a URL is provided
                with st.spinner("Discovering available models..."):
                    models = fetch_available_models(llm_url)
                    if models:
                        st.session_state.available_models = models
                        st.session_state.active_model = next(iter(models.keys()), None)
                        st.success(f"Found {len(models)} models")
                    else:
                        # Use fallback models if fetching fails
                        st.session_state.available_models = FALLBACK_MODELS
                        st.session_state.active_model = next(iter(FALLBACK_MODELS.keys()), None)
                        st.warning("Could not discover models, using fallback options")
            
            st.rerun()  # Refresh to test new URL
        
        st.divider()
        
        st.subheader("Model Settings")
        
        # Show refresh models button
        if st.button("Refresh Available Models"):
            with st.spinner("Discovering available models..."):
                models = fetch_available_models(st.session_state.llm_server_url)
                if models:
                    st.session_state.available_models = models
                    # Keep current model if it's still available, otherwise select first available
                    if st.session_state.active_model not in models:
                        st.session_state.active_model = next(iter(models.keys()), None)
                    st.success(f"Found {len(models)} models")
                else:
                    if st.session_state.llm_server_url:
                        st.error("Could not discover models. Check your URL.")
                    else:
                        st.error("Please enter a server URL first.")
            st.rerun()
        
        # Model selection - now using the dynamically fetched models
        if st.session_state.available_models:
            selected_model = st.selectbox(
                "Select LLM model",
                options=list(st.session_state.available_models.keys()),
                format_func=lambda x: st.session_state.available_models[x],
                index=list(st.session_state.available_models.keys()).index(st.session_state.active_model) 
                    if st.session_state.active_model in st.session_state.available_models else 0
            )
            
            # Check model availability when selection changes
            if selected_model != st.session_state.active_model:
                st.session_state.active_model = selected_model
        else:
            st.warning("No models available. Check LLM server connection.")
    
    # Context Settings Tab
    with tab2:
        st.header("Context Settings")
        
        # Document Upload Section
        st.subheader("📂 Document Context")
        uploaded_files = st.file_uploader(
            "Upload documents for context",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "json", "sql", "csv", "xls", "xlsx"]
        )
        
        # Process uploaded files
        if uploaded_files:
            # Process newly uploaded files
            for uploaded_file in uploaded_files:
                # Check if file already exists in context
                if not any(d['filename'] == uploaded_file.name for d in st.session_state.document_context):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Process the file
                        doc_data = process_uploaded_file(uploaded_file)
                        # Add to context if processing was successful
                        if doc_data['size'] > 0:
                            st.session_state.document_context.append(doc_data)
                            st.success(f"Added: {doc_data['filename']}")
                        else:
                            st.error(f"Failed to process {uploaded_file.name}")
        
        # Clear documents button
        if st.button("Clear All Documents", key="clear_docs"):
            st.session_state.document_context = []
            st.success("All documents cleared")
        
        # Show documents in context
        if st.session_state.document_context:
            with st.expander(f"Document Context ({len(st.session_state.document_context)} files)"):
                for i, doc_data in enumerate(st.session_state.document_context):
                    st.markdown(f"**{i+1}. {doc_data['filename']}** ({doc_data['type']})  \n"
                               f"Size: {doc_data['size']} characters")
        
        st.divider()
        
        # Web URL Input Section
        st.subheader("🌐 Web Context")
        url_input = st.text_input("Enter URL to scrape for context:")
        url_col1, url_col2 = st.columns([3, 1])
        
        with url_col1:
            url_button = st.button("Add URL", type="primary", key="add_url")
        
        with url_col2:
            if st.button("Clear URLs", key="clear_urls"):
                st.session_state.url_context = []
                st.success("URLs cleared")
        
        # Process URL if button clicked
        if url_button and url_input:
            with st.spinner(f"Scraping content from {url_input}..."):
                try:
                    url_data = scrape_url_content(url_input)
                    
                    # Add to context if not already present
                    if not any(u['url'] == url_data['url'] for u in st.session_state.url_context):
                        st.session_state.url_context.append(url_data)
                        st.success(f"Added: {url_data['title']}")
                    else:
                        st.info(f"URL already in context")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Show URLs in context
        if st.session_state.url_context:
            with st.expander(f"Web Context ({len(st.session_state.url_context)} URLs)"):
                for i, url_data in enumerate(st.session_state.url_context):
                    st.markdown(f"**{i+1}. [{url_data['title']}]({url_data['url']})**  \n"
                               f"Size: {url_data['size']} characters")
    
    # Info Tab
    with tab3:
        st.header("Connection Info")
        local_ip = get_local_ip()
        st.write(f"Your local IP: {local_ip}")
        st.info("If accessing from another device, make sure both are on the same network.")
        
        st.divider()
        
        st.header("About")
        st.markdown("""
        **Chat App Simple** lets you interact with locally running LLMs.
        
        Features:
        - Connect to any OpenAI API compatible endpoint
        - Add web pages as context via URL scraping
        - Upload various document types as context
        - Auto-detect available models
        
        [View Documentation](https://github.com/yourusername/chat_app_simple)
        """)
        
        st.divider()
        
        if st.button("Clear Chat History"):
            st.session_state.messages = [st.session_state.messages[0]]  # Keep system message
            st.rerun()

# Update system message based on current context
update_system_message()

# Model status display
if st.session_state.llm_server_url:
    # Check model availability
    if st.session_state.active_model:
        # Only test the currently selected model
        model_status_container = st.empty()
        
        with st.spinner("Testing LLM connection..."):
            is_available = test_model_availability(st.session_state.active_model)
            st.session_state.model_status[st.session_state.active_model] = is_available

            # Show current model status
            if is_available:
                model_name = st.session_state.available_models.get(st.session_state.active_model, st.session_state.active_model)
                model_status_container.success(f"✓ Connected to {model_name}")
            else:
                # Try to find any available model
                available_models = []
                with st.spinner("Testing available models..."):
                    for model in list(st.session_state.available_models.keys())[:3]:  # Test only first 3 models
                        if test_model_availability(model):
                            available_models.append(model)
                            break
                
                if available_models:
                    fallback_model = available_models[0]
                    model_name = st.session_state.available_models.get(st.session_state.active_model, st.session_state.active_model)
                    fallback_name = st.session_state.available_models.get(fallback_model, fallback_model)
                    model_status_container.warning(f"⚠️ {model_name} not available, falling back to {fallback_name}")
                    st.session_state.active_model = fallback_model
                else:
                    model_status_container.error("❌ No LLM models available! Check your server URL.")
    else:
        st.error("No model selected. Check server connection.")
else:
    st.info("👆 Please enter an LLM server URL in the sidebar to get started.")

# Create a container for displaying chat messages
chat_container = st.container(height=500)

# Display chat messages in the container
with chat_container:
    for message in st.session_state.messages:
        if message["role"] != "system":  # Don't show system message
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if we have a server URL and model before proceeding
    if not st.session_state.llm_server_url:
        with st.chat_message("assistant"):
            st.error("Please set an LLM server URL in the sidebar before chatting.")
        # Add error message to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "⚠️ Please set an LLM server URL in the sidebar before chatting."
        })
    else:
        # Find relevant context
        relevant_content = find_relevant_content(prompt)
        context_message = None
        
        # Build context content from relevant sources
        if relevant_content:
            context_content = "Here is content that may help answer the question:\n\n"
            
            # Add content from URLs and documents
            for item in relevant_content:
                if item["type"] == "url":
                    url_data = item["data"]
                    context_content += f"--- Web Page: {url_data['title']} ({url_data['url']}) ---\n{url_data['content'][:5000]}\n\n"
                else:
                    doc_data = item["data"]
                    context_content += f"--- Document: {doc_data['filename']} ({doc_data['type']}) ---\n{doc_data['content'][:5000]}\n\n"
            
            context_message = {"role": "system", "content": context_content}
        
        # Prepare messages for API call
        api_messages = st.session_state.messages.copy()
        if context_message:
            api_messages.insert(-1, context_message)
        
        # Call the LLM API with fallback logic
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Use the selected model with fallback
                model_to_use = st.session_state.active_model
                
                data, used_model = send_to_llm(api_messages, model_to_use)
                
                # If a different model was used, update the active model
                if used_model != model_to_use:
                    st.session_state.active_model = used_model
                    message_placeholder.markdown(f"*Using {st.session_state.available_models[used_model]} instead of {st.session_state.available_models[model_to_use]}*\n\n")
                
                bot_message = data["choices"][0]["message"]["content"]
                message_placeholder.markdown(bot_message)
                st.session_state.messages.append({"role": "assistant", "content": bot_message})
                    
            except Exception as e:
                error_msg = f"Error connecting to the LLM: {str(e)}\n\nMake sure your LLM server is running at {st.session_state.llm_server_url}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {error_msg}"})

# Display a note about models at the bottom of the page
if st.session_state.available_models:
    st.caption(f"Available models: {', '.join(list(st.session_state.available_models.keys()))}")
