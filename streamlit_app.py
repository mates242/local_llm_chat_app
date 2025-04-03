import streamlit as st
import requests
import os
import socket
import time
import json
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import importlib.util
import threading

# Check if remote_access module is available and import it
try:
    import remote_access
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False

# Default LLM server URL
DEFAULT_LLM_SERVER_URL = "http://192.168.68.110:1234"

# Configure page
st.set_page_config(
    page_title="Chat with Local LLM",
    page_icon="💬",
    layout="centered"
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

# Define fallback models in case API doesn't return any
FALLBACK_MODELS = {
    "gemma-3-4b-it": "Gemma 3 4B Instruct",
    "qwen2.5-7b-instruct-1m": "Qwen 2.5 7B Instruct"
}

# Function to fetch available models from the LLM server
def fetch_available_models(server_url):
    models = {}
    try:
        # Try the /v1/models endpoint (standard OpenAI API)
        response = requests.get(f"{server_url}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and isinstance(data["data"], list):
                for model in data["data"]:
                    if "id" in model:
                        # Use the model ID as both key and display name
                        model_id = model["id"]
                        models[model_id] = model.get("name", model_id)
                return models
            
        # If that didn't work, try a completion request with invalid model to get error
        # Some servers return available models in the error message
        response = requests.post(
            f"{server_url}/v1/completions",
            json={"model": "invalid_model_name", "prompt": "test"},
            timeout=3
        )
        error_data = response.json()
        if "error" in error_data and "available models" in str(error_data["error"]).lower():
            error_msg = str(error_data["error"])
            # Try to extract model names from error message
            # This is a heuristic approach and might need adjustment
            model_names = re.findall(r"['\"]([\w\-\d\.]+)['\"]", error_msg)
            if model_names:
                for model_id in model_names:
                    models[model_id] = model_id
                return models
        
        # If we still don't have models, fall back to default list
        return FALLBACK_MODELS
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        return FALLBACK_MODELS

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

if "file_context" not in st.session_state:
    st.session_state.file_context = []

if "url_context" not in st.session_state:
    st.session_state.url_context = []

if "active_model" not in st.session_state:
    st.session_state.active_model = None

if "model_status" not in st.session_state:
    st.session_state.model_status = {}

if "llm_server_url" not in st.session_state:
    st.session_state.llm_server_url = DEFAULT_LLM_SERVER_URL

if "available_models" not in st.session_state:
    st.session_state.available_models = {}

# Ngrok session states
if "ngrok_token" not in st.session_state:
    st.session_state.ngrok_token = ""

if "ngrok_url" not in st.session_state:
    st.session_state.ngrok_url = None

if "ngrok_tunnel_active" not in st.session_state:
    st.session_state.ngrok_tunnel_active = False

if "ngrok_port" not in st.session_state:
    st.session_state.ngrok_port = 1234

# Function to extract port from URL
def extract_port_from_url(url):
    parsed_url = urlparse(url)
    return parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)

# Function to start ngrok tunnel
def start_ngrok_tunnel(token, port):
    if not NGROK_AVAILABLE:
        return None
    
    try:
        # Stop existing tunnel if active
        if st.session_state.ngrok_tunnel_active and st.session_state.ngrok_url:
            remote_access.stop_tunnel(st.session_state.ngrok_url)
        
        # Start new tunnel
        public_url = remote_access.start_tunnel(token, port)
        st.session_state.ngrok_tunnel_active = True
        st.session_state.ngrok_url = public_url
        return public_url
    except Exception as e:
        print(f"Error starting ngrok tunnel: {str(e)}")
        return None

# Function to test if a model is available
def test_model_availability(model_name):
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

# Function to update the system message with context information
def update_system_message():
    system_message = "You are a helpful assistant. "
    
    # Add file context information
    if st.session_state.file_context:
        file_context = st.session_state.file_context
        system_message += f"You have access to the following {len(file_context)} files for context:\n"
        
        # Get folders for display
        folders = set()
        for file in file_context:
            folder = os.path.dirname(file['filename'])
            if folder:
                folders.add(folder)
        
        # Add folder information
        if folders:
            folders = sorted(list(folders))
            system_message += "Subdirectories included: " + ", ".join(folders[:5])
            if len(folders) > 5:
                system_message += f" and {len(folders) - 5} more"
            system_message += "\n\n"
        
        # Add file listing
        system_message += "Files available (sorted by path):\n"
        for i, file in enumerate(file_context):
            if i < 20:  # Only list first 20 files to avoid message size issues
                system_message += f"- {file['filename']} ({file['size']} bytes)\n"
            else:
                remaining = len(file_context) - 20
                system_message += f"... and {remaining} more files\n"
                break
    
    # Add URL context information
    if st.session_state.url_context:
        if st.session_state.file_context:
            system_message += "\n"
        
        url_context = st.session_state.url_context
        system_message += f"You also have access to content from {len(url_context)} web pages:\n"
        
        for i, url_data in enumerate(url_context):
            system_message += f"- [{url_data['title']}]({url_data['url']})\n"
    
    if st.session_state.file_context or st.session_state.url_context:
        system_message += "\nUse these resources to answer questions. You can ask for specific content if needed."
    
    # Update the system message
    st.session_state.messages[0]["content"] = system_message

# Title and description
st.title("Chat with Local LLM")

# Context folder input in the sidebar
with st.sidebar:
    st.header("LLM Server Settings")
    
    # Ngrok Settings Section
    if NGROK_AVAILABLE:
        st.subheader("Remote Access via Ngrok")
        ngrok_token = st.text_input("Ngrok Auth Token", value=st.session_state.ngrok_token, type="password")
        ngrok_port = st.number_input("Local LLM Port", value=st.session_state.ngrok_port, min_value=1, max_value=65535)
        
        # Update token if changed
        if ngrok_token != st.session_state.ngrok_token or ngrok_port != st.session_state.ngrok_port:
            st.session_state.ngrok_token = ngrok_token
            st.session_state.ngrok_port = ngrok_port
            
            # Start tunnel if token provided
            if ngrok_token:
                with st.spinner("Starting ngrok tunnel..."):
                    public_url = start_ngrok_tunnel(ngrok_token, ngrok_port)
                    if public_url:
                        st.success(f"Tunnel created! Public URL: {public_url}")
                        # Update LLM server URL with the ngrok URL
                        st.session_state.llm_server_url = public_url
                        # Fetch models from the new URL
                        st.session_state.available_models = fetch_available_models(public_url)
                    else:
                        st.error("Failed to start ngrok tunnel")

        # Show current tunnel status
        if st.session_state.ngrok_tunnel_active and st.session_state.ngrok_url:
            st.info(f"Active ngrok tunnel: {st.session_state.ngrok_url}")
            if st.button("Stop Tunnel"):
                try:
                    remote_access.stop_tunnel(st.session_state.ngrok_url)
                    st.session_state.ngrok_tunnel_active = False
                    st.session_state.ngrok_url = None
                    st.success("Tunnel stopped")
                except:
                    st.error("Failed to stop tunnel")
    else:
        st.info("To enable remote access, make sure pyngrok is installed: pip install pyngrok")
    
    # LLM Server URL input
    llm_url = st.text_input("LLM Server URL", value=st.session_state.llm_server_url)
    
    # Update URL if changed
    if llm_url != st.session_state.llm_server_url:
        st.session_state.llm_server_url = llm_url
        # Fetch models when URL changes
        with st.spinner("Detecting available models..."):
            st.session_state.available_models = fetch_available_models(llm_url)
        st.session_state.model_status = {}  # Reset model status
        st.rerun()  # Refresh to test new URL
    
    st.header("Model Settings")
    
    # Fetch models if not already done
    if not st.session_state.available_models:
        with st.spinner("Detecting available models..."):
            st.session_state.available_models = fetch_available_models(st.session_state.llm_server_url)
    
    if st.button("Refresh Models"):
        with st.spinner("Detecting available models..."):
            st.session_state.available_models = fetch_available_models(st.session_state.llm_server_url)
    
    # Model selection
    model_options = list(st.session_state.available_models.keys())
    
    if model_options:
        # Set default active model if not already set
        if not st.session_state.active_model or st.session_state.active_model not in model_options:
            st.session_state.active_model = model_options[0]
        
        selected_model = st.selectbox(
            "Select LLM model",
            options=model_options,
            format_func=lambda x: st.session_state.available_models.get(x, x),
            index=model_options.index(st.session_state.active_model)
        )
        
        # Update active model if changed
        if selected_model != st.session_state.active_model:
            st.session_state.active_model = selected_model
    else:
        st.error("No models available. Check your LLM server connection.")
        
    # Model info display
    model_info_container = st.container()
    
    st.header("Context Settings")
    
    # Web URL Input Section
    st.subheader("Web Context")
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
        with st.expander(f"Web Content ({len(st.session_state.url_context)} URLs)"):
            for i, url_data in enumerate(st.session_state.url_context):
                st.markdown(f"**{i+1}. [{url_data['title']}]({url_data['url']})**  \n"
                           f"Size: {url_data['size']} characters")
    
    # Local Folder Section
    st.subheader("File Context")
    folder_path = st.text_input("Enter local folder path for context (includes all subfolders):")
    
    # Add file filtering options
    with st.expander("File Filtering Options"):
        col1, col2 = st.columns(2)
        with col1:
            max_files = st.number_input("Max files to load", min_value=1, value=100)
        with col2:
            max_file_size_kb = st.number_input("Max file size (KB)", min_value=1, value=1000) 
        
        # File extension selection
        default_extensions = ['.sql']
        selected_extensions = st.multiselect(
            "Select file extensions to include",
            options=['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.ts', '.jsx', '.log', 
                    '.yaml', '.yml', '.ini', '.cfg', '.conf', '.sh', '.bat', '.ps1', '.sql'],
            default=default_extensions
        )
    
    # Process local files button
    if st.button("Load Files", type="primary", key="load_files"):
        if not folder_path or not os.path.isdir(folder_path):
            st.error(f"'{folder_path}' is not a valid directory")
        else:
            with st.spinner("Loading files including all subfolders..."):
                # Track stats for progress display
                file_stats = {
                    "processed": 0,
                    "total_chars": 0,
                    "skipped_large": 0,
                    "skipped_ext": 0,
                    "max_chars": 300000,  # 300KB total text limit
                    "folders_processed": set(),
                    "start_time": time.time()
                }
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process files using recursive folder traversal
                context = []
                for root, dirs, files in os.walk(folder_path):
                    # Track folder processing
                    relative_path = os.path.relpath(root, folder_path)
                    if relative_path != ".":
                        file_stats["folders_processed"].add(relative_path)
                    
                    # Update progress message
                    status_text.text(f"Scanning {relative_path}...")
                    
                    # Process each file in this directory
                    for file in files:
                        # Check if we've reached the file limit
                        if len(context) >= max_files:
                            break
                            
                        # Check if this file has an allowed extension
                        if not any(file.lower().endswith(ext) for ext in selected_extensions):
                            file_stats["skipped_ext"] += 1
                            continue
                            
                        file_path = os.path.join(root, file)
                        
                        # Get file size and skip if too large
                        try:
                            file_size_kb = os.path.getsize(file_path) / 1024
                            if file_size_kb > max_file_size_kb:
                                file_stats["skipped_large"] += 1
                                continue
                        except:
                            continue
                        
                        # Try to read the file content
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                                # Skip empty files
                                if not content.strip():
                                    continue
                                
                                # Create relative path for display
                                rel_path = os.path.relpath(file_path, folder_path)
                                
                                # Add to context
                                context.append({
                                    'filename': rel_path, 
                                    'content': content,
                                    'size': len(content)
                                })
                                
                                # Update stats
                                file_stats["processed"] += 1
                                file_stats["total_chars"] += len(content)
                                
                                # Update progress
                                progress_percentage = min(file_stats["total_chars"] / file_stats["max_chars"], 0.99)
                                progress_bar.progress(progress_percentage)
                                status_text.text(f"Loaded {file_stats['processed']} files from {len(file_stats['folders_processed'])} folders...")
                                
                                # Check if we've reached the character limit
                                if file_stats["total_chars"] >= file_stats["max_chars"]:
                                    status_text.text("Reached maximum context size limit")
                                    break
                                    
                        except Exception as e:
                            continue
                    
                    # Check if we need to break the folder traversal
                    if file_stats["total_chars"] >= file_stats["max_chars"] or len(context) >= max_files:
                        break
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Sort by folder/file for better organization
                context.sort(key=lambda x: x['filename'])
                
                # Store in session state
                st.session_state.file_context = context
                
                # Calculate processing time
                processing_time = time.time() - file_stats["start_time"]
                
                # Show results
                st.success(f"Loaded {file_stats['processed']} files from {len(file_stats['folders_processed'])} subfolders ({round(file_stats['total_chars']/1000)}KB) in {processing_time:.2f}s")
                
                if file_stats["skipped_large"] > 0 or file_stats["skipped_ext"] > 0:
                    st.info(f"Skipped {file_stats['skipped_large']} files exceeding size limit and {file_stats['skipped_ext']} files with non-selected extensions")
                
                # Update system message with context
                update_system_message()
                
                # Show some examples of loaded files
                with st.expander("View loaded files"):
                    for i, file in enumerate(context[:10]):  # Show first 10 files
                        st.markdown(f"**{file['filename']}** ({len(file['content'])} chars)")
                        code_type = file['filename'].split('.')[-1] if '.' in file['filename'] else 'text'
                        st.code(file['content'][:500] + "..." if len(file['content']) > 500 else file['content'], language=code_type)
                    
                    if len(context) > 10:
                        st.write(f"... and {len(context) - 10} more files")

    st.header("Connection Info")
    local_ip = get_local_ip()
    st.write(f"Your local IP: {local_ip}")
    st.info("If accessing from another device, make sure both are on the same network.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = [st.session_state.messages[0]]  # Keep system message
        st.rerun()

# Update system message based on current context
update_system_message()

# Display current model status at the top of the sidebar
with model_info_container:
    # Check model availability
    if st.session_state.active_model:
        is_available = test_model_availability(st.session_state.active_model)
        st.session_state.model_status[st.session_state.active_model] = is_available

        # Show current model status
        if st.session_state.model_status.get(st.session_state.active_model, False):
            model_name = st.session_state.available_models.get(st.session_state.active_model, st.session_state.active_model)
            st.success(f"Using {model_name}")
        else:
            st.error(f"Selected model not available. Try another model.")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] != "system":  # Don't show system message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Function to find relevant files in context
def find_relevant_files(query):
    if not st.session_state.file_context:
        return []
    
    query = query.lower()
    keywords = [word for word in query.split() if len(word) > 3]
    keywords = [word.strip(".,?!;:'\"\(\){}[]") for word in keywords]
    
    scored_files = []
    for file in st.session_state.file_context:
        score = 0
        filename = file["filename"].lower()
        content = file["content"].lower()
        
        for keyword in keywords:
            if keyword in filename:
                score += 5
            if keyword in content:
                score += 1
        
        if score > 0:
            scored_files.append({"file": file, "score": score})
    
    scored_files.sort(key=lambda x: x["score"], reverse=True)
    return [item["file"] for item in scored_files[:3]]

# Function to find relevant URLs in context
def find_relevant_urls(query):
    if not st.session_state.url_context:
        return []
    
    query = query.lower()
    keywords = [word for word in query.split() if len(word) > 3]
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
            scored_urls.append({"url_data": url_data, "score": score})
    
    scored_urls.sort(key=lambda x: x["score"], reverse=True)
    return [item["url_data"] for item in scored_urls[:2]]  # Limit to 2 URLs to avoid context overflow

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
    for alt_model in AVAILABLE_MODELS.keys():
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

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Find relevant context
    relevant_files = find_relevant_files(prompt)
    relevant_urls = find_relevant_urls(prompt)
    context_message = None
    
    # Build context content from both files and URLs
    if relevant_files or relevant_urls:
        context_content = "Here is content that may help answer the question:\n\n"
        
        # Add file content
        for file in relevant_files:
            context_content += f"--- File: {file['filename']} ---\n{file['content']}\n\n"
        
        # Add URL content
        for url_data in relevant_urls:
            context_content += f"--- Web Page: {url_data['title']} ({url_data['url']}) ---\n{url_data['content'][:5000]}\n\n"
        
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
                message_placeholder.markdown(f"*Using {AVAILABLE_MODELS[used_model]} instead of {AVAILABLE_MODELS[model_to_use]}*\n\n")
            
            bot_message = data["choices"][0]["message"]["content"]
            message_placeholder.markdown(bot_message)
            st.session_state.messages.append({"role": "assistant", "content": bot_message})
                
        except Exception as e:
            message_placeholder.markdown(f"Error connecting to the LLM: {str(e)}\n\nMake sure your LLM server is running at {st.session_state.llm_server_url}")

# Display a note about how to run the app
st.sidebar.divider()
st.sidebar.markdown("### How to run this app")
st.sidebar.code("streamlit run streamlit_app.py")
st.sidebar.markdown("### LLM Server Info")
st.sidebar.markdown(f"The app expects your LLM server to be running at `{st.session_state.llm_server_url}`")
st.sidebar.markdown("Supported models: " + ", ".join([f"`{k}`" for k in FALLBACK_MODELS.keys()]))

# Display a note about remote access in the sidebar footer
st.sidebar.divider()
st.sidebar.markdown("### Remote Access")
if NGROK_AVAILABLE and st.session_state.ngrok_tunnel_active:
    st.sidebar.success(f"Your app is accessible remotely at: {st.session_state.ngrok_url}")
elif NGROK_AVAILABLE:
    st.sidebar.info("Enter an ngrok token to make your app accessible remotely")
else:
    st.sidebar.warning("Install pyngrok to enable remote access")
