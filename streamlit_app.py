import streamlit as st
import requests
import os
import socket
import time
import json
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse

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

# Function to fetch available models from the LLM server
def fetch_available_models(server_url):
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

if "url_context" not in st.session_state:
    st.session_state.url_context = []

if "available_models" not in st.session_state:
    # Start with an empty dict - will be populated during initialization
    st.session_state.available_models = {}

if "active_model" not in st.session_state:
    st.session_state.active_model = None

if "model_status" not in st.session_state:
    st.session_state.model_status = {}

if "llm_server_url" not in st.session_state:
    st.session_state.llm_server_url = DEFAULT_LLM_SERVER_URL
    # Fetch models during initial load
    models = fetch_available_models(DEFAULT_LLM_SERVER_URL)
    if models:
        st.session_state.available_models = models
        st.session_state.active_model = next(iter(models.keys()), None)
    else:
        # Use fallback models if fetching fails
        st.session_state.available_models = FALLBACK_MODELS
        st.session_state.active_model = next(iter(FALLBACK_MODELS.keys()), None)

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
    
    # Add URL context information
    if st.session_state.url_context:
        url_context = st.session_state.url_context
        system_message += f"You have access to content from {len(url_context)} web pages:\n"
        
        for i, url_data in enumerate(url_context):
            system_message += f"- [{url_data['title']}({url_data['url']})]\n"
        
        system_message += "\nUse these resources to answer questions. You can ask for specific content if needed."
    
    # Update the system message
    st.session_state.messages[0]["content"] = system_message

# Title and description
st.title("Chat with Local LLM")

# Context folder input in the sidebar
with st.sidebar:
    st.header("LLM Server Settings")
    
    # LLM Server URL input
    llm_url = st.text_input("LLM Server URL", value=st.session_state.llm_server_url)
    
    # Update URL if changed
    if llm_url != st.session_state.llm_server_url:
        st.session_state.llm_server_url = llm_url
        st.session_state.model_status = {}  # Reset model status when URL changes
        
        # Fetch available models when URL changes
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
    
    st.header("Model Settings")
    
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
                st.error("Could not discover models")
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
        # Only test the currently selected model
        is_available = test_model_availability(st.session_state.active_model)
        st.session_state.model_status[st.session_state.active_model] = is_available

        # Show current model status
        if is_available:
            model_name = st.session_state.available_models.get(st.session_state.active_model, st.session_state.active_model)
            st.success(f"Using {model_name}")
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
                st.warning(f"{model_name} not available, falling back to {fallback_name}")
                st.session_state.active_model = fallback_model
            else:
                st.error("No LLM models available! Make sure your local LLM server is running.")
    else:
        st.error("No model selected. Check LLM server connection.")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] != "system":  # Don't show system message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Find relevant context
    relevant_urls = find_relevant_urls(prompt)
    context_message = None
    
    # Build context content from URLs
    if relevant_urls:
        context_content = "Here is content that may help answer the question:\n\n"
        
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
                message_placeholder.markdown(f"*Using {st.session_state.available_models[used_model]} instead of {st.session_state.available_models[model_to_use]}*\n\n")
            
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

# Display available models dynamically instead of hardcoded
if st.session_state.available_models:
    model_list = ", ".join([f"`{k}`" for k in st.session_state.available_models.keys()])
    st.sidebar.markdown(f"Available models: {model_list}")
else:
    st.sidebar.markdown("No models discovered. Check server connection.")
