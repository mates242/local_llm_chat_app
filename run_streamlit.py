"""
Helper script to run the Streamlit app without needing the streamlit command
"""
import subprocess
import sys
import os

def find_streamlit():
    """Find the streamlit module in the Python environment"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

if __name__ == "__main__":
    if not find_streamlit():
        print("Installing streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("Streamlit installed successfully!")
    
    print("Starting Streamlit app...")
    streamlit_app = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    subprocess.call([sys.executable, "-m", "streamlit.web.cli", "run", streamlit_app])
