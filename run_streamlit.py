#!/usr/bin/env python3
"""
Helper script to run the Streamlit app without needing the streamlit command
"""
import subprocess
import sys
import os
import platform
import time
import shutil

def check_system_dependencies():
    """Check and install system-level dependencies"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # Check for libmagic which is needed by python-magic
        try:
            # Check if libmagic is already installed 
            result = subprocess.run(["brew", "list", "libmagic"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   check=False)
            
            if result.returncode != 0:
                # Not installed, try to install
                print("Installing libmagic system library...")
                if shutil.which("brew"):
                    subprocess.check_call(["brew", "install", "libmagic"])
                    print("✓ libmagic installed successfully")
                else:
                    print("⚠️ Homebrew not found. Please install libmagic manually:")
                    print("   1. Install Homebrew from https://brew.sh/")
                    print("   2. Run: brew install libmagic")
                    print("   3. Then run this script again")
                    sys.exit(1)
        except Exception as e:
            print(f"⚠️ Error checking for libmagic: {e}")
            print("Please install libmagic manually:")
            print("   1. Install Homebrew from https://brew.sh/")
            print("   2. Run: brew install libmagic")

def check_dependencies():
    """Check and install required dependencies"""
    
    # Check system dependencies first
    check_system_dependencies()
    
    dependencies = [
        "streamlit>=1.22.0",
        "requests>=2.28.2",
        "beautifulsoup4>=4.11.0",
        "pyngrok>=5.1.0",
        "PyPDF2>=2.0.0",
        "python-docx>=0.8.11",
        "pandas>=1.3.5",
        "openpyxl>=3.0.9"
    ]
    
    # Special case for PyMuPDF based on platform
    if platform.system() != "Darwin" or platform.mac_ver()[0] >= '11.0':
        dependencies.append("PyMuPDF>=1.19.0")
    
    # Special case for python-magic based on platform
    if platform.system() == "Windows":
        dependencies.append("python-magic-bin>=0.4.14")
    else:
        dependencies.append("python-magic>=0.4.25")
    
    print("Checking dependencies...")
    for dep in dependencies:
        package = dep.split('>=')[0]
        try:
            __import__(package.lower())
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✓ {package} installed successfully")
    
    print("All dependencies installed!")

def main():
    """Main function to start the streamlit app"""
    # Check dependencies first
    check_dependencies()
    
    print("\nStarting Streamlit app...")
    streamlit_app = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    
    # Get the local IP address to display
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 1))  # connect to Google's DNS
        local_ip = s.getsockname()[0]
        s.close()
        
        print(f"\n💻 App will be available at: http://{local_ip}:8501")
    except:
        print("\n💻 App will be available at: http://localhost:8501")
    
    print("🚀 Launching...")
    
    # Run the Streamlit app
    try:
        subprocess.call([sys.executable, "-m", "streamlit", "run", streamlit_app])
    except KeyboardInterrupt:
        print("\nShutting down the Streamlit app...")
    except Exception as e:
        print(f"\nError starting Streamlit: {e}")
        print("Try running manually: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
