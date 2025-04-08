#!/usr/bin/env python3
"""
Ngrok tunnel manager for Chat App Simple

This script creates a secure tunnel using ngrok to expose a local server to the internet.
It can be used to make your locally running LLM server or Streamlit app accessible remotely.
"""
import argparse
import sys
import time
import signal
from pyngrok import ngrok, exception

def setup_tunnel(token, port, name=None):
    """
    Set up an ngrok tunnel to the specified local port
    
    Args:
        token (str): Ngrok authentication token
        port (int): Local port to expose
        name (str, optional): Name for the tunnel
    
    Returns:
        str: Public URL of the tunnel
    """
    try:
        # Set auth token
        ngrok.set_auth_token(token)
        
        # Connect to local server
        # Pass parameters directly instead of using options dictionary
        if name:
            public_url = ngrok.connect(port, "http", name=name)
        else:
            public_url = ngrok.connect(port, "http")
        return public_url
    except exception.PyngrokError as e:
        print(f"Error setting up ngrok tunnel: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

def print_tunnel_info(public_url, port):
    """Print information about the active tunnel"""
    print("\n" + "=" * 60)
    print(f"🌐 PUBLIC URL: {public_url}")
    print(f"🔌 FORWARDING TO: localhost:{port}")
    print("=" * 60)
    print("\n📋 COPY THIS URL to access your application from anywhere!")
    print("\n⚠️  If using with the chat app, enter this URL in the 'LLM Server URL' field.")
    print("\n⏱️  Tunnel will remain active until this script is terminated.")
    print("\n❗ Press Ctrl+C to stop the tunnel")
    print("=" * 60 + "\n")

def signal_handler(sig, frame):
    """Handle keyboard interrupt and other signals gracefully"""
    print("\nShutting down tunnel...")
    ngrok.kill()
    print("Tunnel closed. Goodbye!")
    sys.exit(0)

def main():
    """Main function to parse arguments and start the tunnel"""
    parser = argparse.ArgumentParser(
        description='Create an ngrok tunnel to expose a local server to the internet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--token', required=True, 
                        help='Your ngrok authentication token (required)')
    parser.add_argument('--port', type=int, default=1234, 
                        help='Local port to expose')
    parser.add_argument('--name', type=str, 
                        help='Optional name for your tunnel')
    
    args = parser.parse_args()
    
    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start the tunnel
    public_url = setup_tunnel(args.token, args.port, args.name)
    
    # Display tunnel information
    print_tunnel_info(public_url, args.port)
    
    # Keep the process running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()
