from pyngrok import ngrok

def start_tunnel(token, port=1234):
    """
    Start an ngrok tunnel to the specified port
    
    Args:
        token (str): The ngrok auth token
        port (int): The local port to tunnel to
        
    Returns:
        str: The public URL for the tunnel
    """
    # Set the auth token
    ngrok.set_auth_token(token)
    
    # Connect to the local server
    public_url = ngrok.connect(port, "http")
    
    return str(public_url)

def stop_tunnel(public_url):
    """
    Stop a running ngrok tunnel
    
    Args:
        public_url (str): The public URL to disconnect
    """
    ngrok.disconnect(public_url)
    
if __name__ == "__main__":
    # Example token - you should provide your token when running this script
    token = "2vDYcgQoFqdEvPW81E2pGlb5Agw_56YEj2T4Pk5wcpArSsKMs"
    
    # Start the tunnel
    public_url = start_tunnel(token)
    print(f"Public URL: {public_url}")
    
    # Keep the script running to maintain the tunnel
    print("Tunnel is active! Press Ctrl+C to terminate.")
    try:
        # Keep the process running
        while True:
            pass
    except KeyboardInterrupt:
        # Disconnect the tunnel when script is stopped
        stop_tunnel(public_url)
        print("Tunnel closed")