from pyngrok import ngrok

# Set your authtoken (from dashboard)
ngrok.set_auth_token("2vDYcgQoFqdEvPW81E2pGlb5Agw_56YEj2T4Pk5wcpArSsKMs")  # Replace with your actual token

# Connect to your local server running on port 1234
public_url = ngrok.connect(1234, "http")
print(f"Public URL: {public_url}")

# Keep the script running to maintain the tunnel
print("Tunnel is active! Press Ctrl+C to terminate.")
try:
    # Keep the process running
    while True:
        pass
except KeyboardInterrupt:
    # Disconnect the tunnel when script is stopped
    ngrok.disconnect(public_url)
    print("Tunnel closed")