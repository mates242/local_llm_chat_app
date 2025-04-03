import argparse
from pyngrok import ngrok

# Set up argument parsing
parser = argparse.ArgumentParser(description='Start ngrok tunnel to local server')
parser.add_argument('--token', required=True, help='Your ngrok authentication token')
parser.add_argument('--port', type=int, default=1234, help='Local port to expose (default: 1234)')
args = parser.parse_args()

# Set auth token from command-line argument
ngrok.set_auth_token(args.token)

# Connect to your local server running on specified port
public_url = ngrok.connect(args.port, "http")
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
