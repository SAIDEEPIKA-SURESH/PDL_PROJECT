import os
import socket
import threading
import requests
from flask import Flask, request, jsonify

# === TRACKER SERVER (CENTRAL INDEX) ===
tracker = Flask(__name__)
files_index = {}  # {filename: [peer1, peer2]}

@tracker.route("/")
def home():
    return "Tracker is running!"

@tracker.route("/favicon.ico")
def favicon():
    return "", 204  # No content response

@tracker.route("/register", methods=["POST"])
def register_file():
    data = request.json
    filename = data["filename"]
    peer_ip = data["peer_ip"]
    
    if filename not in files_index:
        files_index[filename] = []
    files_index[filename].append(peer_ip)
    return jsonify({"message": "File registered successfully!"})

@tracker.route("/search/<filename>", methods=["GET"])
def search_file(filename):
    peers = files_index.get(filename, [])
    return jsonify({"peers": peers})

@tracker.errorhandler(404)
def not_found(error):
    print("404 Error: Page not found")
    return "404 Error: Page not found", 404

# Start tracker server
if __name__ == "__main__":
    tracker.run(host="0.0.0.0", port=5000)


# === PEER NODE ===
def start_peer_server(port):
    peer_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    peer_server.bind(("0.0.0.0", port))
    peer_server.listen(5)
    print(f"Peer listening on port {port}...")
    
    def handle_client(conn, addr):
        filename = conn.recv(1024).decode()
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                conn.sendall(f.read())
        conn.close()
    
    while True:
        conn, addr = peer_server.accept()
        threading.Thread(target=handle_client, args=(conn, addr)).start()

def register_with_tracker(filename, peer_ip):
    url = "http://127.0.0.1:5000/register"
    data = {"filename": filename, "peer_ip": peer_ip}
    requests.post(url, json=data)
    print("Registered file with tracker.")

def search_file(filename):
    url = f"http://127.0.0.1:5000/search/{filename}"
    response = requests.get(url).json()
    return response.get("peers", [])

def download_file(filename, peer_ip):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((peer_ip, 9000))
        s.sendall(filename.encode())
        with open(f"downloaded_{filename}", "wb") as f:
            while chunk := s.recv(1024):
                f.write(chunk)
    print("File downloaded successfully.")

# Example usage:
if __name__ == "__main__":
    peer_ip = "127.0.0.1"
    port = 9000
    threading.Thread(target=start_peer_server, args=(port,)).start()
    
    while True:
        cmd = input("Enter command (upload/search/download/exit): ")
        if cmd == "upload":
            filename = input("Enter filename: ")
            register_with_tracker(filename, peer_ip)
        elif cmd == "search":
            filename = input("Enter filename to search: ")
            peers = search_file(filename)
            print("Peers with file:", peers)
        elif cmd == "download":
            filename = input("Enter filename to download: ")
            peers = search_file(filename)
            if peers:
                download_file(filename, peers[0])
            else:
                print("File not found.")
        elif cmd == "exit":
            break