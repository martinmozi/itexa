import threading
import socket
from simple_websocket_server import WebSocketServer, WebSocket


class SimpleClient(WebSocket):
    def handle(self):
        # Simply handle incoming messages - not needed for your use case
        pass

    def connected(self):
        # Add this client to the server's client list
        self.server.clients.add(self)
        print(f"Client connected. Total clients: {len(self.server.clients)}")

    def handle_close(self):
        # Remove this client from the server's client list
        self.server.clients.remove(self)
        print(f"Client disconnected. Total clients: {len(self.server.clients)}")


class ItexaWebSocketServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self.running = False

    def start(self):
        """Start the WebSocket server in a separate thread"""
        if self.running:
            print("Server is already running")
            return

        # Create a server instance with built-in client tracking
        class ServerWithClients(WebSocketServer):
            def __init__(self, host, port):
                super().__init__(host, port, SimpleClient)
                # Enable socket reuse to prevent "Address already in use" errors
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.clients = set()

        self.server = ServerWithClients(self.host, self.port)

        # Start the server in a separate thread
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        self.running = True
        print(f"WebSocket server started at ws://{self.host}:{self.port}")

    def stop(self):
        """Stop the WebSocket server"""
        if not self.running:
            print("Server is not running")
            return

        if self.server:
            try:
                # For simple_websocket_server, we need to close the server differently
                # First, notify all clients to close their connections
                for client in list(self.server.clients):
                    client.close()

                # Access the internal socket using the server's private attributes
                if hasattr(self.server, 'socket'):
                    self.server.socket.close()
                elif hasattr(self.server, '_socket'):
                    self.server._socket.close()
                elif hasattr(self.server, 'sock'):
                    self.server.sock.close()

                # Set running to false and wait for thread to terminate
                self.running = False
                if self.thread and self.thread.is_alive():
                    self.thread.join(timeout=5)

                print("WebSocket server stopped")
            except Exception as e:
                print(f"Error stopping server: {e}")

    def send_data(self, data):
        """Send data to all connected clients"""
        if not self.running:
            print("Server is not running")
            return False

        # Get the current set of clients
        clients = self.server.clients.copy()
        if not clients:
            print("No clients connected")
            return False

        # Send to all clients
        success_count = 0
        for client in clients:
            try:
                client.send_message(data)
                success_count += 1
            except Exception as e:
                print(f"Error sending message: {e}")

        # print(f"Sent message to {success_count} of {len(clients)} client(s)")
        return success_count > 0
