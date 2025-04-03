from websocket_server import ItexaWebSocketServer
from restapi_server import fastApi
import sys
import signal
import threading
import uvicorn
from simulation import Simulation

if __name__ == "__main__":
    terminateEvent = threading.Event()
    websocketServer = ItexaWebSocketServer(host='0.0.0.0', port=9000)
    websocketServer.start()

    def signal_handler(sig, frame):
        print("SIGTERM received, shutting down...")
        websocketServer.stop()
        terminateEvent.set()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    fastApi.state.simulation = Simulation(websocketServer)
    uvicorn.run(fastApi, host="0.0.0.0", port=8080)

    try:
        terminateEvent.wait()
    except KeyboardInterrupt:
        # Extra safety in case the signal handler doesn't catch it
        print("Keyboard interrupt received, shutting down...")
        websocketServer.stop()

    print("Server has been shut down successfully")
