from websocket_server import ItexaWebSocketServer
from restapi_server import fastApi
import uvicorn
from simulation import Simulation
from fastapi.middleware.cors import CORSMiddleware

if __name__ == "__main__":
    websocketServer = ItexaWebSocketServer(host='0.0.0.0', port=9000)
    websocketServer.start()

    fastApi.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    fastApi.state.simulation = Simulation(websocketServer)
    try:
        uvicorn.run(fastApi, host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    finally:
        websocketServer.stop()

    print("Server has been shut down successfully")