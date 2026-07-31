import socket
import pickle
import struct
import threading
import time
import logging
import traceback
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any

# Constants
SOCKET_BACKLOG = 1
SOCKET_RETRY_DELAY = 3
PROCESS_DELAY = 0.04
REUSE_ADDR = 1

@dataclass
class SocketData:
    vs_front_data: Any
    vs_rear_data: Any

class ActiveSocketConnections:
    def __init__(self) -> None:
        self._active_connections: List[Any] = []
        self.connected_clients: int = 0

    def register_new_connection(self, socket_data: Any) -> None:
        self._active_connections.append(socket_data)
        self.connected_clients = len(self._active_connections)

    def get_client_list(self) -> List[Any]:
        return self._active_connections

class SocketHandler:
    def __init__(self, host: str, port: int, max_listeners: int = 3) -> None:
        """Initialize the socket handler with connection parameters."""
        self.logger = logging.getLogger(f"main.{self.__class__.__name__}")
        self.host_ip = host
        self.host_port = port
        self.max_listeners = max_listeners

        self.active_connections = ActiveSocketConnections()
        self._server_socket: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._addr: Optional[Tuple[str, int]] = None
        self._process_data: Optional[SocketData] = None
        self._client_connected: bool = False
        self._lock = threading.Lock()
        self._running = True
        
        self._start_background_thread()

    def _start_background_thread(self) -> None:
        """Start the background processing thread."""
        self.s_thread = threading.Thread(target=self._bg_task)
        self.s_thread.daemon = True
        self.s_thread.start()

    def _process_and_send_data(self) -> None:
        """Process and send data if available."""
        if self._process_data and self._client_connected and self._conn:
            try:
                data = pickle.dumps(self._process_data.__dict__)
                message = struct.pack("Q", len(data)) + data
                self._conn.sendall(message)
                self._process_data = None
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                raise e
            except Exception as e:
                self.logger.error(f"Error sending data: {str(e)}")
                raise e

    def _bg_task(self) -> None:
        """Handle background socket communication."""
        while self._running:
            try:
                if not self._client_connected:
                    self._open_socket()
                    self.logger.debug("Socket thread started")
                
                self._process_and_send_data()
                time.sleep(PROCESS_DELAY)
                
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                self.logger.info(f"Client disconnected: {str(e)}")
                self._handle_connection_error(e)
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                self._handle_connection_error(e)

    def _open_socket(self) -> None:
        """Initialize and open the server socket."""
        try:
            if self._server_socket is None:
                self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, REUSE_ADDR)
                self._server_socket.bind((self.host_ip, self.host_port))
                self.logger.info(f"Server socket [TCP_IP: {self.host_ip}, TCP_PORT: {self.host_port}] is open")
            
            self._server_socket.listen(SOCKET_BACKLOG)
            self._conn, self._addr = self._server_socket.accept()
            self._client_connected = True
            self.logger.info(f"Server socket [TCP_IP: {self.host_ip}, TCP_PORT: {self.host_port}] is connected with client")
        except Exception as e:
            self.logger.error(f"Error opening socket: {str(e)}")
            self._close_socket()
            raise e

    def _close_socket(self) -> None:
        """Close all socket connections safely."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None
            
        self._client_connected = False
        self.logger.info(f"Server socket [TCP_IP: {self.host_ip}, TCP_PORT: {self.host_port}] is closed")

    def send_image(self, vs_front_slope_data: Any, vs_rear_slope_data: Any) -> None:
        """Thread-safe method to send image data."""
        with self._lock:
            self._process_data = SocketData(vs_front_slope_data, vs_rear_slope_data)

    def _handle_connection_error(self, error: Exception) -> None:
        """Handle connection errors and prepare for reconnection."""
        self._close_socket()
        time.sleep(SOCKET_RETRY_DELAY)

    def stop(self) -> None:
        """Stop the socket handler and clean up resources."""
        self._running = False
        self._close_socket()

    def __del__(self) -> None:
        """Clean up resources on object destruction."""
        self.stop()

    @property
    def client_connected(self) -> bool:
        """Get the current connection status."""
        return self._client_connected