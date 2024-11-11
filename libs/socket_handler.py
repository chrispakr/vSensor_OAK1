import socket, pickle, struct
import threading
import time
import numpy as np
from loguru import logger


class ActiveSocketConnections:
    def __init__(self):
        self._active_connections = []
        self.connected_clients = 0

    def register_new_connection(self, socket_data):
        self._active_connections.append(socket_data)
        self.connected_clients = len(self._active_connections)

    def get_client_list(self):
        return self._active_connections

class SocketHandler:
    def __init__(self, host, port, max_listeners=3):
        self.host_ip = host
        self.host_port = port
        self.max_listeners = max_listeners
        self.active_connections = ActiveSocketConnections()
        self._server_socket = None
        self._conn = None
        self._addr = None
        self._vs_front_process_data = None
        self._vs_rear_process_data = None
        self._client_connected = False
        self._start_transfer = False
        self.s_thread = threading.Thread(target=self._bg_task)
        self.s_thread.daemon = True
        self.s_thread.start()

    def _bg_task(self):
        self.socketOpen()
        logger.debug("Socket thread started")
        while True:
            try:
                if self._vs_front_process_data is not None and self._vs_rear_process_data is not None:
                    # logger.debug("Sending images")
                    socket_data = {
                        "vs_front_data" : self._vs_front_process_data,
                        "vs_rear_data" : self._vs_rear_process_data
                    }
                    a = pickle.dumps(socket_data)
                    message = struct.pack("Q", len(a)) + a
                    self._conn.sendall(message)
                    self._vs_rear_process_data = None
                    self._vs_front_process_data = None

            except Exception as e:
                logger.info(e)
                self._client_connected = False
                self.socketClose()
                time.sleep(3)
                self.s_thread = threading.Thread(target=self._bg_task)
                self.s_thread.start()
                break

            time.sleep(0.04)

    def socketOpen(self):
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host_ip, self.host_port))
        self._server_socket.listen(1)
        logger.info(f"Server socket [ TCP_IP: {self.host_ip}, TCP_PORT: {self.host_port}] is open")
        self._conn, self._addr = self._server_socket.accept()
        self._client_connected = True
        logger.info(f"Server socket [ TCP_IP: {self.host_ip}, TCP_PORT: {self.host_port}] is connected with client")

    def socketClose(self):
        self._server_socket.close()
        logger.info(f"Server socket [ TCP_IP: {self.host_ip}, TCP_PORT: {self.host_port}] is close")

    def send_image(
            self,
            vs_front_slope_data,
            vs_rear_slope_data,
    ):
        self._vs_front_process_data = vs_front_slope_data
        self._vs_rear_process_data = vs_rear_slope_data


    @property
    def client_connected(self):
        return self._client_connected
