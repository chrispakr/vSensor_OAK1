import socket, pickle, struct
import threading
import time
import numpy as np


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
    def __init__(self, host, port, logger, max_listeners=3):
        self.logger = logger
        self.host_ip = host
        self.host_port = port
        self.max_listeners = max_listeners
        self.active_connections = ActiveSocketConnections()
        self._server_socket = None
        self._conn = None
        self._addr = None
        self._t_image = None
        self._client_connected = False
        self._start_transfer = False
        self.s_thread = threading.Thread(target=self._bg_task)
        self.s_thread.daemon = True
        self.s_thread.start()

    def _bg_task(self):
        self.socketOpen()
        while True:
            try:
                if self._start_transfer:
                    # print("transfer image")
                    if self._t_image is not None:
                        start = time.time()
                        a = pickle.dumps(self._t_image)
                        message = struct.pack("Q", len(a)) + a
                        self._conn.sendall(message)
                        self._start_transfer = False
                        self._t_image = None
                        time.sleep(0.04)

            except Exception as e:
                self._log_info(e)
                self._client_connected = False
                self.socketClose()
                time.sleep(1)
                self.s_thread = threading.Thread(target=self._bg_task)
                self.s_thread.start()
                break

    def socketOpen(self):
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host_ip, self.host_port))
        self._server_socket.listen(1)
        self._log_info(u'Server socket [ TCP_IP: ' + self.host_ip + ', TCP_PORT: ' + str(self.host_port) + ' ] is open')
        self._conn, self._addr = self._server_socket.accept()
        self._client_connected = True
        self._log_info(u'Server socket [ TCP_IP: ' + self.host_ip + ', TCP_PORT: ' + str(self.host_port) + ' ] is connected with client')

    def socketClose(self):
        self._server_socket.close()
        self._log_info(u'Server socket [ TCP_IP: ' + self.host_ip + ', TCP_PORT: ' + str(self.host_port) + ' ] is close')

    def send_image(self, image_rear, image_front):
        self._t_image = np.concatenate((image_rear, image_front), axis=1)
        self._start_transfer = True

    @property
    def client_connected(self):
        return self._client_connected

    def _log_info(self, message):
        message = str(message)
        log_message = f"[sHandler]" + " - " + message
        self.logger.info(log_message)