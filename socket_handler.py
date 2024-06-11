import socket, pickle, struct
import threading
import time
import imutils
from turbojpeg import TurboJPEG
import platform


if platform.system() == "Linux":
    # self._log_info_vsensor("set parameters for linux-system")
    jpeg = TurboJPEG()
    showOutput = False

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
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socketOpen()
        self._t_image = None
        self._start_transfer = False
        self.s_thread = threading.Thread(target=self._bg_task)
        self.s_thread.daemon = True
        self.s_thread.start()

    def _bg_task(self):
        while True:
            try:
                # self.client_socket, addr = self.server_socket.accept()
                # if self.client_socket:
                #     self._log_info("Got a connection from {}".format(str(addr)))
                #     self.active_connections.register_new_connection(addr)
                # print(self._start_transfer)
                if self._start_transfer:
                    # print("transfer image")
                    if self._t_image is not None:
                        start = time.time()
                        a = pickle.dumps(self._t_image)
                        message = struct.pack("Q", len(a)) + a
                        self.conn.sendall(message)
                        # print("sendframe with size: ", len(a), len(image_info_jpg), time.time() - start)
                        self._start_transfer = False
                        self._t_image = None
                        time.sleep(0.05)

            except Exception as e:
                self._log_info(e)
                self.socketClose()
                self.socketOpen()
                self.s_thread = threading.Thread(target=self._bg_task)
                self.s_thread.start()

    def socketOpen(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host_ip, self.host_port))
        self.server_socket.listen(self.max_listeners)
        self._log_info(u'Server socket [ TCP_IP: ' + self.host_ip + ', TCP_PORT: ' + str(self.host_port) + ' ] is open')
        self.conn, self.addr = self.server_socket.accept()
        self.active_connections.register_new_connection(self.addr)
        self._log_info(u'Server socket [ TCP_IP: ' + self.host_ip + ', TCP_PORT: ' + str(self.host_port) + ' ] is connected with client')

    def socketClose(self):
        self.server_socket.close()
        self._log_info(u'Server socket [ TCP_IP: ' + self.host_ip + ', TCP_PORT: ' + str(self.host_port) + ' ] is close')

    def send_image(self, image):
        self._t_image = image
        self._start_transfer = True

    def _log_info(self, message):
        message = str(message)
        log_message = f"[sHandler]" + " - " + message
        self.logger.info(log_message)