import socket
import time


class Socket:
    def __init__(self, host, port):
        self.HOST = host
        self.PORT = port
        self.SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def socket_close(self):
        self.SOCKET.send(b'#')  # to terminate connection
        self.SOCKET.close()  # close socket

    def socket_send(self, data):
        try:
            self.SOCKET.send(data)
        except KeyboardInterrupt:
            self.socket_close()

    def socket_connect(self):
        self.SOCKET.connect((self.HOST, self.PORT))


def create_client_socket(host="localhost", port=5001):
    soc = Socket(host, port)  # initiate socket
    soc.socket_connect()  #
    return soc

