import socket
import select
import socks
import sys

SOCKS_HOST = '127.0.0.1'
SOCKS_PORT = 1080
HTTP_PORT = 8080

def handle(client):
    try:
        data = client.recv(8192)
        if not data:
            return
        # Parse the request to get host and port
        first_line = data.split(b'\r\n')[0].decode()
        parts = first_line.split()
        if len(parts) < 2:
            return
        method, url = parts[0], parts[1]
        if method == 'CONNECT':
            host_port = url.split(':')
            host = host_port[0]
            port = int(host_port[1]) if len(host_port) > 1 else 443
            # Create SOCKS5 connection
            remote = socks.socksocket()
            remote.set_proxy(socks.SOCKS5, SOCKS_HOST, SOCKS_PORT)
            remote.connect((host, port))
            client.send(b'HTTP/1.1 200 Connection Established\r\n\r\n')
        else:
            # For GET/POST, extract host from headers
            lines = data.split(b'\r\n')
            host_header = None
            for line in lines:
                if line.lower().startswith(b'host:'):
                    host_header = line.split(b':')[1].strip().decode()
                    break
            if not host_header:
                return
            host = host_header
            port = 80 if url.startswith('http://') else 443
            remote = socks.socksocket()
            remote.set_proxy(socks.SOCKS5, SOCKS_HOST, SOCKS_PORT)
            remote.connect((host, port))
            # Forward the request
            remote.send(data)
        # Bidirectional copy
        sockets = [client, remote]
        while True:
            rlist, _, _ = select.select(sockets, [], [])
            for sock in rlist:
                other = remote if sock is client else client
                data = sock.recv(8192)
                if not data:
                    return
                other.send(data)
    except Exception as e:
        pass
    finally:
        client.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', HTTP_PORT))
    server.listen(5)
    print(f'HTTP proxy running on 127.0.0.1:{HTTP_PORT} -> SOCKS5 {SOCKS_HOST}:{SOCKS_PORT}')
    while True:
        client, _ = server.accept()
        handle(client)

if __name__ == '__main__':
    main()