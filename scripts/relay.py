"""Tiny TCP relay: login-node localhost:<port>  ->  <compute-node>:<port>.

VS Code Remote-SSH reliably forwards *localhost* ports of the machine it is connected to
(the login node), so this makes the teleop server on the GPU node reachable as plain
http://localhost:<port> in your laptop's browser.

    python scripts/relay.py mlgpu017 22874           # usually started by start_teleop.sh

The login node is shared, so if the preferred local port is taken the next free one is used;
the chosen port is printed as "[relay] localhost:<port> -> ...".
"""

import socket
import sys
import threading


def pipe(src: socket.socket, dst: socket.socket):
    try:
        while data := src.recv(65536):
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def main():
    target_host, target_port = sys.argv[1], int(sys.argv[2])
    listen_port = int(sys.argv[3]) if len(sys.argv) > 3 else target_port
    for port in range(listen_port, listen_port + 50):
        srv = socket.socket()
        try:
            srv.bind(("127.0.0.1", port))  # localhost only: not reachable from other machines
            break
        except OSError:
            srv.close()
    else:
        sys.exit(f"[relay] no free port in {listen_port}..{listen_port + 49}")
    listen_port = port
    srv.listen(64)
    print(f"[relay] localhost:{listen_port} -> {target_host}:{target_port}", flush=True)
    while True:
        client, _ = srv.accept()
        try:
            upstream = socket.create_connection((target_host, target_port), timeout=5)
            upstream.settimeout(None)
        except OSError as e:
            print(f"[relay] cannot reach {target_host}:{target_port}: {e}", flush=True)
            client.close()
            continue
        for s in (client, upstream):  # forward small control messages immediately
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for a, b in ((client, upstream), (upstream, client)):
            threading.Thread(target=pipe, args=(a, b), daemon=True).start()


if __name__ == "__main__":
    main()
