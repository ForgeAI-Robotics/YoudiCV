"""Length-framed TCP transport; receiving a partial request never calls handler."""
import json
import socket
import struct
import threading
import time

MAX_TRAJECTORY_BYTES = 2 * 1024 * 1024


def recv_exact(sock, length, deadline=None):
    chunks=[]
    while length:
        if deadline is not None:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('trajectory frame receive deadline')
            sock.settimeout(remaining)
        chunk=sock.recv(min(length,65536))
        if not chunk:raise ConnectionError('incomplete trajectory frame')
        chunks.append(chunk);length-=len(chunk)
    return b''.join(chunks)


def receive_frame(sock, *, timeout_sec=10., max_bytes=MAX_TRAJECTORY_BYTES):
    deadline=time.monotonic()+timeout_sec
    length=struct.unpack('!I',recv_exact(sock,4,deadline))[0]
    if not 0<length<=max_bytes:raise ValueError('trajectory frame exceeds transport bound')
    return recv_exact(sock,length,deadline)


def send_frame(sock, data, *, max_bytes=MAX_TRAJECTORY_BYTES):
    if not 0<len(data)<=max_bytes:raise ValueError('trajectory frame exceeds transport bound')
    sock.sendall(struct.pack('!I',len(data))+data)


class FramedTrajectoryServer:
    def __init__(self, address, allowed_client, handler):
        self.allowed_client=allowed_client;self.handler=handler
        self.stopped=threading.Event();self.slots=threading.BoundedSemaphore(4)
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.sock.bind(address);self.sock.listen(4);self.sock.settimeout(.5)
        self.address=self.sock.getsockname()
        self.thread=threading.Thread(target=self._accept,daemon=True)
        self.thread.start()

    def _accept(self):
        while not self.stopped.is_set():
            try:conn,address=self.sock.accept()
            except socket.timeout:continue
            except OSError:break
            if address[0]!=self.allowed_client or not self.slots.acquire(False):
                conn.close();continue
            threading.Thread(target=self._serve,args=(conn,address),daemon=True).start()

    def _serve(self,conn,address):
        task='unknown'
        try:
            payload=json.loads(receive_frame(conn).decode('utf-8'))
            if not isinstance(payload,dict):raise ValueError('trajectory request must be an object')
            task=str(payload.get('task_id','unknown'))
            response=self.handler(payload,address)
            conn.settimeout(5.)
            send_frame(conn,json.dumps(response,separators=(',',':'),allow_nan=False).encode())
        except Exception as exc:
            try:
                conn.settimeout(1.)
                send_frame(conn,json.dumps(dict(task_id=task,status='failed',error='%s: %s'%(type(exc).__name__,exc))).encode())
            except (OSError,ValueError):pass
        finally:
            conn.close();self.slots.release()

    def close(self):
        self.stopped.set();self.sock.close();self.thread.join(timeout=1.)
