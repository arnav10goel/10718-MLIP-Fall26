"""Loopback media and browser transport for one running monitoring session."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import ipaddress
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import stat


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


class MonitorServer(ThreadingHTTPServer):
    """Expose fixed media paths and a session; never dispatch provider calls here.

    The session owns scheduling, identity, sequence and acknowledgment validation.
    Media are read-only regular files checked against their startup identity.
    Playback does not expose local paths or permit filesystem selection over HTTP.
    """

    daemon_threads = True

    def __init__(self, address, *, session, reference_path, execution_path,
                 viewer_path=None, backend="native-video"):
        try:
            local = address[0] == "localhost" or ipaddress.ip_address(address[0]).is_loopback
        except ValueError:
            local = False
        if not local:
            raise ValueError("MonitorServer must bind a loopback interface")
        if backend not in ("mock", "native-video"):
            raise ValueError("Unsupported monitoring backend")
        if any(not callable(getattr(session, name, None))
               for name in ("snapshot", "playback", "acknowledge", "dismiss")):
            raise ValueError("Expected a monitoring session")
        self.session, self.backend = session, backend
        page = Path(viewer_path) if viewer_path is not None else Path(__file__).with_name("monitor_viewer.html")
        self._html = page.read_bytes()
        self.viewer_sha256 = hashlib.sha256(self._html).hexdigest()
        bind_viewer = getattr(session, "bind_viewer_bytes", None)
        if callable(bind_viewer):
            bind_viewer(self._html)
        self._media = {}
        for role, value in (("reference", reference_path), ("execution", execution_path)):
            path = Path(value).expanduser().resolve(strict=True)
            with path.open("rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                    raise ValueError("Monitoring media must be nonempty regular files")
                signature = _signature(info)
                if _signature(path.stat()) != signature:
                    raise ValueError("Monitoring media changed during startup")
            self._media["/media/" + role] = {
                "path": path, "signature": signature, "size": info.st_size,
                "mime": mimetypes.guess_type(path.name)[0] or "video/mp4"}
        super().__init__(address, _MonitorHandler)

    def snapshot(self):
        return {**self.session.snapshot(), "backend": self.backend,
                "reference_url": "/media/reference", "execution_url": "/media/execution"}


class _MonitorHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _local(self):
        port = self.server.server_address[1]
        hosts = {f"{self.server.server_address[0]}:{port}", f"127.0.0.1:{port}", f"localhost:{port}"}
        host, origin = self.headers.get("Host"), self.headers.get("Origin")
        return (host in hosts and (origin is None or origin == "http://" + host)
                and self.headers.get("Sec-Fetch-Site") != "cross-site")

    def _respond(self, status, body=b"", content_type="application/json", extra=None):
        if content_type == "application/json":
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    def do_GET(self):
        if not self._local():
            return self._respond(403, {"error": "Local monitoring origin required"})
        if self.path == "/":
            return self._respond(200, self.server._html, "text/html; charset=utf-8")
        if self.path == "/api/session":
            return self._respond(200, self.server.snapshot())
        if self.path in self.server._media:
            return self._serve_media(self.server._media[self.path])
        self._respond(404, {"error": "Not found"})

    do_HEAD = do_GET

    def _serve_media(self, media):
        try:
            descriptor = os.open(media["path"], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            stream = os.fdopen(descriptor, "rb")
        except OSError:
            return self._respond(409, {"error": "Media changed or unavailable"})
        with stream:
            def current():
                return (_signature(os.fstat(stream.fileno())) == media["signature"]
                        and _signature(media["path"].stat()) == media["signature"])
            try:
                if not current():
                    raise ValueError
            except (OSError, ValueError):
                return self._respond(409, {"error": "Media changed or unavailable"})
            size = media["size"]
            start, end, status = 0, size - 1, 200
            requested = self.headers.get("Range")
            if requested is not None:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                try:
                    if match is None or not any(match.groups()):
                        raise ValueError
                    first, last = match.groups()
                    if first:
                        start, end = int(first), min(int(last), end) if last else end
                    else:
                        suffix = int(last)
                        if suffix <= 0:
                            raise ValueError
                        start = max(0, size - suffix)
                    if start >= size or end < start:
                        raise ValueError
                except ValueError:
                    return self._respond(416, {"error": "Invalid byte range"},
                                         extra={"Content-Range": f"bytes */{size}"})
                status = 206
            self.send_response(status)
            self.send_header("Content-Type", media["mime"])
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            stream.seek(start)
            remaining = end - start + 1
            try:
                while remaining:
                    if not current():
                        raise ValueError
                    chunk = stream.read(min(256 * 1024, remaining))
                    if not chunk:
                        raise ValueError
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (OSError, ValueError):
                self.close_connection = True

    def do_POST(self):
        if not self._local():
            return self._respond(403, {"error": "Local monitoring origin required"})
        handler = {"/api/playback": self.server.session.playback,
                   "/api/ack": self.server.session.acknowledge,
                   "/api/dismiss": self.server.session.dismiss}.get(self.path)
        if handler is None:
            return self._respond(404, {"error": "Not found"})
        try:
            length = self.headers.get("Content-Length", "")
            if (not length.isdecimal() or not 0 < int(length) <= 8192
                    or self.headers.get("Transfer-Encoding") is not None
                    or self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json"):
                raise ValueError

            def pairs(items):
                result = {}
                for name, value in items:
                    if name in result:
                        raise ValueError
                    result[name] = value
                return result

            def number(value):
                parsed = float(value)
                if not math.isfinite(parsed):
                    raise ValueError
                return parsed

            def reject(value):
                raise ValueError

            body = json.loads(self.rfile.read(int(length)), object_pairs_hook=pairs,
                              parse_constant=reject, parse_float=number)
            handler(body)
        except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
            return self._respond(400, {"error": "Invalid session update"})
        except RuntimeError:
            return self._respond(409, {"error": "Session update is no longer applicable"})
        self._respond(200, self.server.snapshot())

    def _unsupported(self):
        self._respond(405, {"error": "Method not allowed"}, extra={"Allow": "GET, HEAD, POST"})

    do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_TRACE = do_CONNECT = _unsupported
