"""Per-server id source: random uuids by default, reproducible counters when deterministic."""
import threading
import uuid


class IdSource:
    def __init__(self, deterministic=False):
        self.deterministic = deterministic
        self._lock = threading.Lock()
        self._n = {"req": 0, "msg": 0, "embed": 0}

    def reset(self):
        with self._lock:
            for k in self._n:
                self._n[k] = 0

    def _next(self, kind):
        with self._lock:
            self._n[kind] += 1
            return self._n[kind]

    def request_id(self):
        return f"00000000-0000-4000-8000-{self._next('req'):012d}" if self.deterministic else str(uuid.uuid4())

    def anon_id(self):
        """For responses outside the request log (control routes, 404s): never advances a counter."""
        return "00000000-0000-4000-8000-000000000000" if self.deterministic else str(uuid.uuid4())

    def msg_id(self):
        return f"msg_mock_{self._next('msg'):016d}" if self.deterministic else "msg_mock_" + uuid.uuid4().hex[:16]

    def embed_id(self):
        return f"00000000-0000-4000-8000-{self._next('embed'):012d}" if self.deterministic else str(uuid.uuid4())
