from collections import defaultdict, namedtuple
from concurrent.futures import Future
from threading import Lock


class AtomicCounter:
    def __init__(self, initial=0):
        self.value = initial
        self._lock = Lock()
        self._thresholds = defaultdict(list)

    def increment(self, num=1):
        with self._lock:
            self.value += num
            for threshold in sorted(self._thresholds.keys()):
                if threshold <= self.value:
                    for future in self._thresholds.pop(threshold):
                        future.set_result(True)
            return self.value

    def get(self):
        with self._lock:
            return self.value

    def reset(self):
        with self._lock:
            self.value = 0

    def wait_until_reaches(self, value, timeout=None):
        future = Future()
        self._thresholds[value].append(future)
        return future.result(timeout=timeout)


class SharedStorage:
    Snapshot = namedtuple("SharedStorageSnapshot", ["data", "version"])

    def __init__(self, data=None, initial_version=0):
        self.data = data
        self._version = AtomicCounter(initial_version)
        self._lock = Lock()

    def set(self, data):
        with self._lock:
            self.data = data
            self._version.increment()

    def get(self):
        with self._lock:
            return self.Snapshot(self.data, self._version.get())

    def update(self, current: Snapshot):
        if current.version < self._version.get():
            return self.get()
        else:
            return current
