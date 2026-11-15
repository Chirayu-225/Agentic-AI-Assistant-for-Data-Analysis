"""Helpers for test_result_channel. Lives in its own importable module so the
objects pickle by a stable name in both the parent and a spawned worker."""
import os
import pickle


def write_marker(path):
    with open(path, "w") as f:
        f.write("PWNED: code ran inside the parent process")


class Evil:
    """Unpickling this object calls write_marker(path) - arbitrary code execution."""
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (write_marker, (self.path,))


def evil_worker(chan, marker):
    """A COMPROMISED worker: bypasses every guard and sends raw pickle bytes."""
    chan.put_raw(pickle.dumps(Evil(marker)))


def dying_worker(chan):
    os._exit(3)  # killed (out of memory, crash) without sending anything


def good_worker(chan):
    chan.put({"success": True, "error": "", "produced": True, "auto_captured": False, "value": 42})
