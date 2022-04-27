"""
Tools for pretty printing etc.
"""
# pylint: disable=missing-docstring
# pylint: disable=invalid-name

import sys
import contextlib
from datetime import datetime, timedelta, timezone
import struct


def timing_notification(title=""):
    def timer(wrapped):
        def wrapper(*args, **kwargs):
            start = datetime.now()
            print(f"Started {title}")
            ret = wrapped(*args, **kwargs)
            duration = str(datetime.now() - start).split(".")[0]
            print(f"Finished {title} in {duration}")
            return ret

        return wrapper

    return timer


class DummyFile(object):
    # pylint: disable=too-few-public-methods
    def write(self, x):
        pass


@contextlib.contextmanager
def silence():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def progress_bar(it, suffix="", total=None, size=25, fill="▇", empty="░", silent=False):
    # pylint: disable=too-many-arguments
    if silent:
        for item in it:
            yield item
        return
    start_time = datetime.now()
    if total is None:
        total = len(it)
    if total == 0:
        return

    def display(i):
        x = i / total
        prog = fill * int(x * size) + empty * (size - int(x * size))
        duration = datetime.now() - start_time
        statement = f"\r▕{prog}▏ {100*x:.1f} % {suffix} ({str(duration).split('.')[0]}"
        eta = duration * (1 / max(x, 1e-5) - 1)
        if eta > timedelta(days=1):
            statement += " / ETA: > 1 day"
        elif eta > timedelta(seconds=3):
            statement += f" / ETA: {str(eta).split('.')[0]}"
        statement += ")"
        return statement

    for i, item in enumerate(it):
        if i % (total / 1000) <= 1:  # prevent output flooding
            print("\033[K" + display(i), end="\r")
        yield item
    print("\033[K" + display(total))


def parse_cp1252_until_null(fd, debug=False):
    buffer = b""
    while b"\x00" not in buffer:
        buffer += fd.read(1)
    if debug:
        print("\t" + str(buffer))
    try:
        return buffer[:-1].decode("cp1252")
    except UnicodeDecodeError:
        val = buffer[:-1].decode("cp1252", errors="ignore")
        print(f"WARNING: Decoding error in string '{val}'")
        return val


def parse_bytes(buffer, pos, encoding):
    if encoding == "short":
        return struct.unpack("<h", buffer[pos : pos + 2])[0]
    elif encoding == "float":
        return struct.unpack("<f", buffer[pos : pos + 4])[0]
    elif encoding == "time":
        epoch_start = datetime(
            year=1601, month=1, day=1, tzinfo=timezone.utc
        )  # WIN epoch
        win_timestamp = (
            struct.unpack("<Q", buffer[pos : pos + 8])[0] / 1e7
        )  # convert 100ns -> s
        utc_time = epoch_start + timedelta(seconds=win_timestamp)
        return utc_time.timestamp()
    elif encoding == "bool":
        return struct.unpack("<?", buffer[pos : pos + 1])[0]
    else:
        return buffer[pos:].split(b"\x00")[0].decode(encoding)
