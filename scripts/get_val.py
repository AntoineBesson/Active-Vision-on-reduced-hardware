#!/usr/bin/env python3
"""Download only the Cityscapes *val* files needed for Cityscapes-VPS.

Cityscapes ships each package as one big zip (leftImg8bit_sequence is ~324 GB).
Every Cityscapes-VPS video is a Cityscapes val sequence, so only the ``val/``
members are needed. This reads the zips remotely with HTTP range requests
(``remotezip``) and extracts just those members:

    leftImg8bit_sequence/val/   packageID 14   (500 sequences x 30 frames, ~33 GB)
    gtFine/val/                 packageID 1    (~0.1 GB)

    pip install remotezip requests
    python scripts/get_val.py --out D:/cityscapes            # both packages
    python scripts/get_val.py --out D:/cityscapes --only gtFine

Credentials come from $CITYSCAPES_USERNAME / $CITYSCAPES_PASSWORD, else are
prompted for. The download is resumable: a file appears under --out only once
it is complete, and re-running skips what is already there. Network errors
are retried (re-logging in each time); a failed login stops immediately.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import time

import requests
from remotezip import RemoteZip

LOGIN_URL = "https://www.cityscapes-dataset.com/login/"
PACKAGE_URL = "https://www.cityscapes-dataset.com/file-handling/?packageID={}"

# name -> (packageID, member prefix to keep)
PACKAGES = {
    "gtFine": (1, "gtFine/val/"),
    "leftImg8bit_sequence": (14, "leftImg8bit_sequence/val/"),
}


class LoginError(RuntimeError):
    pass


def login(user: str, password: str) -> dict:
    s = requests.Session()
    r = s.post(LOGIN_URL, data={"username": user, "password": password, "submit": "Login"},
               allow_redirects=False, timeout=60)
    cookies = s.cookies.get_dict()
    # A successful login answers with a redirect and a session cookie; a wrong
    # password re-renders the login form (200).
    if r.status_code not in (301, 302, 303) or not cookies:
        raise LoginError(f"login failed (HTTP {r.status_code}); check the username/password")
    return cookies


def download_package(name: str, out: str, user: str, password: str, retry_s: int) -> None:
    package_id, prefix = PACKAGES[name]
    tmp_dir = os.path.join(out, "_tmp")
    while True:
        try:
            with RemoteZip(PACKAGE_URL.format(package_id), cookies=login(user, password)) as z:
                files = [n for n in z.namelist() if n.startswith(prefix) and not n.endswith("/")]
                if not files:
                    raise RuntimeError(f"no member under {prefix!r} in package {package_id}")
                todo = [n for n in files if not os.path.exists(os.path.join(out, n))]
                print(f"[{name}] {len(files) - len(todo)}/{len(files)} files already done")
                for i, n in enumerate(todo):
                    tmp = z.extract(n, tmp_dir)
                    final = os.path.join(out, n)
                    os.makedirs(os.path.dirname(final), exist_ok=True)
                    os.replace(tmp, final)  # a file only appears once it is complete
                    if i % 200 == 0:
                        print(f"[{name}] {i}/{len(todo)}  {n}", flush=True)
            break
        except LoginError:
            raise
        except Exception as e:  # network hiccups, expired session, ...
            print(f"[{name}] error: {type(e).__name__}: {e} -> retrying in {retry_s} s", flush=True)
            time.sleep(retry_s)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[{name}] done")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Output folder (gets leftImg8bit_sequence/val/ and gtFine/val/).")
    p.add_argument("--only", choices=sorted(PACKAGES), action="append",
                   help="Download only this package (repeatable). Default: all.")
    p.add_argument("--retry-seconds", type=int, default=60)
    args = p.parse_args()

    user = os.environ.get("CITYSCAPES_USERNAME") or input("Cityscapes username: ")
    password = os.environ.get("CITYSCAPES_PASSWORD") or getpass.getpass("Cityscapes password: ")
    login(user, password)  # fail fast before touching anything
    os.makedirs(args.out, exist_ok=True)
    # gtFine first: it is small, so a broken setup shows up in seconds.
    for name in [n for n in ("gtFine", "leftImg8bit_sequence") if not args.only or n in args.only]:
        download_package(name, args.out, user, password, args.retry_seconds)


if __name__ == "__main__":
    main()
