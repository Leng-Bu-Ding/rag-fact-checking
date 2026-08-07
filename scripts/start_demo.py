from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / ".tmp"


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.5)
        return connection.connect_ex((host, port)) == 0


def _status_is_ready(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/health", timeout=1.0
        ) as response:
            return response.status == 200
    except OSError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the local RAG web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--wait-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.wait_seconds <= 0:
        raise ValueError("wait-seconds must be greater than zero")
    if _port_is_open(args.host, args.port):
        raise RuntimeError(f"{args.host}:{args.port} is already in use")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = RUNTIME_DIR / "rag-demo.stdout.log"
    stderr_path = RUNTIME_DIR / "rag-demo.stderr.log"
    pid_path = RUNTIME_DIR / "rag-demo.pid"

    creation_flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NO_WINDOW
    )
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.api:app",
                "--host",
                args.host,
                "--port",
                str(args.port),
            ],
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
            close_fds=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="ascii")

    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            error_tail = stderr_path.read_text(
                encoding="utf-8", errors="replace"
            )[-4000:]
            raise RuntimeError(
                f"demo server exited with code {process.returncode}\n{error_tail}"
            )
        if _status_is_ready(args.host, args.port):
            print(
                json.dumps(
                    {
                        "pid": process.pid,
                        "url": f"http://{args.host}:{args.port}",
                        "status": "ready",
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                    },
                    indent=2,
                )
            )
            return
        time.sleep(0.5)
    raise TimeoutError(
        f"demo server did not become ready within {args.wait_seconds} seconds"
    )


if __name__ == "__main__":
    main()
