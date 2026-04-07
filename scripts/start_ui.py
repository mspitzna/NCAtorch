"""Start the NCAtorch interactive web UI."""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Start the NCAtorch interactive UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--device", default=None, help="Force device: cpu, cuda, cuda:0, mps, …")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    if args.device:
        os.environ["NCA_DEVICE"] = args.device

    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.fastapi_backend:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")

    print(f"Starting UI at http://{args.host}:{args.port}")
    if args.device:
        print(f"Device: {args.device}")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
