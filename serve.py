
import os

from waitress import serve

from app import create_app


# --------------------------------------------------
# PRODUCTION SERVER
# --------------------------------------------------

# Waitress replaces Flask's development server and runs on Windows.
# Defaults keep the app on this computer only. Set HOST=0.0.0.0 to
# allow other devices on your network to connect.

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8080"))
THREADS = int(os.environ.get("THREADS", "4"))


if __name__ == "__main__":

    print(f"Serving Ausentinel on http://{HOST}:{PORT} (threads={THREADS})")

    serve(
        create_app(),
        host=HOST,
        port=PORT,
        threads=THREADS,
    )
