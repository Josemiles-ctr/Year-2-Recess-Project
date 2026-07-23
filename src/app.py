import os
import sys

# Ensure the root directory is on the python search path so we can resolve 'src' modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.infrastructure.web.app_setup import create_app

app = create_app()

if __name__ == "__main__":
    # Retrieve port and debug configurations from environment variables
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    print(f" * Launching Flask Framework Expert Assistant (Port: {port})...")
    app.run(host="0.0.0.0", port=port, debug=debug)
