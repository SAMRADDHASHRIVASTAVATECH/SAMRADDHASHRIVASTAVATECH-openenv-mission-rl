"""
server/app.py
OpenEnv Server Entry Point
"""

from inference import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)