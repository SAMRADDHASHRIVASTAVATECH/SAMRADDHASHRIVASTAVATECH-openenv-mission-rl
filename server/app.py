"""
server/app.py
OpenEnv Server Entry Point
"""

import sys
import os

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import app


def main():
    """Main entry point for OpenEnv server."""
    app.run(host="0.0.0.0", port=7860, debug=False)


if __name__ == "__main__":
    main()
