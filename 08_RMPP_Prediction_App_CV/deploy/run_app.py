# -*- coding: utf-8 -*-
# PyInstaller entry point: start Streamlit headless and open the browser.
# Build (inside 08_RMPP_Prediction_App_CV, conda env rmpp-pack):
#   pyinstaller deploy/rmpp_app.spec --noconfirm
import os
import sys
import threading
import webbrowser

from streamlit.web import cli as stcli

PORT = 8501


def _open_browser():
    try:
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(app_dir)
    threading.Timer(3.0, _open_browser).start()
    sys.argv = [
        "streamlit", "run", "app.py",
        "--server.headless=true",
        "--server.address=127.0.0.1",
        f"--server.port={PORT}",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())
