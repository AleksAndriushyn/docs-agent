"""Desktop app: FastAPI on localhost + a native window (pywebview).

    python -m app              # native window
    python -m app --browser    # same UI in the default browser (handy for UI work)
"""
from __future__ import annotations

import argparse
import threading
import time
import webbrowser

import uvicorn


class JsApi:
    """Exposed to the page as window.pywebview.api: native file/folder pickers."""

    def pick_folder(self) -> list[str]:
        return self._dialog(folder=True)

    def pick_files(self) -> list[str]:
        return self._dialog(folder=False)

    def _dialog(self, folder: bool) -> list[str]:
        import webview

        kind = getattr(webview, "FileDialog", None)
        if kind is not None:
            dialog = kind.FOLDER if folder else kind.OPEN
        else:  # pywebview < 5
            dialog = webview.FOLDER_DIALOG if folder else webview.OPEN_DIALOG
        result = webview.windows[0].create_file_dialog(dialog, allow_multiple=not folder)
        return list(result or [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--browser", action="store_true", help="Open in the default browser instead")
    args = ap.parse_args()

    from app.server import api  # heavy imports after argparse so --help is instant

    server = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=args.port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    url = f"http://127.0.0.1:{args.port}"

    if args.browser:
        print(f"Docs Agent: {url} (Ctrl+C to stop)")
        webbrowser.open(url)
        try:
            thread.join()
        except KeyboardInterrupt:
            pass
        return

    import webview

    webview.create_window("Docs Agent", url, js_api=JsApi(), width=1320, height=860,
                          min_size=(900, 600))
    webview.start()
    server.should_exit = True


if __name__ == "__main__":
    main()
