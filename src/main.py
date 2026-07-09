import os

import uvicorn


def main() -> None:
    reload = os.getenv("UVICORN_RELOAD", "").strip().lower() in ("1", "true", "yes")
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=reload, app_dir="src")


if __name__ == "__main__":
    main()
