from pathlib import Path
import sys

import uvicorn


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


if __name__ == "__main__":
    uvicorn.run("review_web_app:app", host="127.0.0.1", port=8512, reload=False)
