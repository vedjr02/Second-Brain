"""Local dev entrypoint: `python -m app` (Render uses uvicorn directly instead)."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
