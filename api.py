"""Root FastAPI runner for backwards compatibility."""
from rag_xper.api.app import app, run_api

if __name__ == "__main__":
    run_api()
