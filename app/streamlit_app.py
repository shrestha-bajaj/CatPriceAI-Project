from pathlib import Path
import importlib.util

SCRIPT_PATH = Path(__file__).resolve().parent / "10_streamlit_app.py"
spec = importlib.util.spec_from_file_location("catpriceai_streamlit_app", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
