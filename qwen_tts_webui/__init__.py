import os

from qwen_tts_webui.config import ROOT_PATH

os.environ["CACHE_HOME"] = os.getenv("CACHE_HOME", (ROOT_PATH / "cache").as_posix())
os.environ["HF_HOME"] = os.getenv("HF_HOME", (ROOT_PATH / "cache" / "huggingface").as_posix())
os.environ["MATPLOTLIBRC"] = os.getenv("MATPLOTLIBRC", (ROOT_PATH / "cache").as_posix())
os.environ["MODELSCOPE_CACHE"] = os.getenv("MODELSCOPE_CACHE", (ROOT_PATH / "cache" / "modelscope" / "hub").as_posix())
os.environ["MS_CACHE_HOME"] = os.getenv("MS_CACHE_HOME", (ROOT_PATH / "cache" / "modelscope" / "hub").as_posix())
os.environ["SYCL_CACHE_DIR"] = os.getenv("SYCL_CACHE_DIR", (ROOT_PATH / "cache" / "libsycl_cache").as_posix())
os.environ["TORCH_HOME"] = os.getenv("TORCH_HOME", (ROOT_PATH / "cache" / "torch").as_posix())
os.environ["U2NET_HOME"] = os.getenv("U2NET_HOME", (ROOT_PATH / "cache" / "u2net").as_posix())
os.environ["XDG_CACHE_HOME"] = os.getenv("XDG_CACHE_HOME", (ROOT_PATH / "cache").as_posix())
os.environ["PIP_CACHE_DIR"] = os.getenv("PIP_CACHE_DIR", (ROOT_PATH / "cache" / "pip").as_posix())
os.environ["PYTHONPYCACHEPREFIX"] = os.getenv("PYTHONPYCACHEPREFIX", (ROOT_PATH / "cache" / "pycache").as_posix())
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.getenv("TORCHINDUCTOR_CACHE_DIR", (ROOT_PATH / "cache" / "torchinductor").as_posix())
os.environ["TRITON_CACHE_DIR"] = os.getenv("TRITON_CACHE_DIR", (ROOT_PATH / "cache" / "triton").as_posix())
os.environ["UV_CACHE_DIR"] = os.getenv("UV_CACHE_DIR", (ROOT_PATH / "cache" / "uv").as_posix())
