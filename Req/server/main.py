import copy
import importlib
import json
import os
import shutil
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Req.demo.run_multi_model_unified6_demo import AVAILABLE_COMBOS, process_existing_app_combo
from Req.tools.parse_flow import preprocess_apk
from Req.tools.report_generator import generate_report
from Req.tools.source_analysis_bridge import run_source_analysis
from Req.tools.zip_utils import create_zip

app = FastAPI()

# Storage configuration
STORAGE_DIR = ROOT_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
DOWNLOAD_DIR = STORAGE_DIR / "downloads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Global Configuration
GLOBAL_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
SERVER_CONFIG_FILE = Path(__file__).resolve().parent / "server_config.json"


def parse_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def load_server_config() -> dict:
    default_config = {
        "auth_enabled": True,
        "auth_verify_url": "http://127.0.0.1:8980/api/auth/user",
    }

    if not SERVER_CONFIG_FILE.exists():
        try:
            with open(SERVER_CONFIG_FILE, "w", encoding="utf-8") as handle:
                json.dump(default_config, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"Failed to create server config file: {exc}")
        return default_config

    try:
        with open(SERVER_CONFIG_FILE, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("server_config.json must be a JSON object")
        config = default_config.copy()
        config.update(raw)
        config["auth_enabled"] = parse_bool(config.get("auth_enabled"), True)
        config["auth_verify_url"] = str(config.get("auth_verify_url", "")).strip() or default_config["auth_verify_url"]
        return config
    except Exception as exc:
        print(f"Failed to load server config file: {exc}")
        return default_config


SERVER_CONFIG = load_server_config()
MOOCTEST_AUTH_ENABLED = SERVER_CONFIG["auth_enabled"]
MOOCTEST_AUTH_VERIFY_URL = SERVER_CONFIG["auth_verify_url"]

# Job Store
JOBS_FILE = ROOT_DIR / "jobs.json"
jobs = {}
jobs_lock = threading.RLock()


def get_session_id(req: Request) -> str:
    header_id = (req.headers.get("X-Session-Id") or "").strip()
    if header_id:
        return header_id
    return (req.query_params.get("session_id") or "").strip()


def verify_session(session_id: str) -> bool:
    if not session_id:
        return False
    auth_req = urllib_request.Request(
        MOOCTEST_AUTH_VERIFY_URL,
        headers={"X-Session-Id": session_id},
        method="GET",
    )
    try:
        with urllib_request.urlopen(auth_req, timeout=3) as resp:
            return 200 <= getattr(resp, "status", 0) < 300
    except urllib_error.HTTPError:
        return False
    except Exception:
        return False


def require_auth(req: Request) -> str:
    if not MOOCTEST_AUTH_ENABLED:
        return "auth_disabled"
    session_id = get_session_id(req)
    if not verify_session(session_id):
        raise HTTPException(status_code=401, detail="Authentication required. Please login from Mooctest home.")
    return session_id


def load_jobs() -> None:
    global jobs
    with jobs_lock:
        if JOBS_FILE.exists():
            try:
                with open(JOBS_FILE, "r", encoding="utf-8") as handle:
                    jobs = json.load(handle)
                print(f"Loaded {len(jobs)} jobs from {JOBS_FILE}")
            except Exception as exc:
                print(f"Failed to load jobs: {exc}")
                jobs = {}


def save_jobs() -> None:
    with jobs_lock:
        tmp_file = JOBS_FILE.with_suffix(".tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as handle:
                json.dump(jobs, handle, indent=2, ensure_ascii=False)
            os.replace(tmp_file, JOBS_FILE)
        except Exception as exc:
            print(f"Failed to save jobs: {exc}")
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)


load_jobs()


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateRequest(BaseModel):
    job_id: str
    combo: str
    lang: str = "zh"
    api_key: Optional[str] = None
    repo_link: Optional[str] = None
    app_intro: Optional[str] = None
    app_name: Optional[str] = None


class ConfigRequest(BaseModel):
    api_key: str


@app.get("/api/auth/session")
def auth_session(_: str = Depends(require_auth)):
    return {"authenticated": True, "auth_enabled": MOOCTEST_AUTH_ENABLED}


@app.get("/api/auth/user")
def auth_user(req: Request):
    session_id = get_session_id(req)
    if not session_id:
        raise HTTPException(status_code=401, detail="No session")
    auth_req = urllib_request.Request(
        MOOCTEST_AUTH_VERIFY_URL,
        headers={"X-Session-Id": session_id},
        method="GET",
    )
    try:
        with urllib_request.urlopen(auth_req, timeout=3) as resp:
            import json as _json
            return _json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError:
        raise HTTPException(status_code=401, detail="Authentication failed")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/config")
def get_config(_: str = Depends(require_auth)):
    return {"has_api_key": bool(GLOBAL_API_KEY), "api_key": GLOBAL_API_KEY}


@app.post("/api/config")
def set_config(config: ConfigRequest, _: str = Depends(require_auth)):
    global GLOBAL_API_KEY
    GLOBAL_API_KEY = config.api_key
    os.environ["DASHSCOPE_API_KEY"] = GLOBAL_API_KEY

    # Refresh runtime config for downstream modules.
    import Req.config.RunConfig

    importlib.reload(Req.config.RunConfig)
    return {"message": "Configuration updated"}


@app.get("/api/combos")
def get_combos(_: str = Depends(require_auth)):
    return {"combos": AVAILABLE_COMBOS}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    app_name: Optional[str] = Form(None),
    job_id: Optional[str] = Form(None),
    _: str = Depends(require_auth),
):
    if not job_id:
        job_id = str(uuid.uuid4())

    with jobs_lock:
        if job_id not in jobs:
            jobs[job_id] = {
                "job_id": job_id,
                "status": JobStatus.PENDING,
                "created_at": str(time.time()),
                "app_name": app_name or "Unknown",
                "combo": "",
                "lang": "",
            }

    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    file_path = job_dir / "app.apk"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with jobs_lock:
        jobs[job_id]["file_path"] = str(file_path)
        jobs[job_id]["original_filename"] = file.filename
        if app_name:
            jobs[job_id]["app_name"] = app_name
        payload = {
            "job_id": job_id,
            "message": "File uploaded successfully",
            "app_name": jobs[job_id]["app_name"],
            "original_filename": jobs[job_id]["original_filename"],
        }
        save_jobs()
    return payload


@app.post("/api/upload_source")
async def upload_source(
    job_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    _: str = Depends(require_auth),
):
    if not job_id:
        job_id = str(uuid.uuid4())

    with jobs_lock:
        if job_id not in jobs:
            jobs[job_id] = {
                "job_id": job_id,
                "status": JobStatus.PENDING,
                "created_at": str(time.time()),
                "app_name": "Unknown",
                "combo": "",
                "lang": "",
            }

    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    source_zip_path = job_dir / "source.zip"
    with open(source_zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with jobs_lock:
        jobs[job_id]["source_zip_path"] = str(source_zip_path)
    save_jobs()
    return {"message": "Source code uploaded successfully", "job_id": job_id}


def process_job_task(job_id: str, combo: str, lang: str, repo_link: str, app_intro: str, api_key: str):
    try:
        with jobs_lock:
            if job_id not in jobs:
                raise Exception("Job not found")
            jobs[job_id]["status"] = JobStatus.RUNNING
            jobs[job_id]["combo"] = combo
            jobs[job_id]["lang"] = lang
            job_info = copy.deepcopy(jobs[job_id])
        save_jobs()

        file_path_str = job_info.get("file_path")
        if not file_path_str:
            raise Exception("APK not uploaded")
        file_path = Path(file_path_str)

        effective_api_key = api_key if api_key else GLOBAL_API_KEY
        if effective_api_key:
            os.environ["DASHSCOPE_API_KEY"] = effective_api_key
            import Req.config.RunConfig

            importlib.reload(Req.config.RunConfig)

        work_dir = file_path.parent / "work"
        work_dir.mkdir(exist_ok=True)
        preprocessed = preprocess_apk(str(file_path), str(work_dir))
        if not preprocessed.get("ok"):
            raise Exception(f"APK Preprocessing failed: {preprocessed.get('message')}")
        app_dir = Path(preprocessed.get("app_dir"))

        code_doc_override = None
        source_zip_path = job_info.get("source_zip_path")
        if source_zip_path and os.path.exists(source_zip_path):
            try:
                source_extract_dir = work_dir / "source_code"
                source_extract_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(source_zip_path, "r") as zip_ref:
                    zip_ref.extractall(source_extract_dir)
                items = list(source_extract_dir.iterdir())
                analysis_target = items[0] if len(items) == 1 and items[0].is_dir() else source_extract_dir
                current_app_name = job_info.get("app_name")
                if current_app_name == "Unknown":
                    current_app_name = None
                code_doc_override = run_source_analysis(
                    str(analysis_target), str(work_dir), app_name=current_app_name
                )
            except Exception as exc:
                print(f"Error processing source zip: {exc}")

        result = process_existing_app_combo(
            app_dir=app_dir,
            combo_label=combo,
            lang=lang,
            code_doc_override=code_doc_override,
        )
        if not result.get("ok"):
            raise Exception(f"Generation failed: {result.get('message')}")

        srs_path = result.get("srs")
        tests_path = result.get("tests")
        test_json_path = result.get("test_json")
        app_name = result.get("app")
        combo_dir = Path(srs_path).parent

        report_path = combo_dir / f"{app_name}_功能测试报告.docx"
        generate_report(srs_path, tests_path, str(report_path), lang=lang)

        zip_path = combo_dir / f"{app_name}_data.zip"
        files_to_zip = [srs_path, tests_path]
        if test_json_path and os.path.exists(test_json_path):
            files_to_zip.append(test_json_path)
        create_zip(files_to_zip, str(zip_path))

        result["report"] = str(report_path)
        result["zip"] = str(zip_path)

        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["result"] = result
                jobs[job_id]["status"] = JobStatus.COMPLETED
        save_jobs()

        try:
            if file_path.parent.exists():
                shutil.rmtree(file_path.parent)
        except Exception as exc:
            print(f"Failed to cleanup job directory {file_path.parent}: {exc}")
    except Exception as exc:
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = JobStatus.FAILED
                jobs[job_id]["error"] = str(exc)
        save_jobs()
        print(f"Job {job_id} failed: {exc}")


@app.post("/api/generate")
async def start_generation(req: GenerateRequest, background_tasks: BackgroundTasks, _: str = Depends(require_auth)):
    with jobs_lock:
        if req.job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        if req.app_name:
            jobs[req.job_id]["app_name"] = req.app_name
    save_jobs()

    background_tasks.add_task(
        process_job_task,
        req.job_id,
        req.combo,
        req.lang,
        req.repo_link,
        req.app_intro,
        req.api_key,
    )
    return {"job_id": req.job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str, _: str = Depends(require_auth)):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return copy.deepcopy(jobs[job_id])


@app.get("/api/history")
async def get_history(_: str = Depends(require_auth)):
    with jobs_lock:
        items = list(jobs.items())
    history = []
    for jid, job in items:
        row = {
            "job_id": jid,
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "original_filename": job.get("original_filename"),
            "app_name": job.get("app_name", "Unknown"),
            "combo": job.get("combo"),
            "lang": job.get("lang"),
            "error": job.get("error"),
        }
        if job.get("status") == JobStatus.COMPLETED:
            row["result"] = job.get("result")
        history.append(row)
    history.sort(key=lambda x: float(x.get("created_at", 0)), reverse=True)
    return {"history": history}


@app.get("/api/download/{job_id}/{file_type}")
async def download_result(job_id: str, file_type: str, _: str = Depends(require_auth)):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = copy.deepcopy(jobs[job_id])
    if job.get("status") != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    result = job.get("result", {})
    file_path = None
    if file_type == "srs":
        file_path = result.get("srs")
    elif file_type == "tests":
        file_path = result.get("tests")
    elif file_type == "test_json":
        file_path = result.get("test_json")
    elif file_type == "report":
        file_path = result.get("report")
    elif file_type == "zip":
        file_path = result.get("zip")

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=Path(file_path).name)


#app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
