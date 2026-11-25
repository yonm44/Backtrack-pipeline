import os
import sys
import json
import time
import shutil
import signal
import gmpy2
from huggingface_hub import HfApi, snapshot_download

# ---------------- CREDENTIALS & HUGGING FACE WORKSPACE CONFIG ----------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Missing HF_TOKEN environment variable. Set it before starting script.py.")

HF_USERNAME = os.getenv("HF_USERNAME", "Hackman88")
DATA_REPO_NAME = os.getenv("DATA_REPO_NAME", "Main3")

api = HfApi(token=HF_TOKEN)

DATA_REPO_DIR = "./workspace/TestingRepo"
os.makedirs(DATA_REPO_DIR, exist_ok=True)
REPO_ID = f"{HF_USERNAME}/{DATA_REPO_NAME}"
RESUME_FILE = os.path.join(DATA_REPO_DIR, "resume.json")

# ---------------- SECP256K1 PARAMETERS USING GMPY2 ----------------
P = gmpy2.mpz(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F)
Gx = gmpy2.mpz(0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798)
Gy = gmpy2.mpz(0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)

X_NEG_G = Gx
Y_NEG_G = (P - Gy) % P


def ec_subtract_g(x1, y1):
    """Computes (x1, y1) - G efficiently using gmpy2 modular inversion."""
    if x1 == X_NEG_G and y1 == Y_NEG_G:
        num = (3 * x1 * x1) % P
        den = (2 * y1) % P
    else:
        num = (Y_NEG_G - y1) % P
        den = (X_NEG_G - x1) % P

    if den == 0:
        return None, None

    lam = (num * gmpy2.invert(den, P)) % P
    x3 = (lam * lam - x1 - X_NEG_G) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return x3, y3


# ---------------- RUNTIME VALUES & BOUNDARIES ----------------
TOTAL_STEPS = 560_000_000
PUSH_INTERVAL = 20_000_000
FILE_SIZE_LIMIT_BYTES = 2 * 1024 * 1024  # 2MB limit
MAX_FILES_PER_FOLDER = 40

START_PUBKEY_X = "78a1b2b0828d1b7ad3d4ea111a96e1f1f692eeed82b634e2d5448da7a041bdd8"
START_PUBKEY_Y = "c7515c431b45ab9889b7de1ff6dead3d050ffd73381b7bcad17be3078897aaff"

curr_x = gmpy2.mpz(int(START_PUBKEY_X, 16))
curr_y = gmpy2.mpz(int(START_PUBKEY_Y, 16))
steps_done = 0
folder_id = 1
file_id = 1
file_handle = None


def build_state():
    return {
        "tracking": {
            "current_folder_id": folder_id,
            "current_file_id": file_id,
            "steps_subtracted": steps_done,
            "last_x": hex(curr_x),
            "last_y": hex(curr_y),
        }
    }


def write_resume_local():
    with open(RESUME_FILE, "w", encoding="utf-8") as f:
        json.dump(build_state(), f, indent=2)


def robust_sync_hf(repo_dir, retries=5, backstop=10):
    delay = backstop
    for attempt in range(1, retries + 1):
        try:
            api.upload_folder(
                folder_path=repo_dir,
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=["resume.json", "folder_*/*.txt"],
            )
            return True, "Synced Successfully", ""
        except Exception as e:
            print(f"[SYNC WARNING] Attempt {attempt}/{retries} failed: {e}", flush=True)
            if attempt == retries:
                return False, "", str(e)
            time.sleep(delay)
            delay *= 2


def cleanup_old_local_folders(repo_dir, active_folder_id):
    for entry in os.listdir(repo_dir):
        if entry.startswith("folder_"):
            try:
                f_num = int(entry.split("_", 1)[1])
                if f_num < active_folder_id:
                    shutil.rmtree(os.path.join(repo_dir, entry), ignore_errors=True)
            except ValueError:
                pass


def save_and_exit(sig=None, frame=None):
    print("\n[INTERRUPT] Saving state and syncing before exit...", flush=True)
    try:
        if file_handle and not file_handle.closed:
            file_handle.flush()
            file_handle.close()
    except Exception:
        pass
    try:
        write_resume_local()
        ok, _, err = robust_sync_hf(DATA_REPO_DIR)
        if ok:
            cleanup_old_local_folders(DATA_REPO_DIR, folder_id)
            print("[STOPPED] State saved safely to Hugging Face.", flush=True)
        else:
            print(f"[STOP WARNING] Sync on exit failed: {err}", flush=True)
    except Exception as e:
        print(f"[STOP ERROR] {e}", flush=True)
    sys.exit(0)


signal.signal(signal.SIGINT, save_and_exit)
signal.signal(signal.SIGTERM, save_and_exit)

# ---------------- REPOSITORY SETUP & RUNTIME RESUME ----------------
api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True)

try:
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DATA_REPO_DIR,
        allow_patterns=["resume.json", "folder_*/*.txt"],
    )
except Exception:
    pass

if os.path.exists(RESUME_FILE):
    with open(RESUME_FILE, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    t_state = loaded["tracking"]
    curr_x = gmpy2.mpz(int(t_state["last_x"], 16))
    curr_y = gmpy2.mpz(int(t_state["last_y"], 16))
    steps_done = int(t_state["steps_subtracted"])
    folder_id = int(t_state["current_folder_id"])
    file_id = int(t_state["current_file_id"])
    print(f"[RESUME] Found state file on {REPO_ID}. Re-engaging from step {steps_done:,} (Folder {folder_id}, Batch {file_id}).", flush=True)
else:
    write_resume_local()
    print(f"[INIT] Fresh backtrack pipeline created on {REPO_ID}.", flush=True)

# ---------------- CORE ARITHMETIC LOOP ----------------
current_folder_path = os.path.join(DATA_REPO_DIR, f"folder_{folder_id}")
os.makedirs(current_folder_path, exist_ok=True)
current_file_path = os.path.join(current_folder_path, f"batch_{file_id}.txt")
current_file_bytes = os.path.getsize(current_file_path) if os.path.exists(current_file_path) else 0
file_handle = open(current_file_path, "a", buffering=8192, encoding="utf-8")

print(f"[ENGINE] GMPY2 C-arithmetic module active. Target: {TOTAL_STEPS:,} steps.", flush=True)

try:
    last_log_time = time.time()
    unpushed = 0

    while steps_done < TOTAL_STEPS:
        next_x, next_y = ec_subtract_g(curr_x, curr_y)
        if next_x is None:
            print("[ERROR] Reached point at infinity (den == 0). Stopping.", flush=True)
            break

        curr_x, curr_y = next_x, next_y
        steps_done += 1
        unpushed += 1

        line = f"{steps_done}:{hex(curr_x)},{hex(curr_y)}\n"
        file_handle.write(line)
        current_file_bytes += len(line.encode("utf-8"))

        if time.time() - last_log_time > 10.0:
            pct = (steps_done / TOTAL_STEPS) * 100
            print(f"[RUNNING] Progress: {steps_done:,} / {TOTAL_STEPS:,} ({pct:.4f}%) | Folder {folder_id} | Batch {file_id}", flush=True)
            last_log_time = time.time()

        if steps_done % 25_000 == 0:
            file_handle.flush()
            if current_file_bytes >= FILE_SIZE_LIMIT_BYTES:
                file_handle.close()
                file_id += 1
                if file_id > MAX_FILES_PER_FOLDER:
                    file_id = 1
                    folder_id += 1
                current_folder_path = os.path.join(DATA_REPO_DIR, f"folder_{folder_id}")
                os.makedirs(current_folder_path, exist_ok=True)
                current_file_path = os.path.join(current_folder_path, f"batch_{file_id}.txt")
                current_file_bytes = os.path.getsize(current_file_path) if os.path.exists(current_file_path) else 0
                file_handle = open(current_file_path, "a", buffering=8192, encoding="utf-8")

        if unpushed >= PUSH_INTERVAL:
            file_handle.flush()
            print(f"[SYNC] Milestone reached ({steps_done:,} steps). Uploading to {REPO_ID}...", flush=True)
            write_resume_local()

            d_ok, _, d_err = robust_sync_hf(DATA_REPO_DIR)
            if d_ok:
                cleanup_old_local_folders(DATA_REPO_DIR, folder_id)
                print(f"[PUSH SUCCESS] Checkpoint {steps_done:,} stored securely.", flush=True)
            else:
                print(f"[PUSH FAILED] {d_err}", flush=True)
            unpushed = 0

    if file_handle and not file_handle.closed:
        file_handle.close()
    print("[COMPLETE] Subtraction iteration target achieved safely.", flush=True)
    write_resume_local()
    ok, _, err = robust_sync_hf(DATA_REPO_DIR)
    if ok:
        cleanup_old_local_folders(DATA_REPO_DIR, folder_id)
    else:
        print(f"[FINAL SYNC FAILED] {err}", flush=True)

except Exception as e:
    try:
        if file_handle and not file_handle.closed:
            file_handle.close()
    except Exception:
        pass
    print(f"[CRITICAL EXCEPTION] {type(e).__name__}: {e}. Saving emergency state...", flush=True)
    try:
        write_resume_local()
        robust_sync_hf(DATA_REPO_DIR)
    except Exception as sync_err:
        print(f"[EMERGENCY SYNC FAILED] {sync_err}", flush=True)
    sys.exit(1)