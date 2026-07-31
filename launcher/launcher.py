#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ai_avatar — GUI launcher for the LiveTalking digital-human ship bundle.

Customer-facing entry point. Double-click this exe, enter the ElevenLabs API Key
+ Voice ID, click "Khoi dong server". The launcher:
  1. writes .env (from .env.example template, filling the two credentials),
  2. launches run_target.ps1 HIDDEN (no PowerShell window visible),
  3. polls port 8010 until the frozen LiveTalkingServer.exe is listening,
  4. shows status + a "Dung server" button that kills the process tree.

No source/.env editing, no PowerShell, no CLI for the customer. The real key
is written to .env on the customer's machine only — it is never shipped, never
logged. eleven_v3 + pcm_48000 are kept (read from .env.example); the launcher
only fills the two credential fields.
"""

import os
import sys
import queue
import socket
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox

PORT = 8010
UI_URL = f"http://127.0.0.1:{PORT}/script_player.html"
POLL_TIMEOUT_SEC = 300  # first boot (PyArmor + torch + musetalk) can take ~5 min

# Windows process creation flags
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200


def ship_root() -> str:
    """Folder containing this exe (frozen) or this script (dev)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def read_env_value(env_path: str, key: str) -> str:
    """Parse a .env-style file and return the value for `key` (or '')."""
    if not env_path or not os.path.exists(env_path):
        return ""
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def write_env(root: str, api_key: str, voice_id: str) -> None:
    """Generate .env from .env.example, filling the two credential fields.

    Reads .env.example line-by-line; replaces the ELEVENLABS_API_KEY= and
    ELEVENLABS_VOICE_ID= values; keeps every other line intact (MODEL_ID=
    eleven_v3, OUTPUT_FORMAT=pcm_48000, comments, optional tuning). Writes
    UTF-8 no-BOM. Never logs the key.
    """
    template = os.path.join(root, ".env.example")
    if not os.path.exists(template):
        raise FileNotFoundError(
            "Khong tim thay .env.example trong cung thu muc. Vui long kiem tra goi ship."
        )
    with open(template, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("ELEVENLABS_API_KEY="):
            out.append(f"ELEVENLABS_API_KEY={api_key}\n")
        elif stripped.startswith("ELEVENLABS_VOICE_ID="):
            out.append(f"ELEVENLABS_VOICE_ID={voice_id}\n")
        else:
            out.append(line)

    env_path = os.path.join(root, ".env")
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(out)


def port_open(port: int, timeout_ms: int = 400) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout_ms / 1000.0)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def pid_on_port(port: int) -> int:
    """Return the PID listening on `port` (via Get-NetTCPConnection), or 0."""
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             f"(Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue "
             f"| Select-Object -First 1).OwningProcess"],
            capture_output=True, text=True, timeout=5,
            creationflags=CREATE_NO_WINDOW,
        )
        n = out.stdout.strip()
        return int(n) if n.isdigit() else 0
    except Exception:
        return 0


class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LiveTalking — Khoi dong avatar")
        self.root.resizable(False, False)

        self._supervisor_pid = None
        self._ui_queue: "queue.Queue[tuple]" = queue.Queue()
        self._poll_thread = None

        root_dir = ship_root()
        self._root_dir = root_dir
        self._run_target = os.path.join(root_dir, "run_target.ps1")
        self._exe = os.path.join(root_dir, "LiveTalkingServer.exe")

        # Pre-fill from existing .env (if any)
        existing_env = os.path.join(root_dir, ".env")
        cur_key = read_env_value(existing_env, "ELEVENLABS_API_KEY")
        cur_voice = read_env_value(existing_env, "ELEVENLABS_VOICE_ID")

        # --- Form ---
        frm = ttk.Frame(root, padding=16)
        frm.grid(column=0, row=0, sticky="nsew")

        ttk.Label(frm, text="ElevenLabs API Key:").grid(column=0, row=0, sticky="w", pady=(0, 4))
        self.key_var = tk.StringVar(value=cur_key)
        self.key_entry = ttk.Entry(frm, textvariable=self.key_var, show="*", width=46)
        self.key_entry.grid(column=0, row=1, sticky="ew", padx=(0, 6))
        self.show_var = tk.BooleanVar(value=False)
        self.show_btn = ttk.Checkbutton(
            frm, text="Hien key", variable=self.show_var,
            command=self._toggle_show, width=9,
        )
        self.show_btn.grid(column=1, row=1, sticky="w")

        ttk.Label(frm, text="Voice ID (giong tieng Viet):").grid(column=0, row=2, sticky="w", pady=(10, 4))
        self.voice_var = tk.StringVar(value=cur_voice)
        self.voice_entry = ttk.Entry(frm, textvariable=self.voice_var, width=46)
        self.voice_entry.grid(column=0, row=3, columnspan=2, sticky="ew")

        note = ("Giu nguyen eleven_v3 + pcm_48000 (da toi uu tieng Viet).\n"
                "Key: elevenlabs.io/api-keys  ·  Voice ID: Voice Library.")
        ttk.Label(frm, text=note, foreground="#666").grid(
            column=0, row=4, columnspan=2, sticky="w", pady=(10, 0)
        )

        # --- Buttons ---
        btn_frm = ttk.Frame(frm)
        btn_frm.grid(column=0, row=5, columnspan=2, sticky="w", pady=(14, 0))
        self.start_btn = ttk.Button(btn_frm, text="Khoi dong server", command=self.on_start, width=18)
        self.start_btn.grid(column=0, row=0, padx=(0, 8))
        self.stop_btn = ttk.Button(btn_frm, text="Dung server", command=self.on_stop, width=14, state="disabled")
        self.stop_btn.grid(column=1, row=0, padx=(0, 8))
        self.log_btn = ttk.Button(btn_frm, text="Mo thu muc logs", command=self.open_logs, width=16)
        self.log_btn.grid(column=2, row=0)

        # --- Status ---
        self.status_var = tk.StringVar(value="San sang.")
        self.status_lbl = ttk.Label(frm, textvariable=self.status_var, foreground="#0a7")
        self.status_lbl.grid(column=0, row=6, columnspan=2, sticky="w", pady=(10, 0))

        frm.columnconfigure(0, weight=1)

        # If server already running on launch, reflect that.
        if port_open(PORT):
            self._supervisor_pid = pid_on_port(PORT) or None
            self._set_running_state("Server da dang chay (port 8010).")

        # Drain UI updates from poll thread.
        self.root.after(200, self._drain_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI helpers ----------
    def _toggle_show(self):
        self.key_entry.config(show="" if self.show_var.get() else "*")

    def _set_running_state(self, status: str):
        self.status_var.set(status)
        self.stop_btn.config(state="normal")
        self.start_btn.config(state="disabled")

    def _set_idle_state(self, status: str):
        self.status_var.set(status)
        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="normal")
        self._supervisor_pid = None

    def open_logs(self):
        log_dir = os.path.join(self._root_dir, "logs")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        try:
            os.startfile(log_dir)
        except Exception as e:
            messagebox.showerror("Mo logs", str(e))

    # ---------- Start ----------
    def on_start(self):
        api_key = self.key_var.get().strip()
        voice_id = self.voice_var.get().strip()
        if not api_key or not voice_id:
            messagebox.showwarning(
                "Thieu thong tin",
                "Vui long nhap day du ElevenLabs API Key va Voice ID.",
            )
            return
        if not api_key.startswith("sk_"):
            if not messagebox.askyesno(
                "Key khac thuong",
                "API Key thuong bat dau bang 'sk_'. Ban co chac day la key ElevenLabs dung?\n\nTiep tuc?",
            ):
                return
        if not os.path.exists(self._run_target):
            messagebox.showerror(
                "Thieu file",
                f"Khong tim thay run_target.ps1 tai:\n{self._run_target}",
            )
            return

        # 1) Write .env from .env.example (never logs key).
        try:
            write_env(self._root_dir, api_key, voice_id)
        except Exception as e:
            messagebox.showerror("Loi ghi .env", str(e))
            return

        # 2) Already running?
        if port_open(PORT):
            self._supervisor_pid = pid_on_port(PORT) or None
            self._set_running_state("Server da dang chay (port 8010).")
            try:
                os.startfile(UI_URL)
            except Exception:
                pass
            return

        # 3) Launch run_target.ps1 hidden.
        try:
            proc = subprocess.Popen(
                ["powershell.exe", "-ExecutionPolicy", "Bypass", "-NoProfile",
                 "-File", self._run_target],
                cwd=self._root_dir,
                creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
            )
            self._supervisor_pid = proc.pid
        except Exception as e:
            messagebox.showerror("Loi khoi dong", str(e))
            return

        self.status_var.set("Dang khoi dong server... (lan dau ~30-60s, vui long doi)")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        # 4) Poll port in background thread.
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._poll_thread = threading.Thread(target=self._poll_port, daemon=True)
        self._poll_thread.start()

    def _poll_port(self):
        import time
        deadline = time.time() + POLL_TIMEOUT_SEC
        while time.time() < deadline:
            if port_open(PORT, 800):
                self._ui_queue.put(("running",))
                return
            time.sleep(1.0)
        self._ui_queue.put(("timeout",))

    def _drain_queue(self):
        try:
            while True:
                kind = self._ui_queue.get_nowait()
                if kind == "running":
                    self._set_running_state(
                        "✓ Server san sang — trinh duyet dang mo script_player.html"
                    )
                    # run_target.ps1 already opens the browser on firstBoot.
                elif kind == "timeout":
                    self._set_idle_state(
                        "Khoi dong that bai (het gio). Xem logs\\livetalking_musetalk_common.err.log"
                    )
                    messagebox.showwarning(
                        "Khoi dong that bai",
                        "Server khong lang nghe port 8010 sau 300s.\n"
                        "Mo thu muc logs va kiem tra err.log.",
                    )
        except queue.Empty:
            pass
        self.root.after(200, self._drain_queue)

    # ---------- Stop ----------
    def on_stop(self):
        self._kill_tree()
        self._set_idle_state("Server da dung.")

    def _kill_tree(self):
        # Kill the supervisor (run_target.ps1) and its child LiveTalkingServer.exe.
        if self._supervisor_pid:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(self._supervisor_pid), "/T", "/F"],
                    capture_output=True, timeout=15,
                    creationflags=CREATE_NO_WINDOW,
                )
            except Exception:
                pass
        # Fallback: kill whatever is listening on 8010 (could be a relapsed exe).
        pid = pid_on_port(PORT)
        if pid:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True, timeout=15,
                    creationflags=CREATE_NO_WINDOW,
                )
            except Exception:
                pass
        self._supervisor_pid = None

    # ---------- Close ----------
    def on_close(self):
        if port_open(PORT):
            ans = messagebox.askyesnocancel(
                "Dung server?",
                "Server dang chay. Dung server truoc khi thoat?\n\n"
                "Co = dung server va thoat\nKhong = giu server chay, thoat launcher\nHuy = o lai",
            )
            if ans is None:
                return  # cancel -> stay
            if ans is True:
                self._kill_tree()
            # if False: leave running (detached) and exit
        self.root.destroy()


def main():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()