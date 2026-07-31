#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script player for LiveTalking.
Reads a text/script file and sends lines sequentially to the avatar via /human API.

Usage:
    python play_script.py --script script.txt --server http://127.0.0.1:8010 --sessionid 0
"""
import argparse
import requests
import time
import re
import sys


def split_script(text, split_by="sentence"):
    """Split script into chunks."""
    text = text.strip()
    if not text:
        return []
    if split_by == "line":
        # Each non-empty line is a chunk
        return [line.strip() for line in text.splitlines() if line.strip()]
    elif split_by == "sentence":
        # Split by sentence-ending punctuation
        chunks = re.split(r'(?<=[.!?。！？])\s+', text)
        return [c.strip() for c in chunks if c.strip()]
    else:
        return [text]


def is_speaking(server, sessionid):
    try:
        r = requests.post(f"{server}/is_speaking", json={"sessionid": sessionid}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("data", False)
    except Exception:
        pass
    return False


def send_human(server, sessionid, text, interrupt=False):
    payload = {
        "sessionid": sessionid,
        "text": text,
        "type": "echo",
        "interrupt": interrupt,
    }
    try:
        r = requests.post(f"{server}/human", json=payload, timeout=10)
        return r.status_code == 200 and r.json().get("code") == 0
    except Exception as e:
        print(f"[ERROR] Failed to send text: {e}")
        return False


def wait_until_silent(server, sessionid, poll_interval=0.5, timeout=60):
    """Wait until avatar finishes speaking."""
    start = time.time()
    while time.time() - start < timeout:
        if not is_speaking(server, sessionid):
            return True
        time.sleep(poll_interval)
    print("[WARN] Timeout waiting for avatar to finish speaking.")
    return False


def play_script(server, sessionid, script_path, split_by="sentence", wait_silent=True, delay=0.5):
    with open(script_path, "r", encoding="utf-8") as f:
        raw = f.read()

    lines = split_script(raw, split_by)
    total = len(lines)
    if total == 0:
        print("[WARN] Script file is empty.")
        return

    print(f"[INFO] Loaded {total} lines from {script_path}")
    print(f"[INFO] Server: {server} | Session: {sessionid}")
    print("-" * 60)

    for i, line in enumerate(lines, 1):
        print(f"[{i}/{total}] {line}")
        if not send_human(server, sessionid, line, interrupt=False):
            print("[ERROR] Send failed, aborting.")
            break
        if wait_silent:
            # wait a bit for TTS to start, then poll until silent
            time.sleep(delay)
            wait_until_silent(server, sessionid)
        else:
            time.sleep(delay)

    print("-" * 60)
    print("[INFO] Script playback finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiveTalking Script Player")
    parser.add_argument("--script", required=True, help="Path to script text file")
    parser.add_argument("--server", default="http://127.0.0.1:8010", help="LiveTalking server URL")
    parser.add_argument("--sessionid", default="0", help="Session ID (default '0' for local server)")
    parser.add_argument("--split-by", choices=["line", "sentence", "all"], default="sentence",
                        help="How to split script: line, sentence, or all (send whole file at once)")
    parser.add_argument("--no-wait", action="store_true", help="Do not wait for avatar to finish before next line")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between lines (seconds)")
    args = parser.parse_args()

    play_script(
        server=args.server,
        sessionid=args.sessionid,
        script_path=args.script,
        split_by=args.split_by,
        wait_silent=not args.no_wait,
        delay=args.delay,
    )
