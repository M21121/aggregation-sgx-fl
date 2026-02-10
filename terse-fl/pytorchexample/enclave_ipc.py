import atexit
import subprocess
import threading
from pathlib import Path
from typing import Optional


class PersistentTrustedRound:
    """
    Keeps ONE `gramine-sgx sgx/trusted_round` process alive and reuses it.

    Protocol (line-based):
      - Python -> Enclave: "DECRYPT <start_ts> <n_chunks> <vector_dim>\\n"
      - Enclave -> Python: "OK\\n" or "ERR <message>\\n"
      - Python -> Enclave: "QUIT\\n"
      - Enclave -> Python: "OK\\n"
    """

    def __init__(
        self,
        project_root: Path,
        cmd: Optional[list[str]] = None,
        cwd: Optional[Path] = None,
        stderr_path: Optional[Path] = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.cwd = Path(cwd).resolve() if cwd is not None else self.project_root
        self.cmd = cmd or ["gramine-sgx", "sgx/trusted_round"]

        self._stderr_file = None
        if stderr_path is not None:
            stderr_path = Path(stderr_path)
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_file = open(stderr_path, "a", buffering=1)

        self.p = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file if self._stderr_file is not None else subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        if self.p.stdin is None or self.p.stdout is None:
            raise RuntimeError("Failed to open stdin/stdout pipes to enclave process")

        # If using stderr=PIPE, drain it to avoid deadlocks.
        self._stop_stderr = threading.Event()
        self._stderr_thread = None
        if self._stderr_file is None and self.p.stderr is not None:
            self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
            self._stderr_thread.start()

        atexit.register(self.close)

    def _drain_stderr(self) -> None:
        try:
            while not self._stop_stderr.is_set():
                line = self.p.stderr.readline()  # type: ignore[union-attr]
                if not line:
                    return
                # If you want: forward to logging/print here.
        except Exception:
            return

    def _readline(self) -> str:
        line = self.p.stdout.readline()
        if line == "":
            rc = self.p.poll()
            raise RuntimeError(f"trusted_round exited unexpectedly (returncode={rc})")
        return line.rstrip("\n")

    def decrypt_round(self, start_ts: int, n_chunks: int, vector_dim: int) -> None:
        if self.p.poll() is not None:
            raise RuntimeError("trusted_round process is not running")

        self.p.stdin.write(f"DECRYPT {start_ts} {n_chunks} {vector_dim}\n")
        self.p.stdin.flush()

        resp = self._readline()
        if resp == "OK":
            return
        raise RuntimeError(f"trusted_round error: {resp}")

    def close(self) -> None:
        if getattr(self, "p", None) is None:
            return
        if self.p.poll() is not None:
            return

        try:
            if self.p.stdin is not None:
                self.p.stdin.write("QUIT\n")
                self.p.stdin.flush()
            try:
                _ = self._readline()
            except Exception:
                pass
        finally:
            try:
                self._stop_stderr.set()
            except Exception:
                pass

            try:
                self.p.terminate()
            except Exception:
                pass

            try:
                self.p.wait(timeout=5)
            except Exception:
                try:
                    self.p.kill()
                except Exception:
                    pass

            try:
                if self._stderr_file is not None:
                    self._stderr_file.close()
            except Exception:
                pass
