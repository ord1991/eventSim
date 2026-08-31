## 2025-05-22 - Path Traversal Prevention in Tkinter File Path Entries

**Vulnerability:** User-controlled string inputs in GUI text entries (`recording_filename`) allowed path traversal sequences (e.g., `../../etc/passwd` or `/etc/shadow`) when passed to `os.path.join()`.
**Learning:** `os.path.join(base_dir, filename)` returns the absolute path directly if `filename` is absolute, or resolves relative path steps like `..` outside `base_dir`.
**Prevention:** Always extract pure filenames using `os.path.basename()` and validate path containment with `Path(full_path).is_relative_to(Path(base_dir).resolve())` before executing file I/O operations.
