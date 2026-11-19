"""SSH transport for the AGX navigation tools."""

import os
from pathlib import Path


def connect_navigation_host(*, timeout):
    host = os.environ.get("YOUDI_NAV_SSH_HOST", "").strip()
    user = os.environ.get("YOUDI_NAV_SSH_USER", "").strip()
    if not host or not user:
        raise RuntimeError("请设置 YOUDI_NAV_SSH_HOST 和 YOUDI_NAV_SSH_USER")

    import paramiko

    key_path = os.environ.get("YOUDI_NAV_SSH_KEY_FILE", "").strip()
    key_filename = str(Path(key_path).expanduser()) if key_path else None
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(host, username=user, timeout=timeout, key_filename=key_filename)
    except Exception:
        client.close()
        raise
    return client
