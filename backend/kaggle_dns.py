"""Work around a local resolver that fails for api.kaggle.com (nslookup works, getaddrinfo doesn't).
TLS still verifies the certificate against the hostname, so this only changes how the name is found."""
import socket, subprocess
_orig = socket.getaddrinfo
_cache = {}
def _lookup(host):
    if host not in _cache:
        out = subprocess.run(["nslookup", host], capture_output=True, text=True).stdout
        ips = [l.split()[-1] for l in out.splitlines() if l.startswith("Address:") and "#" not in l]
        _cache[host] = ips[0] if ips else None
    return _cache[host]
def _patched(host, *a, **k):
    try:
        return _orig(host, *a, **k)
    except socket.gaierror:
        ip = _lookup(host) if isinstance(host, str) else None
        if not ip:
            raise
        return _orig(ip, *a, **k)
socket.getaddrinfo = _patched
