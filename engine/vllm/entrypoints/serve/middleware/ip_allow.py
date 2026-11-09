# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client IP allowlist (Tandem): serve only clients inside
VLLM_TANDEM_ALLOW, a comma-separated list of addresses or CIDR ranges
(e.g. "10.0.0.0/24,192.168.1.7"). Loopback is always allowed.

Load with `--middleware vllm.entrypoints.serve.middleware.ip_allow.IPAllowMiddleware`
(tools/tandem_serve.sh does this for ALLOW=...). The check runs after the
TCP connection is accepted: it decides who is served, it is not a
firewall. Behind a reverse proxy every request carries the proxy's address
(X-Forwarded-For is not trusted).
"""
import ipaddress
import os

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from vllm.logger import init_logger

logger = init_logger(__name__)

_LOOPBACK = (ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128"))


def parse_allow(spec: str) -> list:
    """Networks of a comma-separated address / CIDR list (raises on a bad
    entry: a typo must not open or close the server silently)."""
    nets = []
    for s in spec.split(","):
        s = s.strip()
        if s:
            nets.append(ipaddress.ip_network(s, strict=False))
    if not nets:
        raise ValueError("VLLM_TANDEM_ALLOW is empty")
    return nets


def allowed(host: str | None, nets) -> bool:
    if not host:
        return False
    try:
        a = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    if a.version == 6 and a.ipv4_mapped is not None:  # ::ffff:10.0.0.3
        a = a.ipv4_mapped
    return any(a in n for n in (*_LOOPBACK, *nets) if a.version == n.version)


class IPAllowMiddleware:
    """Pure ASGI: HTTP requests from outside the allowlist get 403,
    websockets are closed before accept."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.nets = parse_allow(os.environ.get("VLLM_TANDEM_ALLOW", ""))
        logger.info("Client IP allowlist: %s (+ loopback)",
                    ", ".join(str(n) for n in self.nets))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            client = scope.get("client")
            if not allowed(client[0] if client else None, self.nets):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                await PlainTextResponse("Forbidden", status_code=403)(
                    scope, receive, send)
                return
        await self.app(scope, receive, send)
