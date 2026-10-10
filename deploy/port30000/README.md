# Port 30000 container deployment

Run all commands from the repository root.

```bash
docker compose --env-file .env -f deploy/port30000/compose.yaml build
docker compose --env-file .env -f deploy/port30000/compose.yaml up -d
docker compose -f deploy/port30000/compose.yaml ps
docker compose -f deploy/port30000/compose.yaml logs -f --tail=100
```

The SRHarness container can write only to `logs/port30000` and its temporary filesystem. It does
not mount the repository, the Docker socket, or other host paths. Its root filesystem is read-only,
all Linux capabilities are dropped, privilege escalation is disabled, and its Docker network has
no direct Internet route. Public HTTP and HTTPS access goes through a separate Squid proxy that
rejects loopback, private, link-local, reserved, and cloud-metadata destinations.

Port 30000 is published by a separate, read-only Nginx ingress container. The SRHarness container
is attached only to the internal network; the ingress container forwards browser traffic to it.

The `/tmp` filesystem is ephemeral. It remains executable because optional scientific backends
such as Julia may install or execute runtime files there; it is still isolated inside the
container and mounted with `nosuid,nodev`.

Stop or update the service with:

```bash
docker compose -f deploy/port30000/compose.yaml down
docker compose --env-file .env -f deploy/port30000/compose.yaml up -d --build
```

The real OpenRouter API key is supplied only to the separate gateway container. Its startup script
removes the key from the running process environment after generating an ephemeral proxy config.
SRHarness receives a non-secret placeholder and can reach OpenRouter only through that gateway.
This prevents scripts in the SRHarness container from reading or exfiltrating the upstream key, but
it cannot prevent them from abusing the gateway to incur model charges. Use a dedicated, low-limit,
revocable API key for a public deployment and monitor its spend limit.

Internet-enabled tools can send workspace or conversation data to arbitrary public services, can
download untrusted code, and can abuse the server's public IP. The proxy limits access to ports 80
and 443 and blocks private networks, but it does not provide domain allow-listing or content
inspection. Do not put sensitive data in this public demo.
