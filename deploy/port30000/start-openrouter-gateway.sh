#!/bin/sh
set -eu

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "OPENROUTER_API_KEY is missing or empty." >&2
    exit 1
fi

envsubst '${OPENROUTER_API_KEY}' \
    < /etc/openrouter-gateway.conf.template \
    > /tmp/openrouter-gateway.conf
unset OPENROUTER_API_KEY

exec nginx -c /tmp/openrouter-gateway.conf -g 'daemon off;'
