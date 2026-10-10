FROM nginxinc/nginx-unprivileged:1.27-alpine

COPY openrouter-gateway.conf.template /etc/openrouter-gateway.conf.template
COPY --chmod=0555 start-openrouter-gateway.sh /usr/local/bin/start-openrouter-gateway

ENTRYPOINT ["/usr/local/bin/start-openrouter-gateway"]
