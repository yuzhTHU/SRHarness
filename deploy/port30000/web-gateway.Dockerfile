FROM nginxinc/nginx-unprivileged:1.27-alpine

COPY web-gateway.conf /etc/web-gateway.conf

ENTRYPOINT ["nginx", "-c", "/etc/web-gateway.conf", "-g", "daemon off;"]
