FROM node:22-slim

WORKDIR /app
COPY server.js ./server.js
COPY index.html ./index.html
COPY fortigate.html ./fortigate.html

EXPOSE 8080
CMD ["node", "server.js"]
