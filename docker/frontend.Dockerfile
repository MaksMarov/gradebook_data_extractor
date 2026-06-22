# syntax=docker/dockerfile:1.7
FROM node:22-alpine AS build

WORKDIR /app

COPY apps/web/frontend/package.json apps/web/frontend/package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    if [ -f package-lock.json ]; then npm ci --prefer-offline; else npm install --prefer-offline; fi

COPY apps/web/frontend ./
RUN npm run build


FROM nginx:1.27-alpine

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
