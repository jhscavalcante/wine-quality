FROM python:3.12-slim

WORKDIR /app

# nginx (gateway público); curl opcional para health manual
RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x start.sh

# nginx escuta PORT (render) ou default 8080 no start.sh; mapeamento típico: -p 80:8080
EXPOSE 8080

CMD ["./start.sh"]
