FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        iperf3 \
        mininet \
        openvswitch-switch \
        python3 \
        python3-os-ken \
        python3-pip \
        python3-venv \
        tcpdump \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN python3 -m venv --system-site-packages .venv \
    && .venv/bin/python -m pip install --upgrade pip \
    && .venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt

COPY . .

CMD [".venv/bin/python", "scripts/health_check.py", "--mode", "python"]
