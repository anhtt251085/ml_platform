# SAI Platform image: API + workers + MCU toolchains.
#   docker build --build-context sai=/path/to/mcu_edge_ai -t sai-platform .
# The SAI compiler is a separate build context, so its source never enters this repository.
# Behind a TLS-inspecting proxy (corporate networks, on-prem installs): add
#   --secret id=extra_ca,src=/path/to/proxy-ca.crt --build-arg https_proxy=... --network host
FROM python:3.12-slim-bookworm

ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then \
        cp /run/secrets/extra_ca /usr/local/share/ca-certificates/extra-ca.crt && update-ca-certificates; fi; \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update -q && apt-get install -y -q --no-install-recommends \
        gcc libc6-dev git \
        gcc-arm-none-eabi libnewlib-arm-none-eabi curl qemu-system-arm qemu-system-misc \
    && rm -rf /var/lib/apt/lists/*

# MD6605: the xPack RISC-V toolchain SAI's MD6605 measurements use (pinned + checksum). Debian's
# riscv64-unknown-elf-gcc 12 has no rv32 libgcc multilib, so it cannot link MD6605 programs.
ARG XPACK_RV_VERSION=13.2.0-2
ARG XPACK_RV_SHA256=52545afb900200fbf65fe05f7ce7090b8a42c64091f4f5d43cae6bb68ea2434a
RUN curl -fsSL -o /tmp/xp.tgz "https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v${XPACK_RV_VERSION}/xpack-riscv-none-elf-gcc-${XPACK_RV_VERSION}-linux-x64.tar.gz" \
    && echo "${XPACK_RV_SHA256}  /tmp/xp.tgz" | sha256sum -c - \
    && mkdir -p /opt/xpack && tar -xzf /tmp/xp.tgz -C /opt/xpack && rm /tmp/xp.tgz \
    && ln -s /opt/xpack/xpack-riscv-none-elf-gcc-${XPACK_RV_VERSION} /opt/xpack/riscv

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bind-mounted for the install only: the SAI source never lands in an image layer.
RUN --mount=type=bind,from=sai,target=/opt/sai-src,rw pip install --no-cache-dir /opt/sai-src && sai version

COPY . .
ARG SAI_PLATFORM_VERSION=dev
ENV SAI_PLATFORM_VERSION=${SAI_PLATFORM_VERSION}
RUN useradd -m -u 1000 sai && mkdir -p /data && chown -R sai /data     # /app stays root-owned (read-only code)
USER sai

ENV SAI_PLATFORM_DATA=/data \
    SAI_MD6605_GCC=/opt/xpack/riscv/bin/riscv-none-elf-gcc \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
# --proxy-headers: behind a TLS-terminating proxy the request scheme is https, so session cookies get Secure.
# Set FORWARDED_ALLOW_IPS to the proxy's address (uvicorn trusts only 127.0.0.1 by default).
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
