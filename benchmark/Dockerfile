FROM rootproject/root:6.34.00-ubuntu24.04


ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"
RUN uv venv /opt/venv --python=3.13
ENV PATH="/opt/venv/bin:$PATH"

ENV PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH

WORKDIR /opt
RUN git clone --depth 1 --branch v0.15.3 https://github.com/mashephe/AmpTools.git
WORKDIR /opt/AmpTools
ENV AMPTOOLS_HOME="/opt/AmpTools"
RUN make -C AmpTools
ENV AMPTOOLS="/opt/AmpTools/AmpTools"

ENV DEBIAN_FRONTEND=noninteractive
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

COPY amptools_bench /opt/amptools_bench
WORKDIR /opt/amptools_bench
RUN make

WORKDIR /work

ENTRYPOINT [ "entrypoint.sh" ]
