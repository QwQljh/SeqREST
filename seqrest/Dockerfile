FROM python:3.9-slim@sha256:2d97f6910b16bd338d3060f261f53f144965f755599aab1acda1e13cf1731b1b

WORKDIR /tool
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# Install before copying code so source edits can reuse the dependency layer.
COPY tools/seqrest/requirements.txt /tool/requirements.txt
RUN pip install --no-cache-dir -i ${PIP_INDEX_URL} -r /tool/requirements.txt \
    && pip check

COPY tools/seqrest/src/ /tool/src/
COPY tools/seqrest/runtime-defaults.json tools/seqrest/model-config.json /tool/
COPY tools/seqrest/run.sh /tool/run.sh
COPY apis/*/specifications/* /specifications/

WORKDIR /tool/src
RUN mkdir -p apis/openai_embedding && chmod +x /tool/run.sh
ENTRYPOINT ["/tool/run.sh"]
