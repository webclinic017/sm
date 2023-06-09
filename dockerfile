ARG IMAGE

FROM ${IMAGE}
LABEL maintainer="Brad Larson"

#USER root



COPY requirements.txt .
RUN --mount=type=cache,target=/var/cache/apt \
    pip3 install -r requirements.txt

EXPOSE 22 3000 5000 6006 8888 29500

# Launch container
CMD ["/bin/bash"]