# Use a standard Python 3.10 image instead of the NVIDIA one
FROM python:3.10-slim

# Create user
ARG UID
ARG MYUSER
RUN useradd -u $UID --create-home ${MYUSER}
USER ${MYUSER}

# default workdir
WORKDIR /home/${MYUSER}/
COPY --chown=${MYUSER} --chmod=765 . .

# install from source if needed + all the requirements
USER root

# install tmux, wget, and other build essentials
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tmux git wget build-essential autoconf automake libtool \
        flex bison groff texlive-latex-base latexmk graphviz \
        python3-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir -U "jax[cpu]==0.4.17" "jaxlib==0.4.17" 

# Install other direct dependencies
# This ensures they are present before your local package is installed.
RUN pip install --no-cache-dir \
    # pandas \
    # optax \
    # distrax \
    # flashbax==0.1.0 \
    wandb \
    # "hydra-core>=1.3.2" \
    # "omegaconf>=2.3.0" \
    # "pettingzoo>=1.24.3" \
    # "tqdm>=4.66.0" \
    # # "flax<=0.7.5" \ # Removed, as it's pinned above
    # safetensors \
    # chex \
    # "brax==0.10.3" \
    # "mujoco==3.1.3" \
    # matplotlib \
    # pillow \
    # "scipy<=1.12" \
    # gymnax\
    wandb[media]

# ------------------------------------------------------------

# ------------------------------------------------------------
# Download, extract, build, and install Spot from tarball
# ------------------------------------------------------------
RUN cd /tmp && \
    wget http://www.lre.epita.fr/dload/spot/spot-2.14.2.tar.gz && \
    tar xzf spot-2.14.2.tar.gz && \
    cd spot-2.14.2 && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/spot-2.14.2*
# ------------------------------------------------------------

# Now install your local package (which should pull in jaxmarl and jaxued)
RUN pip install --no-cache-dir -e .

# Add checks to verify dependencies
RUN echo "--- PIP FREEZE ---" && \
    pip freeze && \
    echo "--- PIP CHECK ---" && \
    pip check && \
    echo "--- CHECK COMPLETE ---"

# switch back to MY${MYUSER}
USER ${MYUSER}

#disabling preallocation (still useful for CPU)
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
#safety measures
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

ENV JAX_LOG_COMPILES=1

# Add this to log JAX compilations.
# This helps confirm the program is compiling, not hanging.
ENV JAX_LOG_COMPILES=1

# Removed WandB ENV vars for security.
# Pass these during `docker run` instead, e.g.:
ENV WANDB_API_KEY="5ec7d61c4826bc3a4ec4059227d66685608968b5"
ENV WANDB_ENTITY="pamfil-1938844"
# docker run -e WANDB_API_KEY="your_key" -e WANDB_ENTITY="your_entity" ...
RUN git config --global --add safe.directory /home/${MYUSER}

