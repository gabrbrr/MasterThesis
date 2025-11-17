FROM nvcr.io/nvidia/jax:23.10-py3

# Create user
ARG UID
ARG MYUSER
RUN useradd -u $UID --create-home ${MYUSER}
USER ${MYUSER}

# default workdir
WORKDIR /home/${MYUSER}/
# RUN chown -R ${MYUSER} /home/${MYUSER}
COPY --chown=${MYUSER} --chmod=765 . .

# install from source if needed + all the requirements
USER root

# install tmux
RUN apt-get update && \
    apt-get install -y tmux

RUN pip install -e .

RUN pip install --no-cache-dir \
    distrax \
    wandb \
    "tqdm>=4.66.0" \
    "orbax-checkpoint==0.5.3"
  #  matplotlib \
  #  pillow \
   # wandb[media]

# switch back to MY${MYUSER}# RUN chown -R ${MYUSER} /home/${MYUSER}
# RUN chmod -R 777 /home/${MYUSER}
USER ${MYUSER}


#disabling preallocation
RUN export XLA_PYTHON_CLIENT_PREALLOCATE=false
#safety measures
RUN export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

ENV WANDB_API_KEY="5ec7d61c4826bc3a4ec4059227d66685608968b5"
ENV WANDB_ENTITY="pamfil-1938844"
RUN git config --global --add safe.directory /home/${MYUSER}