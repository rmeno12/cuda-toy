FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Set up time zone so cmake install doesn't hang
ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install extras
RUN apt-get update \
    && apt-get install git -y \
    && apt-get install -y --no-install-recommends cuda-samples-11-3 \
    && apt-get install -y cmake protobuf-compiler \
    && apt-get install gdb -y

# Set up a user so we're not just in root
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


# Set the default user
USER $USERNAME

WORKDIR /workspaces/cuda-toy

# Set bash as default shell instead of sh
ENV SHELL /bin/bash
