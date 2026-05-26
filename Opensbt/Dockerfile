FROM ubuntu:20.04

# Env vars
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic

# Temel sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    build-essential \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    python3-pip \
    python-is-python3 \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Varsayılan Python sürümünü 3.8 olarak ayarla
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# GPG anahtarını ve ROS paket deposunu ekle
#RUN curl -sSL 'http://packages.ros.org/ros.key' | apt-key add - && \
# sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-get update && apt-get install -y curl gnupg2 lsb-release ca-certificates && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main" \
      > /etc/apt/sources.list.d/ros-latest.list

# ROS Noetic ve bağımlılıkları yükle
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    python3-rosdep \
    ros-noetic-jsk-recognition-msgs \
    && rosdep init \
    && rosdep fix-permissions \
    && rosdep update \
    && rm -rf /var/lib/apt/lists/*

# Integrate Autoware mini messages into ROS
RUN mkdir -p /home/ros_ws/src && \
    cd /home/ros_ws/src && \
    git clone https://github.com/autowarefoundation/autoware_ai_messages.git


# Install dependencies
RUN cd /home/ros_ws && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
RUN cd /home/ros_ws && \
    . /opt/ros/noetic/setup.sh && \
    catkin_make -j2

# Install scenario runner
RUN mkdir /home/scenario_runner 
COPY SCENARIO_RUNNER_ROOT /home/scenario_runner 
RUN pip3 install -r /home/scenario_runner/requirements.txt
RUN pip3 install pexpect

# Carla
RUN pip3 install --upgrade pip
RUN mkdir /home/carla
COPY CARLA_ROOT /home/carla

# Opensbt
RUN mkdir /home/opensbt
COPY OPENSBT_ROOT /home/opensbt

RUN pip3 install --upgrade setuptools
RUN pip3 install --ignore-installed psutil==5.9.8
RUN pip3 install -r /home/opensbt/requirements.txt
RUN pip3 install pygame


# Environment settings
ENV SCENARIO_RUNNER_ROOT=/home/scenario_runner 
RUN echo "export SCENARIO_RUNNER_ROOT=/home/scenario_runner" >> /root/.bashrc
RUN echo "export PYTHONPATH=/home/carla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:/home/carla/PythonAPI/carla/agents:/home/carla/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}" >> /root/.bashrc

# Add Ros env variables
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /home/ros_ws/devel/setup.bash" >> /root/.bashrc

# Shell settings
SHELL ["/bin/bash", "-c"]
RUN source /root/.bashrc

# Set working env
WORKDIR /home/opensbt
CMD ["/bin/bash"]
