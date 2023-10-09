FROM python:3.9-slim-buster
USER root

WORKDIR /app
COPY . /app

RUN apt-get update
RUN apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80
CMD ["/bin/bash"]
