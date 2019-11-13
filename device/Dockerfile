
FROM tsutomu7/scientific-python
MAINTAINER Carlton Shepherd "carlton.shepherd@onespan.com"
COPY . /app

# Install requirements
WORKDIR /app
RUN pip install -r requirements.txt

# Install MPyC from source after giving
# default user (scientist) all ownership
WORKDIR ./mpyc
USER root
RUN chown -R scientist .
USER scientist 
RUN python setup.py install --user

# Launch server
WORKDIR ../
CMD ["python", "./app.py"]