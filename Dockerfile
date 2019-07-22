FROM python:3.7.4-stretch

WORKDIR /home/user

COPY setup.py requirements.txt README.md /home/user/
RUN pip install -r requirements.txt
RUN pip install -e .

COPY farm /home/user/farm

CMD FLASK_APP=farm.inference_rest_api flask run