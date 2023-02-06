from flask import Flask

web = Flask(__name__)
web.config.from_object('config.Config')

from app import routes