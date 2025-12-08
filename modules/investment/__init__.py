from flask import Blueprint

bp = Blueprint(
    "investment",
    __name__,
    template_folder="templates"
)

from . import routes  # noqa

