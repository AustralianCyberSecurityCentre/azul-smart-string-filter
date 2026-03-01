"""Restapi base settings."""

from typing import Annotated

from pydantic import StringConstraints
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Basic settings for smart string filter."""

    log_level: Annotated[str, StringConstraints(to_upper=True)] = "WARN"
