"""YAML-backed Pydantic settings source.

This module supplies structured defaults from ``config/settings.yaml`` while
allowing environment variables and explicit initialization values to override
individual fields.
"""

from __future__ import annotations

import os
import typing

import yaml
from pydantic_settings import PydanticBaseSettingsSource

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from pydantic.fields import FieldInfo
    from pydantic_settings import BaseSettings

_YAML_FILE_ENV = "APP_SETTINGS_YAML_FILE"
_DEFAULT_YAML_FILE = "config/settings.yaml"


class YamlConfigSource(PydanticBaseSettingsSource):
    """Read configuration defaults from a YAML file."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._data = self._read_yaml()

    @staticmethod
    def _read_yaml() -> dict[str, typing.Any]:
        path = os.environ.get(_YAML_FILE_ENV, _DEFAULT_YAML_FILE)
        if not path or not os.path.exists(path):
            return {}

        with open(path, encoding="utf-8") as stream:
            document = yaml.safe_load(stream)

        return document if isinstance(document, dict) else {}

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[typing.Any, str, bool]:
        """Return the value for ``field_name`` if present in the YAML payload."""

        if field_name in self._data:
            return self._data[field_name], field_name, False
        return None, field_name, False

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: typing.Any, value_is_complex: bool
    ) -> typing.Any:
        return value

    def __call__(self) -> dict[str, typing.Any]:
        result: dict[str, typing.Any] = {}
        for field_name, field in self.settings_cls.model_fields.items():
            value, key, complex_value = self.get_field_value(field, field_name)
            if complex_value or value is not None:
                result[key] = value
        return result
