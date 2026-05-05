"""Unit tests for src/config.py"""

from __future__ import annotations

from src.config import Settings


class TestSettingsStructure:
    def test_settings_is_instantiable(self):
        assert Settings() is not None

    def test_all_fields_are_strings(self):
        s = Settings()
        for field in [
            "kaggle_username",
            "kaggle_api_token",
            "fred_api_key",
            "mlflow_tracking_uri",
            "mlflow_artifact_root",
            "azure_storage_account",
            "api_url",
        ]:
            assert isinstance(getattr(s, field), str), f"{field} should be str"

    def test_kaggle_username_field_exists(self):
        s = Settings()
        assert hasattr(s, "kaggle_username")

    def test_kaggle_api_token_field_exists(self):
        s = Settings()
        assert hasattr(s, "kaggle_api_token")

    def test_fred_api_key_field_exists(self):
        s = Settings()
        assert hasattr(s, "fred_api_key")

    def test_mlflow_tracking_uri_field_exists(self):
        s = Settings()
        assert hasattr(s, "mlflow_tracking_uri")

    def test_mlflow_artifact_root_field_exists(self):
        s = Settings()
        assert hasattr(s, "mlflow_artifact_root")

    def test_azure_storage_account_field_exists(self):
        s = Settings()
        assert hasattr(s, "azure_storage_account")

    def test_api_url_field_exists(self):
        s = Settings()
        assert hasattr(s, "api_url")

    def test_settings_overridable_via_kwargs(self):
        s = Settings(kaggle_username="test_user", fred_api_key="test_key")
        assert s.kaggle_username == "test_user"
        assert s.fred_api_key == "test_key"

    def test_api_url_kwarg_override(self):
        s = Settings(api_url="http://example.com:9000")
        assert s.api_url == "http://example.com:9000"

    def test_mlflow_tracking_uri_kwarg_override(self):
        s = Settings(mlflow_tracking_uri="sqlite:///test.db")
        assert s.mlflow_tracking_uri == "sqlite:///test.db"

    def test_settings_with_no_env_file(self, tmp_path):
        """Settings created without any .env file should fall back to defaults."""
        s = Settings(_env_file=str(tmp_path / "nonexistent.env"))
        assert isinstance(s.kaggle_username, str)
        assert isinstance(s.api_url, str)

    def test_default_mlflow_uri_is_sqlite_in_fresh_env(self, tmp_path):
        s = Settings(_env_file=str(tmp_path / "nonexistent.env"))
        assert s.mlflow_tracking_uri == "sqlite:///mlflow.db"

    def test_default_api_url_in_fresh_env(self, tmp_path):
        s = Settings(_env_file=str(tmp_path / "nonexistent.env"))
        assert s.api_url == "http://localhost:8000"
