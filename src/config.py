from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PARENT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PARENT_DIR / ".env"


class Settings(BaseSettings):
    kaggle_username: str = Field(default="kaggle_your_username")
    kaggle_api_token: str = Field(default="kaggle_your_key")
    fred_api_key: str = Field(default="fred-api-key")
    mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db")
    mlflow_artifact_root: str = Field(default="")
    azure_storage_account: str = Field(default="")
    api_url: str = Field(default="http://localhost:8000")
    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH)


settings = Settings()
