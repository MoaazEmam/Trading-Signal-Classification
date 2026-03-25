from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    kaggle_username: str = Field(default="kaggle_your_username")
    kaggle_key: str = Field(default="kaggle_your_key")
    mlflow_tracking_uri: str = Field(default="mlflow_uri")
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
