import dataclasses
import json
from typing import Dict

@dataclasses.dataclass
class Config:
    """Stores configuration settings for the script.

    Attributes:
        debug_mode: Enables detailed logging and error reporting.
        log_file: Path to the log file.
        max_retries: Maximum number of retries for network requests.
        timeout_seconds: Timeout for network requests in seconds.
        data_format: Format for data exchange (e.g., "json", "xml").
        api_key: API key for accessing external services.
        api_secret: API secret for accessing external services.
        database_uri: URI for connecting to the database.
        table_name: Name of the table to use in the database.
        max_connections: Maximum number of database connections.
        cache_enabled: Enables caching for frequently accessed data.
        cache_duration_minutes: Duration for which cached data is valid.
        feature_flags: Dictionary of feature flags and their status.
        proxy_url: URL of the proxy server to use for network requests.
        proxy_port: Port number of the proxy server.
        user_agent: User-agent string for network requests.
        rate_limit_per_second: Maximum number of requests per second.
        backup_directory: Directory for storing backups.
        notification_email: Email address for sending notifications.
        allowed_ips: List of IP addresses allowed to access the script.
        default_language: Default language for localization.
        timezone: Timezone for date and time operations.
        export_path: Default path for exporting data.
        retry_delay_seconds: Delay between retries in seconds.
        log_level: Logging level (e.g., "INFO", "DEBUG").
        analytics_enabled: Enables analytics tracking.
        max_file_size_mb: Maximum file size for uploads in megabytes.
        session_timeout_minutes: Session timeout duration in minutes.
        theme: UI theme for the script.
        custom_scripts_path: Path to custom scripts directory.
    """
    debug_mode: bool = False
    log_file: str = "script.log"
    max_retries: int = 3
    timeout_seconds: int = 30
    data_format: str = "json"
    api_key: str = ""
    api_secret: str = ""
    database_uri: str = "sqlite:///./default.db"
    table_name: str = "data"
    max_connections: int = 10
    cache_enabled: bool = True
    cache_duration_minutes: int = 60
    feature_flags: Dict[str, bool] = dataclasses.field(default_factory=dict)
    proxy_url: str = ""
    proxy_port: int = 8080
    user_agent: str = "AEX-Script/1.0"
    rate_limit_per_second: int = 5
    backup_directory: str = "./backups"
    notification_email: str = ""
    allowed_ips: list = dataclasses.field(default_factory=list)
    default_language: str = "en"
    timezone: str = "UTC"
    export_path: str = "./exports"
    retry_delay_seconds: int = 5
    log_level: str = "INFO"
    analytics_enabled: bool = False
    max_file_size_mb: int = 100
    session_timeout_minutes: int = 30
    theme: str = "dark"
    custom_scripts_path: str = "./custom_scripts"

class ConfigManager:
    """Manages loading and saving of configuration settings.

    Attributes:
        config_file_path: Path to the configuration file.
        config: An instance of the Config dataclass.
    """

    def __init__(self, config_file_path: str = "config.json"):
        """Initializes ConfigManager with the path to the config file."""
        self.config_file_path = config_file_path
        self.config = self._load_config()

    def _load_config(self) -> Config:
        """Loads configuration from the JSON file.

        If the file doesn't exist or is invalid, returns a default Config.
        """
        try:
            with open(self.config_file_path, 'r') as f:
                config_data = json.load(f)
            return Config(**config_data)
        except (FileNotFoundError, json.JSONDecodeError):
            return Config()  # Return default config if file not found or invalid

    def save_config(self):
        """Saves the current configuration to the JSON file."""
        with open(self.config_file_path, 'w') as f:
            json.dump(dataclasses.asdict(self.config), f, indent=4)

    def get_setting(self, key: str):
        """Retrieves a specific setting from the configuration."""
        return getattr(self.config, key, None)

    def update_setting(self, key: str, value):
        """Updates a specific setting in the configuration."""
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            self.save_config()
        else:
            raise AttributeError(f"Config has no attribute '{key}'")
