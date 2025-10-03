# Example configuration for the Phase 1 ingestion components.
# Copy this file to `settings.yml` and replace the placeholder
# environment variables with your own secrets or configure a `.env`
# file to provide them at runtime.

exchange:
  binance:
    api_key: "${BINANCE_API_KEY}"
    api_secret: "${BINANCE_API_SECRET}"
    base_url: "${BINANCE_BASE_URL:https://api.binance.com}"

database:
  mysql:
    host: "${MYSQL_HOST:localhost}"
    port: "${MYSQL_PORT:3306}"
    user: "${MYSQL_USER}"
    password: "${MYSQL_PASSWORD}"
    database: "${MYSQL_DATABASE}"