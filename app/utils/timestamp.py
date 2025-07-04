from datetime import UTC, datetime


def get_current_timestamp_ms() -> int:
    """Current timestamp in ms"""
    return int(datetime.now(UTC).timestamp() * 1000)


def validate_timestamp(timestamp: int) -> bool:
    """Validate timestamp range"""
    current_ts = get_current_timestamp_ms()
    one_year_ms = 365 * 24 * 60 * 60 * 1000
    return abs(current_ts - timestamp) <= one_year_ms
