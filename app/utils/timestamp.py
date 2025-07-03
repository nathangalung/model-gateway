from datetime import UTC, datetime


def get_current_timestamp_ms() -> int:
    """Get current timestamp in milliseconds GMT+0"""
    return int(datetime.now(UTC).timestamp() * 1000)


def validate_timestamp(timestamp: int) -> bool:
    """Validate if timestamp is reasonable (not too far in past/future)"""
    current_ts = get_current_timestamp_ms()
    # Allow timestamps within 1 year range
    one_year_ms = 365 * 24 * 60 * 60 * 1000
    return abs(current_ts - timestamp) <= one_year_ms
