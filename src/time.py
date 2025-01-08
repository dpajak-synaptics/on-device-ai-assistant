#!/usr/bin/env python3
"""
time.py

Get Las Vegas time from UTC without environment changes or extra modules.
"""

import datetime

def get_time_text() -> str:
    # Get current UTC time
    now_utc = datetime.datetime.utcnow()
    
    # Las Vegas offset from UTC (ignoring DST):
    offset_hours = -8
    las_vegas_time = now_utc + datetime.timedelta(hours=offset_hours)
    
    hour_24 = las_vegas_time.hour
    minute = las_vegas_time.minute

    if hour_24 == 0:
        hour_12 = 12
        am_pm = "AM"
    elif 1 <= hour_24 < 12:
        hour_12 = hour_24
        am_pm = "AM"
    elif hour_24 == 12:
        hour_12 = 12
        am_pm = "PM"
    else:
        hour_12 = hour_24 - 12
        am_pm = "PM"

    minute_str = str(minute).zfill(2)
    return f"The time in Las Vegas is {hour_12}:{minute_str} {am_pm}."

if __name__ == "__main__":
    time_text = get_time_text()
    print(time_text)
