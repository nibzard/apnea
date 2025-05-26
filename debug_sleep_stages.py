#!/usr/bin/env python3
"""
Debug script to investigate sleep stage mapping issues
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

def parse_timestamp(timestamp):
    """Parse timestamp like in the main script"""
    if not timestamp:
        return None

    try:
        if isinstance(timestamp, str):
            if timestamp.endswith(".0"):
                timestamp = timestamp[:-2]
            if timestamp.endswith("Z"):
                timestamp = timestamp.replace("Z", "+00:00")
            elif not timestamp.endswith("+00:00") and "T" in timestamp:
                timestamp = timestamp + "+00:00"
            return datetime.fromisoformat(timestamp)
        elif isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
    except Exception as e:
        print(f"Failed to parse timestamp {timestamp}: {e}")
    return None

def create_stage_lookup(sleep_stages):
    """Create timestamp to sleep stage lookup"""
    stage_lookup = {}

    print(f"Processing {len(sleep_stages)} sleep stages...")

    for i, stage in enumerate(sleep_stages):
        start_time = parse_timestamp(stage.get("startGMT"))
        end_time = parse_timestamp(stage.get("endGMT"))
        activity_level = stage.get("activityLevel")

        print(f"Stage {i}: {start_time} to {end_time}, level: {activity_level}")

        if start_time and end_time and activity_level is not None:
            # Round start time down to the minute and end time up to the minute
            start_minute = start_time.replace(second=0, microsecond=0)
            end_minute = end_time.replace(second=0, microsecond=0)
            if end_time.second > 0 or end_time.microsecond > 0:
                end_minute += timedelta(minutes=1)

            # Create minute-by-minute lookup
            current_time = start_minute
            count = 0
            while current_time < end_minute:
                stage_lookup[current_time] = activity_level
                current_time += timedelta(minutes=1)
                count += 1
            print(f"  Added {count} minute entries from {start_minute} to {end_minute}")

    print(f"Total stage lookup entries: {len(stage_lookup)}")
    return stage_lookup

def main():
    # Load one file for testing
    data_file = Path("data/2025-04-20.json")

    with open(data_file, "r") as f:
        data = json.load(f)

    # Get sleep stages and SpO2 data
    sleep_stages = data.get("sleepLevels", [])
    spo2_data = data.get("wellnessEpochSPO2DataDTOList", [])

    print(f"Found {len(sleep_stages)} sleep stages")
    print(f"Found {len(spo2_data)} SpO2 readings")

    # Create stage lookup
    stage_lookup = create_stage_lookup(sleep_stages)

    # Check some SpO2 timestamps
    print("\nChecking first 10 SpO2 readings:")
    for i, reading in enumerate(spo2_data[:10]):
        timestamp = parse_timestamp(reading.get("epochTimestamp"))
        spo2_value = reading.get("spo2Reading")

        # Look up sleep stage
        sleep_stage = stage_lookup.get(timestamp, "NOT_FOUND")
        stage_names = {0.0: "Deep", 1.0: "Light", 2.0: "REM", 3.0: "Awake"}
        stage_name = stage_names.get(sleep_stage, f"Unknown({sleep_stage})")

        print(f"  {i}: {timestamp} -> SpO2: {spo2_value}, Stage: {stage_name}")

    # Check for low SpO2 events
    print("\nChecking for low SpO2 readings (<90%):")
    low_readings = []
    for reading in spo2_data:
        spo2_value = reading.get("spo2Reading")
        if spo2_value and spo2_value < 90:
            timestamp = parse_timestamp(reading.get("epochTimestamp"))
            sleep_stage = stage_lookup.get(timestamp, "NOT_FOUND")
            stage_names = {0.0: "Deep", 1.0: "Light", 2.0: "REM", 3.0: "Awake"}
            stage_name = stage_names.get(sleep_stage, f"Unknown({sleep_stage})")
            low_readings.append((timestamp, spo2_value, stage_name))

    print(f"Found {len(low_readings)} low SpO2 readings")
    for timestamp, spo2, stage in low_readings[:10]:  # Show first 10
        print(f"  {timestamp}: SpO2 {spo2}% during {stage}")

if __name__ == "__main__":
    main()