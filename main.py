import json
import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.signal import correlate


# --- Configuration ---
class Config:
    """Global configuration settings."""

    DATA_FOLDER = Path("data")
    OUTPUT_FOLDER = Path("sleep_analysis_output")
    PLOTS_SUBFOLDER = "event_plots"
    ANALYSIS_FILE = "analysis_report.txt"

    # SpO2 thresholds - enhanced with multiple levels
    LOW_SPO2_THRESHOLD = 90  # Percentage
    MODERATE_SPO2_THRESHOLD = 85  # Moderate desaturation
    SEVERE_SPO2_THRESHOLD = 80  # Severe desaturation
    CRITICAL_SPO2_THRESHOLD = 70  # Critical desaturation

    # Enhanced event detection parameters - improved sensitivity
    MIN_EVENT_DURATION_SEC = 10  # Reduced to catch brief events
    MIN_VALID_READINGS = 1  # Reduced to catch brief events
    MAX_GAP_BETWEEN_EVENTS_SEC = 120  # Maximum gap to merge events

    # Confidence thresholds - more permissive
    MIN_CONFIDENCE = 1  # Accept lower confidence readings
    MAX_CONFIDENCE = 100  # Maximum confidence value to consider

    # ODI thresholds - multiple clinical standards
    ODI_3_THRESHOLD = 3  # 3% desaturation
    ODI_4_THRESHOLD = 4  # 4% desaturation
    ODI_DESATURATION_THRESHOLD = 4  # Percentage drop for ODI event (legacy)

    # Plotting and analysis parameters
    MIN_EVENT_DURATION_PLOT_SEC = 60  # Minimum duration for detailed plot
    DEBUG = False  # Enable detailed debug logging
    MIN_SPO2_VALUE = 50  # Minimum valid SpO2 reading
    MAX_SPO2_VALUE = 100  # Maximum valid SpO2 reading
    HR_ANALYSIS_WINDOW_MINUTES = 5  # Window for HR before/after event analysis
    POTENTIAL_RESP_PAUSE_THRESHOLD_RPM = 2  # RPM for potential pause
    MIN_DURATION_FOR_PAUSE_SEC = 10  # Min duration for pause (sec)

    # Analysis windows for baseline and recovery
    BASELINE_WINDOW_MINUTES = 5
    RECOVERY_WINDOW_MINUTES = 5


# Configure logging
def setup_logging(debug: bool = False) -> None:
    """Configure logging with appropriate level and format."""
    level = logging.DEBUG if debug else logging.INFO
    format_str = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("sleep_analysis.log"),
        ],
    )


# Initialize logging
setup_logging(Config.DEBUG)
logger = logging.getLogger(__name__)

# Type aliases for clarity
Timestamp = datetime
JsonDict = Dict[str, Any]
EventData = Dict[str, Any]
PlotData = Dict[str, List[Dict[str, Any]]]

# --- Helper Functions ---


def parse_gmt_timestamp_str(ts_str):
    """Parses ISO 8601 GMT timestamp string to datetime object robustly."""
    if not ts_str:
        return None

    if ts_str.endswith("Z"):
        ts_to_parse = ts_str[:-1] + "+00:00"
    else:
        ts_to_parse = ts_str

    try:
        dt_obj = datetime.fromisoformat(ts_to_parse)
        if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        elif dt_obj.tzinfo != timezone.utc:
            dt_obj = dt_obj.astimezone(timezone.utc)
        return dt_obj
    except ValueError:
        common_formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",  # For fractional seconds
            "%Y-%m-%dT%H:%M:%S.0",  # For single decimal place
        ]
        base_ts_str = ts_str.split(".")[0]  # Part before potential millis
        if base_ts_str.endswith("Z"):
            base_ts_str = base_ts_str[:-1]

        for fmt in common_formats:
            try:
                if fmt.endswith(".0"):
                    # Special case for single decimal place
                    dt_obj_strptime = datetime.strptime(ts_str, fmt)
                else:
                    dt_obj_strptime = datetime.strptime(base_ts_str, fmt)
                return dt_obj_strptime.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        if "." in ts_str:  # Final attempt with fractional seconds
            try:
                ts_for_f = ts_str[:-1] if ts_str.endswith("Z") else ts_str
                dt_f = datetime.strptime(ts_for_f, "%Y-%m-%dT%H:%M:%S.%f")
                return dt_f.replace(tzinfo=timezone.utc)
            except ValueError:
                pass  # Fall through to returning None
        # print(f"Failed to parse timestamp: '{ts_str}'")  # For debugging
        return None


def parse_ms_timestamp(ms_int):
    """Converts GMT millisecond epoch to datetime object."""
    if ms_int is None:
        return None
    return datetime.fromtimestamp(ms_int / 1000, tz=timezone.utc)


def get_value_at_time(
    target_time_utc,
    data_list,
    time_key_parser,
    value_key,
    time_key_name="startGMT",
    window_seconds=30,
):
    """
    Finds value from timestamped data list closest to target_time_utc.
    Averages values within the window if multiple exist.
    """
    if not data_list:
        return None, []

    relevant_values = []
    for item in data_list:
        item_time_raw = item.get(time_key_name)
        if item_time_raw is None:
            continue
        item_time = time_key_parser(item_time_raw)
        if item_time is None:
            continue

        epoch_start = target_time_utc
        epoch_end = target_time_utc + timedelta(seconds=59)

        if epoch_start <= item_time <= epoch_end:
            value = item.get(value_key)
            if value is not None and value != -1:  # Exclude invalid markers
                relevant_values.append(value)

    if relevant_values:
        return np.mean(relevant_values), relevant_values
    return None, []


def get_interval_value_at_time(
    target_time_utc, interval_list, start_key_parser, end_key_parser, value_key
):
    """Finds value from time intervals containing target_time_utc."""
    if not interval_list:
        return None
    for item in interval_list:
        start_time_raw = item.get("startGMT")
        end_time_raw = item.get("endGMT")
        if start_time_raw is None or end_time_raw is None:
            continue

        start_time = start_key_parser(start_time_raw)
        end_time = end_key_parser(end_time_raw)

        if start_time is None or end_time is None:
            continue

        if start_time <= target_time_utc < end_time:
            return item.get(value_key)
    return None


def format_timedelta(td):
    """Formats timedelta into H:M:S string."""
    if td is None:
        return "N/A"
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def classify_event_severity(min_spo2: float) -> str:
    """Classify event severity based on minimum SpO2 value."""
    if min_spo2 >= Config.MODERATE_SPO2_THRESHOLD:
        return "Mild"
    elif min_spo2 >= Config.SEVERE_SPO2_THRESHOLD:
        return "Moderate"
    elif min_spo2 >= Config.CRITICAL_SPO2_THRESHOLD:
        return "Severe"
    else:
        return "Critical"


def calculate_baseline_spo2(spo2_data: List[Dict[str, Any]]) -> float:
    """Calculate baseline SpO2 from non-event periods."""
    if not spo2_data:
        return 95.0  # Default baseline

    # Use readings above threshold as baseline
    baseline_readings = []
    for reading in spo2_data:
        spo2_value = reading.get("spo2Reading") or reading.get("spo2")
        confidence = reading.get("readingConfidence") or reading.get("confidence", 0)

        if (spo2_value is not None and
                confidence is not None and
                isinstance(spo2_value, (int, float)) and
                isinstance(confidence, (int, float)) and
                spo2_value >= Config.LOW_SPO2_THRESHOLD and
                confidence >= Config.MIN_CONFIDENCE):
            baseline_readings.append(spo2_value)

    return np.mean(baseline_readings) if baseline_readings else 95.0


def should_merge_events(
    event1: Dict[str, Any], event2: Dict[str, Any]
) -> bool:
    """Determine if two events should be merged based on gap between them."""
    if not event1 or not event2:
        return False

    event1_end = event1.get("event_end_time_utc")
    event2_start = event2.get("event_start_time_utc")

    if not event1_end or not event2_start:
        return False

    gap_seconds = (event2_start - event1_end).total_seconds()
    return gap_seconds <= Config.MAX_GAP_BETWEEN_EVENTS_SEC


def merge_events(event1: Dict[str, Any], event2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two events into a single event."""
    merged_event = event1.copy()

    # Update end time to the later event's end time
    merged_event["event_end_time_utc"] = event2["event_end_time_utc"]

    # Merge readings and values
    merge_keys = [
        "spo2_readings", "hr_values_in_event", "resp_values_in_event",
        "stress_values_in_event", "hrv_values_in_event",
        "movement_values_in_event", "sleep_stages_in_event",
        "event_spo2_hr_values_for_corr", "event_spo2_resp_values_for_corr"
    ]
    for key in merge_keys:
        if key in event2:
            merged_event[key].extend(event2[key])

    # Update counts
    merged_event["potential_resp_pauses_count"] += event2.get(
        "potential_resp_pauses_count", 0
    )

    return merged_event


def plot_event_details(
    event,
    spo2_data,
    hr_data,
    resp_data,
    sleep_levels_data,
    calendar_date,
    event_idx,
    output_dir,
):
    """Plots SpO2, HR, and Respiration for a given event."""
    event_start_dt = event["event_start_time_utc"]
    event_end_dt = event["event_end_time_utc"]

    plot_window_start = event_start_dt - timedelta(minutes=30)
    plot_window_end = event_end_dt + timedelta(minutes=30)

    event_spo2_tuples = []
    for d in spo2_data:
        ts_str = d.get("epochTimestamp")
        parsed_ts = parse_gmt_timestamp_str(ts_str)
        spo2_reading = d.get("spo2Reading")
        if (
            parsed_ts
            and spo2_reading is not None
            and plot_window_start <= parsed_ts <= plot_window_end
        ):
            event_spo2_tuples.append((parsed_ts, spo2_reading))

    event_hr_tuples = []
    for d in hr_data:
        ts_ms = d.get("startGMT")
        parsed_ts = parse_ms_timestamp(ts_ms)
        hr_value = d.get("value")
        if (
            parsed_ts
            and hr_value is not None
            and plot_window_start <= parsed_ts <= plot_window_end
        ):
            event_hr_tuples.append((parsed_ts, hr_value))

    event_resp_tuples = []
    for d in resp_data:
        ts_ms = d.get("startTimeGMT")
        parsed_ts = parse_ms_timestamp(ts_ms)
        resp_value = d.get("respirationValue")
        if (
            parsed_ts
            and resp_value is not None
            and resp_value > 0
            and plot_window_start <= parsed_ts <= plot_window_end
        ):
            event_resp_tuples.append((parsed_ts, resp_value))

    if not event_spo2_tuples:
        print(f"No SpO2 data for event {event_idx} on {calendar_date}")
        return

    # Create figure with optimized size for single column legend
    fig, ax1 = plt.subplots(figsize=(13, 6.5))
    fig.suptitle(
        f"Low SpO2 Event - {calendar_date} - Event {event_idx+1}\n"
        f"Duration: {format_timedelta(event_end_dt - event_start_dt)}, "
        f"Min SpO2: {event['min_spo2_in_event']}%",
        fontsize=14,
        y=0.95,
    )

    # Plot SpO2 with reduced marker frequency but continuous line
    spo2_times, spo2_values = zip(*event_spo2_tuples)
    ax1.plot(
        spo2_times,
        spo2_values,
        "-",
        color="blue",
        label="SpO2 (%)",
        linewidth=2,
        alpha=0.7,
    )
    # Add markers with reduced frequency
    marker_indices = range(0, len(spo2_times), 2)  # Show every 2nd point
    ax1.plot(
        [spo2_times[i] for i in marker_indices],
        [spo2_values[i] for i in marker_indices],
        "bo",
        markersize=6,
        alpha=0.6,
    )

    ax1.set_xlabel("Time (GMT)", fontsize=10, labelpad=10)
    ax1.set_ylabel("SpO2 (%)", color="blue", fontsize=10, labelpad=10)
    ax1.tick_params(axis="y", labelcolor="blue")

    # Make threshold line more visible
    ax1.axhline(
        Config.LOW_SPO2_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Low Threshold ({Config.LOW_SPO2_THRESHOLD}%)",
    )

    # Make event highlight more visible
    ax1.axvspan(
        event_start_dt, event_end_dt, color="red", alpha=0.15, label="Low SpO2 Event"
    )
    ax1.set_ylim(min(spo2_values) - 5 if spo2_values else 70, 105)

    ax2 = ax1.twinx()
    if event_hr_tuples:
        hr_times, hr_values = zip(*event_hr_tuples)
        # Plot HR with reduced marker frequency
        ax2.plot(
            hr_times,
            hr_values,
            "-",
            color="green",
            label="Heart Rate (bpm)",
            linewidth=2,
            alpha=0.7,
        )
        marker_indices = range(0, len(hr_times), 2)
        ax2.plot(
            [hr_times[i] for i in marker_indices],
            [hr_values[i] for i in marker_indices],
            "gs",
            markersize=6,
            alpha=0.6,
        )
        ax2.set_ylabel("Heart Rate (bpm)", color="green", fontsize=10, labelpad=10)
        ax2.tick_params(axis="y", labelcolor="green")
        hr_min_val = min(hr_values) - 10 if hr_values else 40
        hr_max_val = max(hr_values) + 10 if hr_values else 120
        ax2.set_ylim(hr_min_val, hr_max_val)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    if event_resp_tuples:
        resp_times, resp_values = zip(*event_resp_tuples)
        # Plot respiration with reduced marker frequency
        ax3.plot(
            resp_times,
            resp_values,
            "-",
            color="magenta",
            label="Respiration (rpm)",
            linewidth=2,
            alpha=0.7,
        )
        marker_indices = range(0, len(resp_times), 2)
        ax3.plot(
            [resp_times[i] for i in marker_indices],
            [resp_values[i] for i in marker_indices],
            "mD",
            markersize=6,
            alpha=0.6,
        )
        ax3.set_ylabel("Respiration (rpm)", color="magenta", fontsize=10, labelpad=10)
        ax3.tick_params(axis="y", labelcolor="magenta")
        resp_min_val = min(resp_values) - 5 if resp_values else 5
        resp_max_val = max(resp_values) + 5 if resp_values else 30
        ax3.set_ylim(resp_min_val, resp_max_val)

    # Define sleep stage colors with better visibility
    stage_colors = {
        0: "navy",  # Deep sleep
        1: "royalblue",  # Light sleep
        2: "purple",  # REM
        3: "gray",  # Awake
    }
    stage_labels = {0: "Deep Sleep", 1: "Light Sleep", 2: "REM Sleep", 3: "Awake"}

    # Track which sleep stages we've already added to legend
    added_stages = set()
    legend_handles = []

    # Plot sleep stages with improved visibility
    for stage_interval in sleep_levels_data:
        stage_start_str = stage_interval.get("startGMT")
        stage_end_str = stage_interval.get("endGMT")
        stage_start = parse_gmt_timestamp_str(stage_start_str)
        stage_end = parse_gmt_timestamp_str(stage_end_str)
        activity_level = stage_interval.get("activityLevel")

        if (
            stage_start
            and stage_end
            and activity_level is not None
            and stage_start <= plot_window_end
            and stage_end >= plot_window_start
        ):
            color = stage_colors.get(activity_level, "white")
            ax1.axvspan(stage_start, stage_end, facecolor=color, alpha=0.15, zorder=-1)
            # Add to legend only once per stage type
            if activity_level in stage_colors and activity_level not in added_stages:
                legend_handles.append(
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=stage_colors[activity_level],
                        alpha=0.15,
                        label=stage_labels[activity_level],
                    )
                )
                added_stages.add(activity_level)

    # Format x-axis for better readability
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45)

    # Combine all legends in a single box with 1 column
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()

    # Place legend outside the plot with optimized positioning
    fig.legend(
        lines1 + lines2 + lines3 + legend_handles,
        labels1 + labels2 + labels3 + [h.get_label() for h in legend_handles],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=9,
    )

    # Adjust layout to prevent label cutoff and provide more space for the plot
    plt.tight_layout(rect=[0, 0.05, 0.88, 0.95])

    plot_filename = os.path.join(output_dir, f"{calendar_date}_event_{event_idx+1}.png")
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return plot_filename


# --- Main Analysis Function ---


def validate_spo2_reading(epoch: Dict[str, Any], calendar_date: str) -> Dict[str, Any]:
    """Validate and normalize a single SpO2 reading."""
    epoch_time_str = epoch.get("epochTimestamp")
    spo2_value = epoch.get("spo2Reading")
    confidence = epoch.get("readingConfidence", 0)

    if epoch_time_str is None:
        logger.warning("Found epoch with missing timestamp")
        return None

    if spo2_value is None:
        logger.warning(f"Found epoch with missing SpO2 value at {epoch_time_str}")
        return None

    # Additional validation for SpO2 values
    try:
        spo2_value = float(spo2_value)
        if not Config.MIN_SPO2_VALUE <= spo2_value <= Config.MAX_SPO2_VALUE:
            logger.warning(f"Invalid SpO2 value {spo2_value}% at {epoch_time_str}")
            return None
    except (ValueError, TypeError):
        logger.warning(f"Non-numeric SpO2 value {spo2_value} at {epoch_time_str}")
        return None

    # Validate confidence value
    try:
        confidence = float(confidence)
        if not Config.MIN_CONFIDENCE <= confidence <= Config.MAX_CONFIDENCE:
            logger.warning(f"Invalid confidence value {confidence} at {epoch_time_str}")
            confidence = None
    except (ValueError, TypeError):
        logger.warning(f"Non-numeric confidence value {confidence} at {epoch_time_str}")
        confidence = None

    epoch_time_utc = parse_gmt_timestamp_str(epoch_time_str)
    if epoch_time_utc is None:
        logger.warning(f"Failed to parse timestamp: {epoch_time_str}")
        return None

    return {
        "time": epoch_time_utc,
        "spo2": spo2_value,
        "confidence": confidence,
        "raw_time": epoch_time_str,
    }


def process_event_statistics(event: EventData, calendar_date: str) -> EventData:
    """Calculate and add statistics to an event."""
    spo2_vals = [r["value"] for r in event["spo2_readings"]]

    if len(spo2_vals) < Config.MIN_VALID_READINGS:
        logger.warning(
            f"Event on {calendar_date} has too few readings "
            f"({len(spo2_vals)} < {Config.MIN_VALID_READINGS})"
        )
        return None

    duration = (
        event["event_end_time_utc"] - event["event_start_time_utc"]
    ).total_seconds()

    if duration < Config.MIN_EVENT_DURATION_SEC:
        logger.warning(
            f"Event on {calendar_date} is too short "
            f"({duration}s < {Config.MIN_EVENT_DURATION_SEC}s)"
        )
        return None

    min_spo2 = min(spo2_vals)
    avg_spo2 = np.mean(spo2_vals)

    # Calculate baseline SpO2 (could be enhanced with actual baseline data)
    baseline_spo2 = 95.0  # Default baseline, could be calculated from data

    # Calculate desaturation magnitude
    max_desaturation = baseline_spo2 - min_spo2

    # Classify event severity
    severity = classify_event_severity(min_spo2)

    event.update(
        {
            "min_spo2_in_event": min_spo2,
            "avg_spo2_in_event": avg_spo2,
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
            "num_readings": len(spo2_vals),
            "baseline_spo2": baseline_spo2,
            "max_desaturation": max_desaturation,
            "severity": severity,
        }
    )

    # Calculate averages for other physiological parameters
    for param in [
        "hr_values_in_event",
        "resp_values_in_event",
        "stress_values_in_event",
        "hrv_values_in_event",
        "movement_values_in_event",
    ]:
        values = event[param]
        if values:
            avg_key = f"avg_{param[:-13]}"
            event[avg_key] = np.mean(values)
            logger.debug(
                f"{calendar_date} - {avg_key}: {event[avg_key]:.1f} "
                f"({len(values)} values)"
            )

    # Calculate dominant sleep stage
    sleep_stages = event["sleep_stages_in_event"]
    if sleep_stages:
        event["dominant_sleep_stage_in_event"] = max(
            set(sleep_stages), key=sleep_stages.count
        )
    else:
        event["dominant_sleep_stage_in_event"] = None

    # Log enhanced event information
    logger.info(
        f"Enhanced event stats - Duration: {duration:.1f}s, "
        f"Min SpO2: {min_spo2}%, Severity: {severity}, "
        f"Desaturation: {max_desaturation:.1f}%"
    )

    return event


def analyze_sleep_data(
    filepath: str,
) -> Tuple[
    str, List[EventData], PlotData, List[Dict[str, Any]], List[Dict[str, Any]], int
]:
    """Analyzes a single sleep data JSON file for low SpO2 events."""
    logger.info(f"Starting analysis of file: {filepath}")

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            logger.debug(f"Successfully loaded JSON data from {filepath}")
    except Exception as e:
        logger.error(f"Failed to load or parse {filepath}: {e}")
        return None, [], {}, [], [], 0

    dto = data.get("dailySleepDTO")
    if not dto:
        logger.error(f"No 'dailySleepDTO' found in {filepath}")
        return None, [], {}, [], [], 0

    calendar_date = dto.get("calendarDate")
    logger.info(f"Processing sleep data for date: {calendar_date}")

    # Extract all required data arrays from root level
    spo2_epochs_raw = data.get("wellnessEpochSPO2DataDTOList", [])
    hr_raw = data.get("sleepHeartRate", [])
    resp_raw = data.get("wellnessEpochRespirationDataDTOList", [])
    sleep_levels_raw = data.get("sleepLevels", [])
    movement_raw = data.get("sleepMovement", [])
    stress_raw = data.get("sleepStress", [])
    hrv_raw = data.get("hrvData", [])

    # Log data availability
    logger.debug("Found data arrays:")
    logger.debug(f"- SpO2 epochs: {len(spo2_epochs_raw)}")
    logger.debug(f"- Heart rate readings: {len(hr_raw)}")
    logger.debug(f"- Respiration readings: {len(resp_raw)}")
    logger.debug(f"- Sleep levels: {len(sleep_levels_raw)}")
    logger.debug(f"- Movement data: {len(movement_raw)}")
    logger.debug(f"- Stress data: {len(stress_raw)}")
    logger.debug(f"- HRV data: {len(hrv_raw)}")

    plot_data_package = {
        "spo2_data": spo2_epochs_raw,
        "hr_data": hr_raw,
        "resp_data": resp_raw,
        "sleep_levels_data": sleep_levels_raw,
    }

    if not spo2_epochs_raw:
        logger.warning(f"No SpO2 data found for {calendar_date}")
        return calendar_date, [], plot_data_package, [], [], 0

    # Validate and sort epochs
    valid_epochs = []
    for epoch in spo2_epochs_raw:
        validated = validate_spo2_reading(epoch, calendar_date)
        if validated:
            valid_epochs.append(validated)

    # Sort valid epochs by timestamp
    valid_epochs.sort(key=lambda x: x["time"])
    logger.info(
        f"Found {len(valid_epochs)} valid SpO2 readings "
        f"out of {len(spo2_epochs_raw)} total"
    )

    # Process epochs to detect events
    low_spo2_events = []
    current_event = None

    for idx, epoch in enumerate(valid_epochs):
        epoch_time_utc = epoch["time"]
        spo2_value = epoch["spo2"]
        confidence = epoch["confidence"]

        logger.debug(
            f"Processing epoch {idx}: "
            f"Time={epoch_time_utc}, "
            f"SpO2={spo2_value}, "
            f"Confidence={confidence}"
        )

        epoch_end_time_utc = epoch_time_utc + timedelta(minutes=1)
        is_low = spo2_value < Config.LOW_SPO2_THRESHOLD

        if is_low:
            if current_event is None:
                logger.info(
                    f"Starting new low SpO2 event at {epoch_time_utc} "
                    f"with SpO2={spo2_value}%"
                )
                current_event = {
                    "date": calendar_date,
                    "event_start_time_utc": epoch_time_utc,
                    "spo2_readings": [],
                    "hr_values_in_event": [],
                    "hr_values_before_event": [],
                    "hr_values_after_event": [],
                    "resp_values_in_event": [],
                    "stress_values_in_event": [],
                    "hrv_values_in_event": [],
                    "movement_values_in_event": [],
                    "sleep_stages_in_event": [],
                    "potential_resp_pauses_count": 0,
                    "event_spo2_hr_values_for_corr": [],
                    "event_spo2_resp_values_for_corr": [],
                }

            # Update event end time and add reading
            current_event["event_end_time_utc"] = epoch_end_time_utc
            current_event["spo2_readings"].append(
                {"time": epoch_time_utc, "value": spo2_value, "confidence": confidence}
            )

            # Collect associated physiological data
            hr_val, hr_vals_window = get_value_at_time(
                epoch_time_utc, hr_raw, parse_ms_timestamp, "value"
            )
            if hr_val is not None:
                current_event["hr_values_in_event"].append(hr_val)
                current_event["event_spo2_hr_values_for_corr"].append(
                    [spo2_value, hr_val]
                )

            resp_val, resp_vals_window = get_value_at_time(
                epoch_time_utc,
                resp_raw,
                parse_ms_timestamp,
                "respirationValue",
                time_key_name="startTimeGMT",
            )
            if resp_val is not None and resp_val > 0:
                current_event["resp_values_in_event"].append(resp_val)
                current_event["event_spo2_resp_values_for_corr"].append(
                    [spo2_value, resp_val]
                )

            # Add other physiological parameters
            for param_data, param_name, key_name in [
                (stress_raw, "stress", "value"),
                (hrv_raw, "hrv", "value"),
                (sleep_levels_raw, "sleep_stage", "activityLevel"),
                (movement_raw, "movement", "activityLevel"),
            ]:
                val = get_interval_value_at_time(
                    epoch_time_utc,
                    param_data,
                    parse_gmt_timestamp_str,
                    parse_gmt_timestamp_str,
                    key_name,
                )
                if val is not None:
                    if param_name == "sleep_stage":
                        current_event["sleep_stages_in_event"].append(val)
                    else:
                        current_event[f"{param_name}_values_in_event"].append(val)
                    logger.debug(f"Added {param_name} value: {val}")

        else:  # SpO2 is normal
            if current_event is not None:
                # Finalize the current event
                logger.info(
                    f"Finalizing event from "
                    f"{current_event['event_start_time_utc']} to "
                    f"{current_event['event_end_time_utc']}"
                )

                processed_event = process_event_statistics(current_event, calendar_date)
                if processed_event:
                    low_spo2_events.append(processed_event)
                    # ODI check and HR before/after collection will be done in
                    # the loop over low_spo2_events later
                    logger.info(
                        f"Event summary: Duration={processed_event['duration_seconds']}s, "
                        f"Min SpO2={processed_event['min_spo2_in_event']}%, "
                        f"Avg SpO2={processed_event['avg_spo2_in_event']:.1f}%"
                    )

                current_event = None

    # Handle event in progress at end of data
    if current_event is not None:
        logger.info("Finalizing event at end of data")
        processed_event = process_event_statistics(current_event, calendar_date)
        if processed_event:
            low_spo2_events.append(processed_event)
            # ODI check and HR before/after collection will be done in the loop
            # over low_spo2_events later

    # Enhanced event merging logic - merge events that are close together
    if len(low_spo2_events) > 1:
        logger.info(f"Checking {len(low_spo2_events)} events for potential merging...")
        merged_events = []
        i = 0

        while i < len(low_spo2_events):
            current_event = low_spo2_events[i]

            # Look ahead to see if we should merge with the next event
            while (i + 1 < len(low_spo2_events) and
                   should_merge_events(current_event, low_spo2_events[i + 1])):
                next_event = low_spo2_events[i + 1]
                logger.info(
                    f"Merging events: {current_event['event_start_time_utc']} "
                    f"and {next_event['event_start_time_utc']}"
                )
                current_event = merge_events(current_event, next_event)
                i += 1  # Skip the merged event

            # Recalculate statistics for merged event
            recalculated_event = process_event_statistics(current_event, calendar_date)
            if recalculated_event:
                merged_events.append(recalculated_event)

            i += 1

        original_count = len(low_spo2_events)
        low_spo2_events = merged_events
        merged_count = len(low_spo2_events)

        if original_count != merged_count:
            logger.info(
                f"Event merging complete: {original_count} events merged into "
                f"{merged_count} events"
            )

    # Corrected ODI logic and HR before/after collection
    final_odi_count = 0
    for i_event, event_obj in enumerate(low_spo2_events):

        # Collect HR data before and after the event
        event_start_dt = event_obj["event_start_time_utc"]
        event_end_dt = event_obj["event_end_time_utc"]
        hr_window_delta = timedelta(minutes=Config.HR_ANALYSIS_WINDOW_MINUTES)

        # HR Before
        before_start_dt = event_start_dt - hr_window_delta
        hr_values_before = []
        for hr_reading in hr_raw:  # hr_raw is from the main data load
            hr_time_ms = hr_reading.get("startGMT")
            hr_val = hr_reading.get("value")
            if hr_time_ms is not None and hr_val is not None:
                hr_dt = parse_ms_timestamp(hr_time_ms)
                if hr_dt and before_start_dt <= hr_dt < event_start_dt:
                    hr_values_before.append(hr_val)
        event_obj["hr_values_before_event"] = hr_values_before
        if hr_values_before:  # Log if data found
            logger.debug(
                f"Event {event_start_dt}: Found {len(hr_values_before)} "
                f"HR readings before event."
            )

        # HR After
        after_end_dt = event_end_dt + hr_window_delta
        hr_values_after = []
        for hr_reading in hr_raw:
            hr_time_ms = hr_reading.get("startGMT")
            hr_val = hr_reading.get("value")
            if hr_time_ms is not None and hr_val is not None:
                hr_dt = parse_ms_timestamp(hr_time_ms)
                if hr_dt and event_end_dt <= hr_dt < after_end_dt:
                    hr_values_after.append(hr_val)
        event_obj["hr_values_after_event"] = hr_values_after
        if hr_values_after:  # Log if data found
            logger.debug(
                f"Event {event_start_dt}: Found {len(hr_values_after)} "
                f"HR readings after event."
            )

        # Analyze Respiration for Potential Pauses within the event period
        event_start_dt = event_obj["event_start_time_utc"]
        event_end_dt = event_obj["event_end_time_utc"]
        potential_pauses_count_for_event = 0
        consecutive_low_rpm_duration_sec = 0
        last_valid_resp_time_for_pause_calc = None

        # Filter resp_raw for data points relevant to this event window
        # Consider a slightly extended window to catch pauses starting/ending
        # around event boundaries but only count them if they overlap
        # significantly with the SpO2 event itself.
        pause_analysis_start_dt = event_start_dt - timedelta(
            seconds=Config.MIN_DURATION_FOR_PAUSE_SEC * 2
        )  # Look a bit before
        pause_analysis_end_dt = event_end_dt + timedelta(
            seconds=Config.MIN_DURATION_FOR_PAUSE_SEC * 2
        )  # Look a bit after

        relevant_resp_points = []
        for r_point in resp_raw:  # resp_raw is for the whole day
            ts_ms = r_point.get("startTimeGMT")
            resp_val = r_point.get("respirationValue")
            if ts_ms is not None and resp_val is not None:  # Ensure data is valid
                dt = parse_ms_timestamp(ts_ms)
                if dt and pause_analysis_start_dt <= dt <= pause_analysis_end_dt:
                    relevant_resp_points.append({"time": dt, "value": resp_val})

        relevant_resp_points.sort(
            key=lambda x: x["time"]
        )  # Crucial: process in time order

        for i_rp, resp_reading in enumerate(relevant_resp_points):
            # Determine the duration this reading represents
            current_reading_timestamp = resp_reading["time"]
            reading_duration_sec = 60  # Default if it's the last point or next is far
            if i_rp < len(relevant_resp_points) - 1:
                next_reading_timestamp = relevant_resp_points[i_rp + 1]["time"]
                time_diff_to_next = (
                    next_reading_timestamp - current_reading_timestamp
                ).total_seconds()
                # Assuming resp data points are not more frequent than
                # MIN_DURATION_FOR_PAUSE_SEC and a reading represents the
                # state until the next reading or a typical interval (e.g., 60s)
                reading_duration_sec = min(
                    time_diff_to_next, 90
                )  # Cap at 1.5 min, realistic for sleep data points
            elif (
                last_valid_resp_time_for_pause_calc
            ):  # last point, use diff from previous if available
                time_diff_from_prev = (
                    current_reading_timestamp - last_valid_resp_time_for_pause_calc
                ).total_seconds()
                reading_duration_sec = min(time_diff_from_prev, 90)

            # Only consider this reading for pause counting if its period
            # overlaps with the actual SpO2 event
            reading_period_start = current_reading_timestamp
            reading_period_end = current_reading_timestamp + timedelta(
                seconds=reading_duration_sec
            )

            # Check for overlap: (StartA <= EndB) and (EndA >= StartB)
            overlaps_with_spo2_event = (
                reading_period_start < event_end_dt
                and reading_period_end > event_start_dt
            )

            if overlaps_with_spo2_event:
                if resp_reading["value"] <= Config.POTENTIAL_RESP_PAUSE_THRESHOLD_RPM:
                    consecutive_low_rpm_duration_sec += reading_duration_sec
                else:  # Respiration is above threshold
                    if (
                        consecutive_low_rpm_duration_sec
                        >= Config.MIN_DURATION_FOR_PAUSE_SEC
                    ):
                        potential_pauses_count_for_event += 1
                    consecutive_low_rpm_duration_sec = 0  # Reset counter
            else:  # This resp reading is outside the SpO2 event, reset any
                # ongoing pause counting for this event
                if (
                    consecutive_low_rpm_duration_sec
                    >= Config.MIN_DURATION_FOR_PAUSE_SEC
                ):
                    # This check ensures that if a pause was ongoing and the
                    # readings move out of the spo2 event window, it is
                    # counted before resetting. This might be redundant if the
                    # main check is robust.
                    potential_pauses_count_for_event += (
                        1  # Count it if it was long enough and ended here
                    )
                consecutive_low_rpm_duration_sec = 0

            last_valid_resp_time_for_pause_calc = current_reading_timestamp

        # Check if a pause was ongoing at the end of the relevant respiration data processing
        # and this pause period was overlapping with the event
        if consecutive_low_rpm_duration_sec >= Config.MIN_DURATION_FOR_PAUSE_SEC:
            # This final check needs to be careful not to double count if the last resp_reading was outside spo2_event
            # However, the logic above should handle it. This catches a pause that extends to the end of resp_data.
            potential_pauses_count_for_event += 1

        event_obj["potential_resp_pauses_count"] = potential_pauses_count_for_event
        if potential_pauses_count_for_event > 0:
            logger.debug(
                f"Event {event_start_dt}: Found {potential_pauses_count_for_event} potential respiratory pauses."
            )

        # Existing ODI logic starts here
        min_spo2_in_current_event = event_obj.get(
            "min_spo2_in_event"
        )  # Use .get for safety
        if (
            min_spo2_in_current_event is None
        ):  # Explicitly check if min_spo2_in_event is missing
            event_obj["is_odi_event"] = False
            logger.debug(
                f"Event {event_obj['event_start_time_utc']} missing min_spo2_in_event, "
                f"cannot calculate ODI."
            )
            continue  # Skip ODI calculation for this event

        event_start_time = event_obj[
            "event_start_time_utc"
        ]  # Already have event_start_dt

        pre_event_spo2_sum = 0
        pre_event_spo2_count = 0

        current_event_start_epoch_index = -1
        for i_epoch, epoch_data in enumerate(valid_epochs):
            if epoch_data["time"] == event_start_time:
                current_event_start_epoch_index = i_epoch
                break

        if current_event_start_epoch_index == -1:
            logger.warning(
                f"Could not find start epoch for event beginning at {event_start_time} for ODI."
            )
            event_obj["is_odi_event"] = False
            continue

        for k in range(
            max(0, current_event_start_epoch_index - 5), current_event_start_epoch_index
        ):
            candidate_epoch = valid_epochs[k]
            candidate_time = candidate_epoch["time"]
            candidate_spo2 = candidate_epoch["spo2"]
            is_in_another_event = False
            for other_event_idx, other_event_obj in enumerate(low_spo2_events):
                if i_event == other_event_idx:
                    continue
                if (
                    other_event_obj["event_start_time_utc"]
                    <= candidate_time
                    < other_event_obj["event_end_time_utc"]
                ):
                    is_in_another_event = True
                    break
            if not is_in_another_event and candidate_spo2 >= Config.LOW_SPO2_THRESHOLD:
                pre_event_spo2_sum += candidate_spo2
                pre_event_spo2_count += 1
            elif is_in_another_event or candidate_spo2 < Config.LOW_SPO2_THRESHOLD:
                pre_event_spo2_sum = 0
                pre_event_spo2_count = 0

        if pre_event_spo2_count > 0:
            pre_event_baseline_spo2 = pre_event_spo2_sum / pre_event_spo2_count
            desaturation_depth = pre_event_baseline_spo2 - min_spo2_in_current_event
            if desaturation_depth >= Config.ODI_DESATURATION_THRESHOLD:
                final_odi_count += 1
                event_obj["is_odi_event"] = True
                logger.info(
                    f"ODI event counted for event starting {event_start_time}: "
                    f"Pre-baseline {pre_event_baseline_spo2:.1f}%, "
                    f"Min SpO2 {min_spo2_in_current_event:.1f}%, "
                    f"Drop {desaturation_depth:.1f}%"
                )
            else:
                event_obj["is_odi_event"] = False
        else:
            event_obj["is_odi_event"] = False
            logger.debug(
                f"Not enough pre-event data to determine ODI for event starting {event_start_time}"
            )
        # --- End of ODI Calculation ---

    logger.info(
        f"Analysis complete for {calendar_date}. "
        f"Found {len(low_spo2_events)} events. ODI events: {final_odi_count}"
    )
    return (
        calendar_date,
        low_spo2_events,
        plot_data_package,
        valid_epochs,
        sleep_levels_raw,
        final_odi_count,
    )


def format_physiological_value(
    param_name: str, value: float, unit: str = "", precision: int = 1
) -> str:
    """Format a physiological value with appropriate precision."""
    if value is None:
        return "N/A"
    if unit:
        return f"{value:.{precision}f} {unit}"
    return f"{value:.{precision}f}"


def write_event_to_report(f, event, event_idx):
    """Writes a single event's details to the analysis file."""
    stage_map = {0: "Deep", 1: "Light", 2: "REM", 3: "Awake", None: "N/A"}
    f.write(f"    Event {event_idx + 1}:\n")

    # Format timestamps
    start_time_fmt = event["event_start_time_utc"].strftime("%Y-%m-%d %H:%M:%S")
    end_time_fmt = event["event_end_time_utc"].strftime("%Y-%m-%d %H:%M:%S")

    # Format durations and values
    duration_fmt = format_timedelta(timedelta(seconds=event["duration_seconds"]))
    min_spo2_fmt = format_physiological_value(
        "min_spo2", event["min_spo2_in_event"], "%"
    )
    avg_spo2_fmt = format_physiological_value(
        "avg_spo2", event["avg_spo2_in_event"], "%"
    )

    # Write basic event info
    f.write(f"      Start Time (GMT): {start_time_fmt}\n")
    f.write(f"      End Time (GMT):   {end_time_fmt}\n")
    f.write(f"      Duration:         {duration_fmt}\n")
    f.write(f"      Min SpO2:         {min_spo2_fmt}\n")
    f.write(f"      Avg SpO2:         {avg_spo2_fmt}\n")

    # Enhanced metrics from improved analysis
    if "severity" in event:
        f.write(f"      Severity:         {event['severity']}\n")

    if "baseline_spo2" in event:
        baseline_fmt = format_physiological_value(
            "baseline", event["baseline_spo2"], "%"
        )
        f.write(f"      Baseline SpO2:    {baseline_fmt}\n")

    if "max_desaturation" in event:
        desat_fmt = format_physiological_value(
            "desaturation", event["max_desaturation"], "%"
        )
        f.write(f"      Max Desaturation: {desat_fmt}\n")

    if "duration_minutes" in event:
        duration_min_fmt = format_physiological_value(
            "duration", event["duration_minutes"], "min"
        )
        f.write(f"      Duration (min):   {duration_min_fmt}\n")

    f.write("      SpO2 Readings (Time, Value, Confidence):\n")

    # Write detailed SpO2 readings
    for r in event["spo2_readings"]:
        time_fmt = r["time"].strftime("%H:%M:%S")
        conf_str = (
            f"Conf: {r['confidence']}" if r["confidence"] is not None else "Conf: N/A"
        )
        spo2_fmt = format_physiological_value("spo2", r["value"], "%")
        f.write(f"        - {time_fmt}: {spo2_fmt} ({conf_str})\n")

    # Write physiological parameters
    param_formats = [
        ("hr", "bpm"),
        ("resp", "rpm"),
        ("stress", ""),
        ("hrv", "ms"),
        ("movement", ""),
    ]

    for param, unit in param_formats:
        avg_key = f"avg_{param}"
        if avg_key in event:
            value = event[avg_key]
            precision = 2 if param == "movement" else 1
            formatted_value = format_physiological_value(param, value, unit, precision)
            param_label = f"Avg {param.title()}:"
            f.write(f"      {param_label:<16} {formatted_value}\n")

    # Write sleep stage
    dominant_stage_key = event["dominant_sleep_stage_in_event"]
    stage_name = stage_map.get(dominant_stage_key, "N/A")
    f.write(f"      Sleep Stage:      {stage_name}\n")

    # Write potential respiration pauses if any
    if event.get("potential_resp_pauses_count", 0) > 0:
        f.write(
            f"      Potential Resp Pauses: {event['potential_resp_pauses_count']}\n"
        )

    # Write ODI information if available
    if "is_odi_event" in event:
        odi_status = "Yes" if event["is_odi_event"] else "No"
        f.write(f"      ODI Event:        {odi_status}\n")

    f.write("\n")  # Extra newline for spacing between events in report


def analyze_time_series_correlation(
    spo2_data, hr_data, resp_data, calendar_date, output_dir, report_file=None
):
    """
    Compute and plot correlations (Pearson, Spearman, cross-correlation) between SpO2, HR, and Respiration.
    Save plots and write a summary to the report file if provided.
    """

    # Early check for empty or missing data
    def has_valid(data, time_key, value_key):
        return any(
            d.get(value_key) is not None and d.get(time_key) is not None for d in data
        )

    if not (
        has_valid(spo2_data, "epochTimestamp", "spo2Reading")
        and has_valid(hr_data, "startGMT", "value")
        and has_valid(resp_data, "startTimeGMT", "respirationValue")
    ):
        if report_file:
            report_file.write(
                "  [Correlation analysis skipped: insufficient SpO2, HR, or Resp data.]\n"
            )
        return None
    # Convert to DataFrames
    spo2_df = pd.DataFrame(
        [
            {
                "time": parse_gmt_timestamp_str(d.get("epochTimestamp")),
                "spo2": d.get("spo2Reading"),
            }
            for d in spo2_data
            if d.get("spo2Reading") is not None
            and parse_gmt_timestamp_str(d.get("epochTimestamp")) is not None
        ]
    ).set_index("time")
    hr_df = pd.DataFrame(
        [
            {"time": parse_ms_timestamp(d.get("startGMT")), "hr": d.get("value")}
            for d in hr_data
            if d.get("value") is not None
            and parse_ms_timestamp(d.get("startGMT")) is not None
        ]
    ).set_index("time")
    resp_df = pd.DataFrame(
        [
            {
                "time": parse_ms_timestamp(d.get("startTimeGMT")),
                "resp": d.get("respirationValue"),
            }
            for d in resp_data
            if d.get("respirationValue") is not None
            and parse_ms_timestamp(d.get("startTimeGMT")) is not None
        ]
    ).set_index("time")

    # Resample to 1-min intervals and align
    df = pd.concat([spo2_df, hr_df, resp_df], axis=1)
    df = df.resample("1T").mean().interpolate()
    df = df.dropna()  # Only keep rows where all three are present

    if len(df) < 10:
        if report_file:
            report_file.write(
                "  [Correlation analysis skipped: insufficient aligned data points.]\n"
            )
        return None

    # Compute correlations
    pearson_corr = df.corr(method="pearson")
    spearman_corr = df.corr(method="spearman")

    # Save correlation matrix plot
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Pearson Correlation ({calendar_date})")
    pearson_plot_path = os.path.join(output_dir, f"{calendar_date}_pearson_corr.png")
    plt.tight_layout()
    plt.savefig(pearson_plot_path)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Spearman Correlation ({calendar_date})")
    spearman_plot_path = os.path.join(output_dir, f"{calendar_date}_spearman_corr.png")
    plt.tight_layout()
    plt.savefig(spearman_plot_path)
    plt.close()

    # Cross-correlation (example: SpO2 vs Resp, SpO2 vs HR)
    crosscorr_results = {}
    for a, b in [("spo2", "resp"), ("spo2", "hr"), ("hr", "resp")]:
        x = df[a] - df[a].mean()
        y = df[b] - df[b].mean()
        cross_corr = correlate(x, y, mode="full")
        lags = np.arange(-len(x) + 1, len(x))
        max_corr_idx = np.argmax(np.abs(cross_corr))
        lag_at_max = lags[max_corr_idx]
        crosscorr_results[(a, b)] = (lag_at_max, cross_corr[max_corr_idx])
        # Plot
        plt.figure(figsize=(7, 3))
        plt.plot(lags, cross_corr)
        plt.title(f"Cross-correlation: {a} vs {b} ({calendar_date})")
        plt.xlabel("Lag (minutes)")
        plt.ylabel("Correlation")
        plt.axvline(0, color="k", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plot_path = os.path.join(
            output_dir, f"{calendar_date}_crosscorr_{a}_vs_{b}.png"
        )
        plt.savefig(plot_path)
        plt.close()

    # Write summary to report
    if report_file:
        report_file.write("  [Time Series Correlation Analysis]\n")
        report_file.write(
            f"    Pearson Correlation Matrix:\n{pearson_corr.round(2).to_string()}\n"
        )
        report_file.write(
            f"    Spearman Correlation Matrix:\n{spearman_corr.round(2).to_string()}\n"
        )
        for (a, b), (lag, val) in crosscorr_results.items():
            report_file.write(
                f"    Max cross-correlation {a} vs {b}: lag={lag} min, value={val:.2f}\n"
            )
        report_file.write(
            f"    Plots: {os.path.basename(pearson_plot_path)}, {os.path.basename(spearman_plot_path)}, and cross-corr plots.\n\n"
        )


def event_based_correlation_analysis(
    all_files_events, hr_data_by_date, resp_data_by_date, report_file=None
):
    """
    For each event, compute HR/Resp mean before, during, after, and correlation between SpO2 and HR/Resp within the event.
    Aggregate results and write summary to report.
    """
    before_deltas_hr = []
    after_deltas_hr = []
    before_deltas_resp = []
    after_deltas_resp = []
    event_spo2_hr_corrs = []
    event_spo2_resp_corrs = []
    window = 5  # minutes before/after
    for event in all_files_events:
        date = event["date"]
        event_start = event["event_start_time_utc"]
        event_end = event["event_end_time_utc"]
        # HR
        hr_data = hr_data_by_date.get(date, [])
        hr_times = [parse_ms_timestamp(d.get("startGMT")) for d in hr_data]
        hr_vals = [d.get("value") for d in hr_data]
        # Resp
        resp_data = resp_data_by_date.get(date, [])
        resp_times = [parse_ms_timestamp(d.get("startTimeGMT")) for d in resp_data]
        resp_vals = [d.get("respirationValue") for d in resp_data]
        # SpO2 (for event window correlation)
        # spo2_times = [r['time'] for r in event['spo2_readings']] # Unused
        # spo2_vals = [r['value'] for r in event['spo2_readings']]

        # Extract pre-aligned SpO2-HR and SpO2-Resp values for correlation
        event_spo2_hr_pairs = event.get("event_spo2_hr_values_for_corr", [])
        event_spo2_resp_pairs = event.get("event_spo2_resp_values_for_corr", [])

        spo2_for_hr_corr = [pair[0] for pair in event_spo2_hr_pairs]
        hr_for_spo2_corr = [pair[1] for pair in event_spo2_hr_pairs]

        spo2_for_resp_corr = [pair[0] for pair in event_spo2_resp_pairs]
        resp_for_spo2_corr = [pair[1] for pair in event_spo2_resp_pairs]

        # HR before/during/after (for deltas - 'during' here is the general list, not for correlation)
        hr_before = [
            v
            for t, v in zip(hr_times, hr_vals)
            if t and event_start - timedelta(minutes=window) <= t < event_start
        ]
        # Use pre-calculated aligned HR values during the event for correlation
        hr_during = event.get("hr_values_in_event", [])
        hr_after = [
            v
            for t, v in zip(hr_times, hr_vals)
            if t and event_end <= t < event_end + timedelta(minutes=window)
        ]
        # Resp before/during/after
        resp_before = [
            v
            for t, v in zip(resp_times, resp_vals)
            if t and event_start - timedelta(minutes=window) <= t < event_start
        ]
        # Use pre-calculated aligned Resp values during the event for correlation
        resp_during = event.get("resp_values_in_event", [])
        resp_after = [
            v
            for t, v in zip(resp_times, resp_vals)
            if t and event_end <= t < event_end + timedelta(minutes=window)
        ]
        # Compute deltas
        if hr_before and hr_during:
            before_deltas_hr.append(np.mean(hr_during) - np.mean(hr_before))
        if hr_during and hr_after:
            after_deltas_hr.append(np.mean(hr_after) - np.mean(hr_during))
        if resp_before and resp_during:
            before_deltas_resp.append(np.mean(resp_during) - np.mean(resp_before))
        if resp_during and resp_after:
            after_deltas_resp.append(np.mean(resp_after) - np.mean(resp_during))
        # Event window SpO2 correlation
        if (
            len(spo2_for_hr_corr) > 2 and len(hr_for_spo2_corr) > 2
        ):  # Ensure we have enough pairs
            # Check for variance
            if np.std(spo2_for_hr_corr) > 0 and np.std(hr_for_spo2_corr) > 0:
                corr_hr = np.corrcoef(spo2_for_hr_corr, hr_for_spo2_corr)[0, 1]
                if not np.isnan(corr_hr):
                    event_spo2_hr_corrs.append(corr_hr)

        if (
            len(spo2_for_resp_corr) > 2 and len(resp_for_spo2_corr) > 2
        ):  # Ensure we have enough pairs
            if np.std(spo2_for_resp_corr) > 0 and np.std(resp_for_spo2_corr) > 0:
                corr_resp = np.corrcoef(spo2_for_resp_corr, resp_for_spo2_corr)[0, 1]
                if not np.isnan(corr_resp):
                    event_spo2_resp_corrs.append(corr_resp)
    # Aggregate and write summary
    if report_file:
        report_file.write("\n[Event-Based HR/Resp Changes and Event Correlations]\n")

        def fmt(arr):
            return (
                f"mean={np.mean(arr):+.2f}, median={np.median(arr):+.2f}, std={np.std(arr):.2f}, n={len(arr)}"
                if arr
                else "N/A"
            )

        report_file.write(f"  HR change (during - before): {fmt(before_deltas_hr)}\n")
        report_file.write(f"  HR change (after - during): {fmt(after_deltas_hr)}\n")
        report_file.write(
            f"  Resp change (during - before): {fmt(before_deltas_resp)}\n"
        )
        report_file.write(f"  Resp change (after - during): {fmt(after_deltas_resp)}\n")
        report_file.write(f"  Event SpO2-HR corr: {fmt(event_spo2_hr_corrs)}\n")
        report_file.write(f"  Event SpO2-Resp corr: {fmt(event_spo2_resp_corrs)}\n\n")
    return {
        "before_deltas_hr": before_deltas_hr,
        "after_deltas_hr": after_deltas_hr,
        "before_deltas_resp": before_deltas_resp,
        "after_deltas_resp": after_deltas_resp,
        "event_spo2_hr_corrs": event_spo2_hr_corrs,
        "event_spo2_resp_corrs": event_spo2_resp_corrs,
    }


def main():
    if not os.path.exists(Config.DATA_FOLDER):
        print(f"Data folder '{Config.DATA_FOLDER}' not found.")
        return

    if not os.path.exists(Config.OUTPUT_FOLDER):
        os.makedirs(Config.OUTPUT_FOLDER)
    plots_dir = os.path.join(Config.OUTPUT_FOLDER, Config.PLOTS_SUBFOLDER)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    all_files_events = []
    all_spo2_readings_for_baseline = []
    all_sleep_stage_data = []
    accumulated_total_odi_events_from_all_days = 0
    daily_summary_for_trends = []
    all_inter_event_intervals_seconds = []
    # Initialize variables that will be calculated in the overall summary
    total_events = 0
    all_min_spo2 = []
    all_durations = []
    total_sleep_time_td_overall = (
        timedelta()
    )  # Ensure this is initialized for global use

    # New aggregators for SpO2 averages
    parsed_all_sleep_intervals_for_spo2_avg = []

    # New dictionaries to store HR and Resp data by date
    hr_data_by_date = {}
    resp_data_by_date = {}

    json_files = [f for f in os.listdir(Config.DATA_FOLDER) if f.endswith(".json")]
    json_files.sort()

    analysis_filepath = os.path.join(Config.OUTPUT_FOLDER, Config.ANALYSIS_FILE)
    with open(analysis_filepath, "w") as report_file:
        report_file.write("Sleep Data Low SpO2 Analysis Report\n")
        report_file.write(f"Low SpO2 Threshold: < {Config.LOW_SPO2_THRESHOLD}%\n")
        report_file.write("=" * 40 + "\n\n")

        for filename in json_files:
            filepath = os.path.join(Config.DATA_FOLDER, filename)
            print(f"Processing {filename}...")
            (
                calendar_date,
                daily_events,
                plot_data_pkg,
                daily_spo2_readings,
                daily_sleep_stages,
                daily_odi_event_count,
            ) = analyze_sleep_data(filepath)
            all_spo2_readings_for_baseline.extend(daily_spo2_readings)
            all_sleep_stage_data.extend(daily_sleep_stages)
            all_files_events.extend(daily_events)
            accumulated_total_odi_events_from_all_days += daily_odi_event_count

            # Store HR and Resp data for event-based correlation
            if calendar_date:
                hr_data_by_date[calendar_date] = plot_data_pkg.get("hr_data", [])
                resp_data_by_date[calendar_date] = plot_data_pkg.get("resp_data", [])

            # Parse and aggregate sleep intervals for later SpO2 average calculation
            for interval in daily_sleep_stages:
                start_dt = parse_gmt_timestamp_str(interval.get("startGMT"))
                end_dt = parse_gmt_timestamp_str(interval.get("endGMT"))
                activity = interval.get("activityLevel")
                if start_dt and end_dt and activity is not None:
                    parsed_all_sleep_intervals_for_spo2_avg.append(
                        (start_dt, end_dt, activity)
                    )

            # Collect data for temporal trends
            if calendar_date:
                num_daily_events = len(daily_events)
                avg_min_spo2_for_day = None
                if daily_events:
                    min_spo2s_today = [
                        e["min_spo2_in_event"]
                        for e in daily_events
                        if "min_spo2_in_event" in e
                    ]
                    if min_spo2s_today:
                        avg_min_spo2_for_day = np.mean(min_spo2s_today)
                daily_summary_for_trends.append(
                    {
                        "date": calendar_date,
                        "event_count": num_daily_events,
                        "avg_min_spo2": avg_min_spo2_for_day,
                    }
                )

            if calendar_date and daily_events:
                report_file.write(f"--- Analysis for Date: {calendar_date} ---\n")
                report_file.write(f"  Found {len(daily_events)} low SpO2 event(s).\n\n")
                for i, event in enumerate(daily_events):
                    write_event_to_report(report_file, event, i)
                    if event["duration_seconds"] >= Config.MIN_EVENT_DURATION_PLOT_SEC:
                        plot_path = plot_event_details(
                            event,
                            **plot_data_pkg,
                            calendar_date=calendar_date,
                            event_idx=i,
                            output_dir=plots_dir,
                        )
                        report_file.write(
                            f"      Plot generated: "
                            f"{os.path.basename(plot_path)}\n\n"
                        )
                report_file.write("\n")
            elif calendar_date:
                report_file.write(f"--- Analysis for Date: {calendar_date} ---\n")
                report_file.write("  No low SpO2 events detected.\n\n")

            # Corrected placement for daily inter-event interval calculation
            if daily_events and len(daily_events) > 1:
                # Sort events by start time to correctly calculate intervals
                sorted_daily_events = sorted(
                    daily_events, key=lambda e: e["event_start_time_utc"]
                )
                for i in range(len(sorted_daily_events) - 1):
                    event_end = sorted_daily_events[i]["event_end_time_utc"]
                    next_event_start = sorted_daily_events[i + 1][
                        "event_start_time_utc"
                    ]
                    interval_seconds = (next_event_start - event_end).total_seconds()
                    if interval_seconds > 0:  # Only positive intervals
                        all_inter_event_intervals_seconds.append(interval_seconds)

            # Correlation analysis and plots
            analyze_time_series_correlation(
                plot_data_pkg.get("spo2_data", []),
                plot_data_pkg.get("hr_data", []),
                plot_data_pkg.get("resp_data", []),
                calendar_date,
                plots_dir,
                report_file,
            )

            # --- Start of Reinserted Block for Daily Reporting ---
            # This block was previously part of the loop, after analyze_sleep_data and daily calculations.
            # It writes the daily summary to the report.

            # Note: The overall summary (total_events, all_min_spo2, etc.) is calculated *after* this loop.
            # Daily ODI (if applicable) - this was part of the original per-file logic.
            # For a meaningful daily ODI, we'd need daily sleep time.
            # The current `accumulated_total_odi_events_from_all_days` is for the *overall* ODI.
            # Let's report the number of ODI events for the day from `daily_odi_event_count`.
            if calendar_date:  # Ensure we have a date to report against
                report_file.write(
                    f"  Daily ODI Events (Desaturation >= {Config.ODI_DESATURATION_THRESHOLD}%): {daily_odi_event_count}\n"
                )

            # Daily Heart Rate Response patterns for events of THIS day
            if daily_events:
                daily_hr_changes_during = []
                daily_hr_changes_after = []
                for event in daily_events:  # Iterate only daily events
                    hr_before = event.get("hr_values_before_event", [])
                    hr_during = event.get("hr_values_in_event", [])
                    hr_after = event.get("hr_values_after_event", [])

                    if hr_before and hr_during:
                        avg_hr_before = np.mean(hr_before)
                        avg_hr_during = np.mean(hr_during)
                        daily_hr_changes_during.append(avg_hr_during - avg_hr_before)

                    if hr_during and hr_after:
                        avg_hr_during = np.mean(hr_during)
                        avg_hr_after = np.mean(hr_after)
                        daily_hr_changes_after.append(avg_hr_after - avg_hr_during)

                if daily_hr_changes_during:
                    avg_daily_hr_increase = np.mean(daily_hr_changes_during)
                    report_file.write(
                        f"  Average HR change during daily events: {avg_daily_hr_increase:+.1f} bpm\n"
                    )
                else:
                    report_file.write(f"  Average HR change during daily events: N/A\n")

                if daily_hr_changes_after:
                    avg_daily_hr_recovery = np.mean(daily_hr_changes_after)
                    report_file.write(
                        f"  Average HR change after daily events: {avg_daily_hr_recovery:+.1f} bpm\n"
                    )
                else:
                    report_file.write(f"  Average HR change after daily events: N/A\n")
                report_file.write(
                    "\n"
                )  # Add a newline after daily HR stats for the day
            # --- End of Reinserted Block for Daily Reporting ---

        # Call event-based correlation analysis AFTER the loop
        event_based_correlation_analysis(
            all_files_events, hr_data_by_date, resp_data_by_date, report_file
        )

        # Overall Dataset Summary - Calculations should happen *after* processing all files
        report_file.write("\n" + "=" * 40 + "\n")
        report_file.write("Overall Dataset Summary\n")
        report_file.write("=" * 40 + "\n")

        # Calculate overall sleep stage statistics first
        time_in_stage_overall = {
            0: timedelta(),
            1: timedelta(),
            2: timedelta(),
            3: timedelta(),
        }
        total_time_in_bed_td_overall = timedelta()
        sleep_efficiency_overall = 0.0  # Initialize as float

        if all_sleep_stage_data:
            first_timestamp_overall = None
            last_timestamp_overall = None
            for stage_interval in all_sleep_stage_data:
                start_str = stage_interval.get("startGMT")
                end_str = stage_interval.get("endGMT")
                activity_level = stage_interval.get("activityLevel")
                if (
                    start_str
                    and end_str
                    and activity_level is not None
                    and activity_level in time_in_stage_overall
                ):
                    start_time = parse_gmt_timestamp_str(start_str)
                    end_time = parse_gmt_timestamp_str(end_str)
                    if start_time and end_time:
                        duration = end_time - start_time
                        time_in_stage_overall[activity_level] += duration
                        if (
                            first_timestamp_overall is None
                            or start_time < first_timestamp_overall
                        ):
                            first_timestamp_overall = start_time
                        if (
                            last_timestamp_overall is None
                            or end_time > last_timestamp_overall
                        ):
                            last_timestamp_overall = end_time

            # total_sleep_time_td_overall was initialized at the start of main
            total_sleep_time_td_overall = (
                time_in_stage_overall[0]
                + time_in_stage_overall[1]
                + time_in_stage_overall[2]
            )
            if first_timestamp_overall and last_timestamp_overall:
                total_time_in_bed_td_overall = (
                    last_timestamp_overall - first_timestamp_overall
                )

            if total_time_in_bed_td_overall.total_seconds() > 0:
                sleep_efficiency_overall = (
                    total_sleep_time_td_overall.total_seconds()
                    / total_time_in_bed_td_overall.total_seconds()
                ) * 100
        # This is where the previous edit incorrectly placed the overall summary generation.
        # The overall summary generation will follow this block, using the aggregated data.

        # Now process event-related summaries using all_files_events
        if all_files_events:
            total_events = len(all_files_events)
            all_min_spo2 = [
                e["min_spo2_in_event"]
                for e in all_files_events
                if "min_spo2_in_event" in e and e["min_spo2_in_event"] is not None
            ]
            all_durations = [
                e["duration_seconds"]
                for e in all_files_events
                if "duration_seconds" in e
            ]
        else:
            total_events = 0
            all_min_spo2 = []
            all_durations = []

        # ODI Calculation uses total_sleep_time_td_overall and accumulated_total_odi_events_from_all_days
        report_file.write("\n# Overall Sleep and Event Statistics\n")
        report_file.write(f"Total Low SpO2 Events: {total_events}\n")

        if all_min_spo2:  # Check if list is not empty
            report_file.write(f"Avg Min SpO2 / Event: {np.mean(all_min_spo2):.1f}%\n")
            report_file.write(f"Overall Lowest SpO2: {min(all_min_spo2):.1f}%\n")
        else:
            report_file.write("Avg Min SpO2 / Event: N/A\n")
            report_file.write("Overall Lowest SpO2: N/A\n")

        if all_durations:  # Check if list is not empty
            mean_secs = np.mean(all_durations)
            max_secs = max(all_durations)
            avg_dur_td = timedelta(seconds=mean_secs)
            max_dur_td = timedelta(seconds=max_secs)
            report_file.write(
                f"Average Event Duration: " f"{format_timedelta(avg_dur_td)}\n"
            )
            report_file.write(
                f"Longest Event Duration: " f"{format_timedelta(max_dur_td)}\n"
            )
        else:
            report_file.write("Average Event Duration: N/A\n")
            report_file.write("Longest Event Duration: N/A\n")

        # Baseline SpO2 (ensure all_spo2_readings_for_baseline and all_files_events are used correctly)
        if all_spo2_readings_for_baseline:
            spo2_outside_events = []
            if all_files_events:  # Only filter if there are events to filter by
                for reading in all_spo2_readings_for_baseline:
                    is_in_event = False
                    for event in all_files_events:
                        if (
                            event["event_start_time_utc"]
                            <= reading["time"]
                            < event["event_end_time_utc"]
                        ):
                            is_in_event = True
                            break
                    if not is_in_event:
                        spo2_outside_events.append(reading["spo2"])
            else:  # No events, so all readings are outside events
                spo2_outside_events = [
                    r["spo2"] for r in all_spo2_readings_for_baseline
                ]

            if spo2_outside_events:
                baseline_spo2 = np.mean(spo2_outside_events)
                report_file.write(
                    f"Average Baseline SpO2 (outside events): {baseline_spo2:.1f}%\n"
                )
            else:
                report_file.write(
                    "Average Baseline SpO2 (outside events): N/A (no valid readings outside events)\n"
                )
        else:
            report_file.write(
                "Average Baseline SpO2 (outside events): N/A (no SpO2 readings)\n"
            )

        # Sleep Stage Statistics (time_in_stage_overall, total_sleep_time_td_overall already calculated)
        if all_sleep_stage_data:
            report_file.write(
                f"Total Time in Deep Sleep: {format_timedelta(time_in_stage_overall[0])}\n"
            )
            report_file.write(
                f"Total Time in Light Sleep: {format_timedelta(time_in_stage_overall[1])}\n"
            )
            report_file.write(
                f"Total Time in REM Sleep: {format_timedelta(time_in_stage_overall[2])}\n"
            )
            report_file.write(
                f"Total Time Awake: {format_timedelta(time_in_stage_overall[3])}\n"
            )
            report_file.write(
                f"Total Sleep Time (Deep+Light+REM): {format_timedelta(total_sleep_time_td_overall)}\n"
            )
            report_file.write(
                f"Total Time in Bed: {format_timedelta(total_time_in_bed_td_overall)}\n"
            )
            report_file.write(f"Sleep Efficiency: {sleep_efficiency_overall:.1f}%\n")
        else:
            report_file.write("Sleep Stage Statistics: N/A (no sleep stage data)\n")

        # Event frequency by sleep stage
        if all_files_events and all_sleep_stage_data:
            events_by_stage = {0: 0, 1: 0, 2: 0, 3: 0}
            for event in all_files_events:
                stage = event.get("dominant_sleep_stage_in_event")
                if stage is not None and stage in events_by_stage:
                    events_by_stage[stage] += 1
            report_file.write("\nEvent Frequency by Sleep Stage:\n")
            stage_map_report = {0: "Deep", 1: "Light", 2: "REM", 3: "Awake"}
            for stage_code, count in events_by_stage.items():
                stage_name = stage_map_report.get(stage_code)
                if time_in_stage_overall[stage_code].total_seconds() > 0:
                    total_hours_in_stage = (
                        time_in_stage_overall[stage_code].total_seconds() / 3600
                    )
                    rate_per_hour = count / total_hours_in_stage
                    report_file.write(
                        f"  {stage_name} Sleep: {count} events, Rate: {rate_per_hour:.2f} events/hour\n"
                    )
                else:
                    report_file.write(
                        f"  {stage_name} Sleep: {count} events, Rate: N/A (no time in stage)\n"
                    )
        elif all_files_events:
            report_file.write(
                "\nEvent Frequency by Sleep Stage: N/A (no sleep stage data for rates)\n"
            )
        else:
            report_file.write("\nEvent Frequency by Sleep Stage: N/A (no events)\n")

        # ODI Calculation
        if total_sleep_time_td_overall.total_seconds() > 0:
            total_sleep_hours = total_sleep_time_td_overall.total_seconds() / 3600
            if (
                total_sleep_hours > 0
            ):  # Avoid division by zero if TST is positive but rounds to 0 hours
                odi_value = (
                    accumulated_total_odi_events_from_all_days / total_sleep_hours
                )
                report_file.write(
                    f"Oxygen Desaturation Index (ODI >= {Config.ODI_DESATURATION_THRESHOLD}%): "
                    f"{odi_value:.1f} events/hour\n"
                )
            else:
                report_file.write(
                    f"Oxygen Desaturation Index (ODI >= {Config.ODI_DESATURATION_THRESHOLD}%): "
                    f"N/A (total sleep time is zero hours)\n"
                )
        else:
            report_file.write(
                f"Oxygen Desaturation Index (ODI >= {Config.ODI_DESATURATION_THRESHOLD}%): "
                f"N/A (no sleep time recorded for ODI calculation)\n"
            )

        # HR Response
        hr_changes_during_event = []
        hr_changes_after_event = []
        if all_files_events:  # Only calculate if there are events
            for event in all_files_events:
                hr_before = event.get("hr_values_before_event", [])
                hr_during = event.get("hr_values_in_event", [])
                hr_after = event.get("hr_values_after_event", [])
                if hr_before and hr_during:
                    avg_hr_before = np.mean(hr_before)
                    avg_hr_during = np.mean(hr_during)
                    hr_changes_during_event.append(avg_hr_during - avg_hr_before)
                if hr_during and hr_after:
                    avg_hr_during = np.mean(hr_during)
                    avg_hr_after = np.mean(hr_after)
                    hr_changes_after_event.append(avg_hr_after - avg_hr_during)

        if hr_changes_during_event:
            avg_hr_increase_during = np.mean(hr_changes_during_event)
            report_file.write(
                f"Average HR change during event (vs. pre-event): {avg_hr_increase_during:+.1f} bpm\n"
            )
        else:
            report_file.write(
                "Average HR change during event: N/A (insufficient data)\n"
            )
        if hr_changes_after_event:
            avg_hr_recovery_after = np.mean(hr_changes_after_event)
            report_file.write(
                f"Average HR change after event (vs. during event): {avg_hr_recovery_after:+.1f} bpm\n"
            )
        else:
            report_file.write(
                "Average HR change after event: N/A (insufficient data)\n"
            )

        # Event Clustering
        if all_inter_event_intervals_seconds:
            avg_inter_event_interval_seconds = np.mean(
                all_inter_event_intervals_seconds
            )
            avg_inter_event_interval_td = timedelta(
                seconds=avg_inter_event_interval_seconds
            )
            report_file.write(
                f"Average Inter-Event Interval (on nights with >1 event): "
                f"{format_timedelta(avg_inter_event_interval_td)}\n"
            )
        else:
            report_file.write(
                "Average Inter-Event Interval: N/A (no nights with multiple events or no intervals found)\n"
            )

        # Potential Respiratory Pauses
        if all_files_events and total_events > 0:
            events_with_pauses = sum(
                1
                for e in all_files_events
                if e.get("potential_resp_pauses_count", 0) > 0
            )
            percentage_events_with_pauses = (events_with_pauses / total_events) * 100
            report_file.write(
                f"Percentage of Low SpO2 Events with Potential Respiratory Pauses: {percentage_events_with_pauses:.1f}%\n"
            )
        elif (
            all_files_events
        ):  # total_events is 0 but all_files_events might exist (empty)
            report_file.write(
                "Percentage of Low SpO2 Events with Potential Respiratory Pauses: N/A (0 events to calculate percentage)\n"
            )
        else:  # No events at all
            report_file.write(
                "Percentage of Low SpO2 Events with Potential Respiratory Pauses: N/A (no events)\n"
            )

        # Plotting overall distributions (should only happen if there were events for most plots)
        if all_files_events:
            if all_min_spo2:
                plt.figure(figsize=(10, 6))
                plt.hist(all_min_spo2, bins=15, edgecolor="black", color="skyblue")
                plt.title("Distribution of Minimum SpO2 in Events")
                plt.xlabel("Minimum SpO2 (%) during Event")
                plt.ylabel("Number of Events")
                plt.grid(axis="y", alpha=0.75)
                plot_p = os.path.join(plots_dir, "_overall_min_spo2_distribution.png")
                plt.savefig(plot_p)
                plt.close()
                report_file.write(f"  Plot: {os.path.basename(plot_p)}\n")

            if all_durations:
                plt.figure(figsize=(10, 6))
                durations_mins = [d / 60 for d in all_durations]
                plt.hist(durations_mins, bins=15, edgecolor="black", color="salmon")
                plt.title("Distribution of Low SpO2 Event Durations")
                plt.xlabel("Event Duration (minutes)")
                plt.ylabel("Number of Events")
                plt.grid(axis="y", alpha=0.75)
                plot_p = os.path.join(
                    plots_dir, "_overall_event_duration_distribution.png"
                )
                plt.savefig(plot_p)
                plt.close()
                report_file.write(f"  Plot: {os.path.basename(plot_p)}\n")

            event_start_hours = [
                e["event_start_time_utc"].hour for e in all_files_events
            ]
            if event_start_hours:
                plt.figure(figsize=(10, 6))
                plt.hist(
                    event_start_hours,
                    bins=range(25),
                    edgecolor="black",
                    color="lightgreen",
                    align="left",
                )
                plt.title("Low SpO2 Events by Hour of Day (GMT)")
                plt.xlabel("Hour of Day (GMT)")
                plt.ylabel("Number of Events")
                plt.xticks(range(24))
                plt.grid(axis="y", alpha=0.75)
                plot_p = os.path.join(plots_dir, "_overall_events_by_hour.png")
                plt.savefig(plot_p)
                plt.close()
                report_file.write(f"  Plot: {os.path.basename(plot_p)}\n")
        else:  # No events, so plots related to events cannot be generated
            report_file.write("No low SpO2 events for overall distribution plots.\n")

        # Temporal Trend Plots (can be generated even if no events, if daily_summary_for_trends has data from days with no events)
        if daily_summary_for_trends:
            daily_summary_for_trends.sort(
                key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d")
            )
            dates = [
                datetime.strptime(d["date"], "%Y-%m-%d")
                for d in daily_summary_for_trends
            ]
            event_counts = [d["event_count"] for d in daily_summary_for_trends]
            avg_min_spo2s = [d["avg_min_spo2"] for d in daily_summary_for_trends]

            report_file.write("\nTemporal Trends Over Study Period:\n")
            plt.figure(figsize=(12, 6))
            plt.plot(dates, event_counts, marker="o", linestyle="-", color="purple")
            plt.title("Daily Low SpO2 Event Count Over Study Period")
            plt.xlabel("Date")
            plt.ylabel("Number of Low SpO2 Events")
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plot_p_trend_count = os.path.join(plots_dir, "_temporal_event_counts.png")
            plt.savefig(plot_p_trend_count)
            plt.close()
            report_file.write(
                f"  Plot (Daily Event Counts): {os.path.basename(plot_p_trend_count)}\n"
            )

            plottable_dates_spo2 = [
                dates[i] for i, v in enumerate(avg_min_spo2s) if v is not None
            ]
            plottable_avg_min_spo2s = [v for v in avg_min_spo2s if v is not None]
            if plottable_avg_min_spo2s:
                plt.figure(figsize=(12, 6))
                plt.plot(
                    plottable_dates_spo2,
                    plottable_avg_min_spo2s,
                    marker="s",
                    linestyle="-",
                    color="orange",
                )
                plt.title("Daily Average Minimum SpO2 in Events Over Study Period")
                plt.xlabel("Date")
                plt.ylabel("Average Minimum SpO2 (%)")
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                plt.xticks(rotation=45, ha="right")
                plt.grid(True, alpha=0.5)
                plt.tight_layout()
                plot_p_trend_spo2 = os.path.join(
                    plots_dir, "_temporal_avg_min_spo2.png"
                )
                plt.savefig(plot_p_trend_spo2)
                plt.close()
                report_file.write(
                    f"  Plot (Daily Avg Min SpO2): {os.path.basename(plot_p_trend_spo2)}\n"
                )
            else:
                report_file.write(
                    "  Plot (Daily Avg Min SpO2): Not generated (no days with events and SpO2 data).\n"
                )
        else:
            report_file.write("No data for temporal trend plots.\n")

        # --- New SpO2 Averages Calculation ---
        avg_spo2_overall = None
        avg_spo2_during_sleep = None
        avg_spo2_during_awake = None

        if all_spo2_readings_for_baseline:
            overall_spo2_sum = 0.0
            overall_spo2_count = 0
            sleep_spo2_sum = 0.0
            sleep_spo2_count = 0
            awake_spo2_sum = 0.0
            awake_spo2_count = 0

            SLEEP_ACTIVITY_LEVELS = {0, 1, 2}  # Deep, Light, REM
            AWAKE_ACTIVITY_LEVEL = 3

            # Sort sleep intervals for efficient lookup
            parsed_all_sleep_intervals_for_spo2_avg.sort(key=lambda x: x[0])

            # Sort spo2 readings by time for potentially optimized lookup (though list is already sorted from analyze_sleep_data)
            # all_spo2_readings_for_baseline.sort(key=lambda x: x['time']) # Already sorted by construction

            current_interval_idx = 0
            for (
                spo2_reading
            ) in (
                all_spo2_readings_for_baseline
            ):  # This list contains {'time': dt, 'spo2': val}
                reading_time = spo2_reading["time"]
                reading_spo2 = spo2_reading["spo2"]

                overall_spo2_sum += reading_spo2
                overall_spo2_count += 1

                activity_level_for_reading = None
                # Optimized search for activity level since both lists are sorted (or can be)
                # Iterate through sleep intervals to find the one covering the spo2_reading time
                # Start search from current_interval_idx to leverage sorted nature
                temp_search_idx = current_interval_idx
                while temp_search_idx < len(parsed_all_sleep_intervals_for_spo2_avg):
                    interval_start, interval_end, activity = (
                        parsed_all_sleep_intervals_for_spo2_avg[temp_search_idx]
                    )
                    if reading_time < interval_start:
                        # Reading is before this interval, it won't be in later ones if spo2 readings are sorted.
                        # If spo2 readings are not sorted, this break is not fully correct for optimization.
                        # However, current logic is that spo2 readings are globally sorted.
                        break
                    if interval_start <= reading_time < interval_end:
                        activity_level_for_reading = activity
                        current_interval_idx = (
                            temp_search_idx  # Update main index for next spo2 reading
                        )
                        break
                    if reading_time >= interval_end:
                        # This spo2_reading is past the current interval, advance search index
                        temp_search_idx += 1
                        current_interval_idx = (
                            temp_search_idx  # Keep main index in sync
                        )
                    else:  # Should ideally not be reached if spo2_reading times are strictly increasing
                        # and intervals are contiguous or properly gapped.
                        break

                if activity_level_for_reading is not None:
                    if activity_level_for_reading in SLEEP_ACTIVITY_LEVELS:
                        sleep_spo2_sum += reading_spo2
                        sleep_spo2_count += 1
                    elif activity_level_for_reading == AWAKE_ACTIVITY_LEVEL:
                        awake_spo2_sum += reading_spo2
                        awake_spo2_count += 1

            if overall_spo2_count > 0:
                avg_spo2_overall = overall_spo2_sum / overall_spo2_count
            if sleep_spo2_count > 0:
                avg_spo2_during_sleep = sleep_spo2_sum / sleep_spo2_count
            if awake_spo2_count > 0:
                avg_spo2_during_awake = awake_spo2_sum / awake_spo2_count

        report_file.write(
            f"Average SpO2 (All Valid Readings): {avg_spo2_overall:.1f}%\n"
            if avg_spo2_overall is not None
            else "Average SpO2 (All Valid Readings): N/A\n"
        )
        report_file.write(
            f"Average SpO2 (During Sleep Periods): {avg_spo2_during_sleep:.1f}%\n"
            if avg_spo2_during_sleep is not None
            else "Average SpO2 (During Sleep Periods): N/A\n"
        )
        report_file.write(
            f"Average SpO2 (During Awake Periods): {avg_spo2_during_awake:.1f}%\n"
            if avg_spo2_during_awake is not None
            else "Average SpO2 (During Awake Periods): N/A\n"
        )
        # --- End New SpO2 Averages Calculation ---

    print(f"\nAnalysis complete. Report saved to: {analysis_filepath}")
    print(f"Plots saved in: {plots_dir}")


if __name__ == "__main__":
    main()
