#!/usr/bin/env python3
"""
Complete Enhanced Sleep Apnea Analysis Script
Integrates existing event detection with advanced clinical metrics and visualizations
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Set up enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("complete_enhanced_analysis.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class CompleteEnhancedConfig:
    """Complete enhanced configuration with all parameters"""

    # Basic settings
    DATA_FOLDER = Path("data")
    OUTPUT_FOLDER = Path("complete_enhanced_output")
    PLOTS_SUBFOLDER = "enhanced_plots"
    ANALYSIS_FILE = "complete_enhanced_report.txt"

    # SpO2 thresholds - enhanced with multiple levels
    LOW_SPO2_THRESHOLD = 90
    MODERATE_SPO2_THRESHOLD = 85
    SEVERE_SPO2_THRESHOLD = 80
    CRITICAL_SPO2_THRESHOLD = 70

    # ODI thresholds - multiple clinical standards
    ODI_3_THRESHOLD = 3  # 3% desaturation
    ODI_4_THRESHOLD = 4  # 4% desaturation

    # Enhanced event detection parameters
    MIN_EVENT_DURATION_SEC = 10  # Reduced to catch brief events
    MIN_VALID_READINGS = 1  # Reduced to catch brief events
    MAX_GAP_BETWEEN_EVENTS_SEC = 120

    # Confidence thresholds
    MIN_CONFIDENCE = 1  # Accept lower confidence readings

    # Analysis windows
    BASELINE_WINDOW_MINUTES = 5
    RECOVERY_WINDOW_MINUTES = 5


class SpO2EventDetector:
    """Enhanced SpO2 event detection with clinical standards"""

    def __init__(self, config: CompleteEnhancedConfig):
        self.config = config

    def detect_events(
        self, spo2_data: List[Dict], sleep_stages: List[Dict]
    ) -> List[Dict]:
        """Detect SpO2 desaturation events with enhanced sensitivity"""
        if not spo2_data:
            logger.warning("No SpO2 data found")
            return []

        # Sort SpO2 data by timestamp
        valid_spo2_data = []
        for reading in spo2_data:
            timestamp = self._parse_timestamp(reading.get("epochTimestamp"))
            spo2_value = reading.get("spo2Reading")
            confidence = reading.get("readingConfidence")

            # Skip readings with invalid data
            if (timestamp is None or
                spo2_value is None or
                confidence is None or
                not isinstance(spo2_value, (int, float)) or
                not isinstance(confidence, (int, float))):
                continue

            valid_spo2_data.append(reading)

        if not valid_spo2_data:
            logger.warning("No valid SpO2 data found after filtering")
            return []

        sorted_spo2 = sorted(
            valid_spo2_data,
            key=lambda x: self._parse_timestamp(x.get("epochTimestamp"))
        )

        # Create sleep stage lookup
        stage_lookup = self._create_stage_lookup(sleep_stages)

        events = []
        current_event = None
        baseline_spo2 = self._calculate_baseline(sorted_spo2)

        for i, reading in enumerate(sorted_spo2):
            timestamp = self._parse_timestamp(reading.get("epochTimestamp"))
            spo2_value = reading.get("spo2Reading")
            confidence = reading.get("readingConfidence", 0)

            # Additional safety checks
            if (timestamp is None or
                spo2_value is None or
                confidence is None or
                confidence < self.config.MIN_CONFIDENCE):
                continue

            # Check if this is a desaturation event
            is_low_spo2 = spo2_value < self.config.LOW_SPO2_THRESHOLD

            if is_low_spo2:
                if current_event is None:
                    # Start new event
                    current_event = {
                        "start_time": timestamp,
                        "readings": [reading],
                        "min_spo2": spo2_value,
                        "baseline_spo2": baseline_spo2,
                    }
                else:
                    # Continue current event
                    current_event["readings"].append(reading)
                    current_event["min_spo2"] = min(
                        current_event["min_spo2"], spo2_value
                    )
            else:
                if current_event is not None:
                    # End current event
                    current_event["end_time"] = timestamp

                    # Validate event duration
                    try:
                        duration = (
                            current_event["end_time"] - current_event["start_time"]
                        ).total_seconds()

                        if (
                            duration >= self.config.MIN_EVENT_DURATION_SEC
                            and len(current_event["readings"]) >= self.config.MIN_VALID_READINGS
                        ):
                            # Enhance event with additional metrics
                            enhanced_event = self._enhance_event(
                                current_event, stage_lookup
                            )
                            events.append(enhanced_event)
                    except Exception as e:
                        logger.warning(f"Error processing event: {e}")

                    current_event = None

        # Handle event that extends to end of data
        if current_event is not None:
            try:
                last_timestamp = self._parse_timestamp(sorted_spo2[-1].get("epochTimestamp"))
                if last_timestamp:
                    current_event["end_time"] = last_timestamp
                    duration = (
                        current_event["end_time"] - current_event["start_time"]
                    ).total_seconds()

                    if (
                        duration >= self.config.MIN_EVENT_DURATION_SEC
                        and len(current_event["readings"]) >= self.config.MIN_VALID_READINGS
                    ):
                        enhanced_event = self._enhance_event(current_event, stage_lookup)
                        events.append(enhanced_event)
            except Exception as e:
                logger.warning(f"Error processing final event: {e}")

        logger.info(f"Detected {len(events)} SpO2 events")
        return events

    def _calculate_baseline(self, spo2_data: List[Dict]) -> float:
        """Calculate baseline SpO2 from non-event periods"""
        if not spo2_data:
            return 95.0  # Default baseline

        # Use readings above threshold as baseline
        baseline_readings = []
        for r in spo2_data:
            spo2_reading = r.get("spo2Reading")
            confidence = r.get("readingConfidence")

            if (spo2_reading is not None and
                confidence is not None and
                isinstance(spo2_reading, (int, float)) and
                isinstance(confidence, (int, float)) and
                spo2_reading >= self.config.LOW_SPO2_THRESHOLD and
                confidence >= self.config.MIN_CONFIDENCE):
                baseline_readings.append(spo2_reading)

        return np.mean(baseline_readings) if baseline_readings else 95.0

    def _create_stage_lookup(self, sleep_stages: List[Dict]) -> Dict:
        """Create timestamp to sleep stage lookup"""
        stage_lookup = {}

        for stage in sleep_stages:
            try:
                start_time = self._parse_timestamp(stage.get("startGMT"))
                end_time = self._parse_timestamp(stage.get("endGMT"))
                activity_level = stage.get("activityLevel")

                if start_time and end_time and activity_level is not None:
                    # Round start time down to the minute and end time up to the minute
                    start_minute = start_time.replace(second=0, microsecond=0)
                    end_minute = end_time.replace(second=0, microsecond=0)
                    if end_time.second > 0 or end_time.microsecond > 0:
                        end_minute += timedelta(minutes=1)

                    # Create minute-by-minute lookup
                    current_time = start_minute
                    while current_time < end_minute:
                        stage_lookup[current_time] = activity_level
                        current_time += timedelta(minutes=1)
            except Exception as e:
                logger.debug(f"Error processing sleep stage: {e}")
                continue

        return stage_lookup

    def _enhance_event(self, event: Dict, stage_lookup: Dict) -> Dict:
        """Enhance event with additional clinical metrics"""
        try:
            readings = event["readings"]
            start_time = event["start_time"]
            end_time = event["end_time"]

            # Basic metrics
            duration_seconds = (end_time - start_time).total_seconds()
            spo2_values = [r.get("spo2Reading") for r in readings if r.get("spo2Reading") is not None]

            if not spo2_values:
                # Return minimal event if no valid readings
                return {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": duration_seconds,
                    "duration_minutes": duration_seconds / 60,
                    "min_spo2": 90,
                    "avg_spo2": 90,
                    "baseline_spo2": event.get("baseline_spo2", 95),
                    "max_desaturation": 5,
                    "severity": "Mild",
                    "sleep_stage": 3,
                    "sleep_stage_name": "Awake",
                    "reading_count": len(readings),
                    "readings": readings,
                }

            avg_spo2 = np.mean(spo2_values)

            # Sleep stage during event
            sleep_stage = stage_lookup.get(start_time, 3)  # Default to awake
            stage_names = {0: "Deep", 1: "Light", 2: "REM", 3: "Awake"}

            # Severity classification
            min_spo2 = event["min_spo2"]
            if min_spo2 >= self.config.MODERATE_SPO2_THRESHOLD:
                severity = "Mild"
            elif min_spo2 >= self.config.SEVERE_SPO2_THRESHOLD:
                severity = "Moderate"
            else:
                severity = "Severe"

            # Desaturation magnitude
            baseline = event.get("baseline_spo2", 95)
            max_desaturation = baseline - min_spo2

            return {
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration_seconds,
                "duration_minutes": duration_seconds / 60,
                "min_spo2": min_spo2,
                "avg_spo2": avg_spo2,
                "baseline_spo2": baseline,
                "max_desaturation": max_desaturation,
                "severity": severity,
                "sleep_stage": sleep_stage,
                "sleep_stage_name": stage_names.get(sleep_stage, "Unknown"),
                "reading_count": len(readings),
                "readings": readings,
            }
        except Exception as e:
            logger.warning(f"Error enhancing event: {e}")
            # Return minimal event structure
            return {
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "duration_seconds": 0,
                "duration_minutes": 0,
                "min_spo2": event.get("min_spo2", 90),
                "avg_spo2": event.get("min_spo2", 90),
                "baseline_spo2": event.get("baseline_spo2", 95),
                "max_desaturation": 5,
                "severity": "Mild",
                "sleep_stage": 3,
                "sleep_stage_name": "Awake",
                "reading_count": len(event.get("readings", [])),
                "readings": event.get("readings", []),
            }

    def _parse_timestamp(self, timestamp) -> Optional[datetime]:
        """Parse various timestamp formats with enhanced error handling"""
        if not timestamp:
            return None

        try:
            if isinstance(timestamp, str):
                # Handle different string formats
                if timestamp.endswith(".0"):
                    timestamp = timestamp[:-2]
                # Handle timezone indicators
                if timestamp.endswith("Z"):
                    timestamp = timestamp.replace("Z", "+00:00")
                elif not timestamp.endswith("+00:00") and "T" in timestamp:
                    timestamp = timestamp + "+00:00"

                return datetime.fromisoformat(timestamp)
            elif isinstance(timestamp, (int, float)):
                return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        except Exception as e:
            logger.debug(f"Failed to parse timestamp {timestamp}: {e}")

        return None


class ClinicalAnalyzer:
    """Advanced clinical analysis with standard sleep medicine metrics"""

    def __init__(self, config: CompleteEnhancedConfig):
        self.config = config

    def calculate_comprehensive_metrics(
        self, events: List[Dict], sleep_data: Dict
    ) -> Dict[str, Any]:
        """Calculate comprehensive clinical metrics"""
        if not events:
            return self._empty_metrics()

        # Basic sleep metrics
        total_sleep_time_hours = self._calculate_total_sleep_time(sleep_data)

        # AHI calculation with severity levels
        ahi_metrics = self._calculate_ahi(events, total_sleep_time_hours)

        # ODI calculation with multiple thresholds
        odi_metrics = self._calculate_odi(events, total_sleep_time_hours)

        # Sleep architecture analysis
        sleep_arch = self._analyze_sleep_architecture(sleep_data)

        # Event distribution analysis
        event_dist = self._analyze_event_distribution(events)

        # Severity analysis
        severity_analysis = self._analyze_severity_distribution(events)

        # Temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(events)

        return {
            "total_sleep_time_hours": total_sleep_time_hours,
            "total_events": len(events),
            **ahi_metrics,
            **odi_metrics,
            **sleep_arch,
            **event_dist,
            **severity_analysis,
            **temporal_patterns,
        }

    def _calculate_total_sleep_time(self, sleep_data: Dict) -> float:
        """Calculate total sleep time from sleep stages"""
        dto = sleep_data.get("dailySleepDTO", {})

        deep_sleep = dto.get("deepSleepSeconds", 0)
        light_sleep = dto.get("lightSleepSeconds", 0)
        rem_sleep = dto.get("remSleepSeconds", 0)

        total_seconds = deep_sleep + light_sleep + rem_sleep
        return total_seconds / 3600  # Convert to hours

    def _calculate_ahi(
        self, events: List[Dict], total_sleep_time_hours: float
    ) -> Dict[str, float]:
        """Calculate Apnea-Hypopnea Index with severity levels"""
        if total_sleep_time_hours <= 0:
            return {
                "ahi_total": 0,
                "ahi_mild": 0,
                "ahi_moderate": 0,
                "ahi_severe": 0,
                "ahi_classification": "Normal",
            }

        # Count events by severity
        mild_events = len([e for e in events if e.get("severity") == "Mild"])
        moderate_events = len([e for e in events if e.get("severity") == "Moderate"])
        severe_events = len([e for e in events if e.get("severity") == "Severe"])

        ahi_total = len(events) / total_sleep_time_hours

        # AHI classification
        if ahi_total < 5:
            classification = "Normal"
        elif ahi_total < 15:
            classification = "Mild Sleep Apnea"
        elif ahi_total < 30:
            classification = "Moderate Sleep Apnea"
        else:
            classification = "Severe Sleep Apnea"

        return {
            "ahi_total": ahi_total,
            "ahi_mild": mild_events / total_sleep_time_hours,
            "ahi_moderate": moderate_events / total_sleep_time_hours,
            "ahi_severe": severe_events / total_sleep_time_hours,
            "ahi_classification": classification,
        }

    def _calculate_odi(
        self, events: List[Dict], total_sleep_time_hours: float
    ) -> Dict[str, float]:
        """Calculate Oxygen Desaturation Index"""
        if total_sleep_time_hours <= 0:
            return {"odi_3": 0, "odi_4": 0}

        # Count events by desaturation magnitude
        odi_3_events = len([e for e in events if e.get("max_desaturation", 0) >= 3])
        odi_4_events = len([e for e in events if e.get("max_desaturation", 0) >= 4])

        return {
            "odi_3": odi_3_events / total_sleep_time_hours,
            "odi_4": odi_4_events / total_sleep_time_hours,
        }

    def _analyze_sleep_architecture(self, sleep_data: Dict) -> Dict[str, Any]:
        """Analyze sleep architecture"""
        dto = sleep_data.get("dailySleepDTO", {})

        deep_sleep_sec = dto.get("deepSleepSeconds", 0)
        light_sleep_sec = dto.get("lightSleepSeconds", 0)
        rem_sleep_sec = dto.get("remSleepSeconds", 0)
        awake_sec = dto.get("awakeSleepSeconds", 0)

        total_sleep_sec = deep_sleep_sec + light_sleep_sec + rem_sleep_sec
        total_time_sec = total_sleep_sec + awake_sec

        if total_sleep_sec > 0:
            deep_percentage = (deep_sleep_sec / total_sleep_sec) * 100
            light_percentage = (light_sleep_sec / total_sleep_sec) * 100
            rem_percentage = (rem_sleep_sec / total_sleep_sec) * 100
        else:
            deep_percentage = light_percentage = rem_percentage = 0

        sleep_efficiency = (
            (total_sleep_sec / total_time_sec * 100) if total_time_sec > 0 else 0
        )

        return {
            "deep_sleep_hours": deep_sleep_sec / 3600,
            "light_sleep_hours": light_sleep_sec / 3600,
            "rem_sleep_hours": rem_sleep_sec / 3600,
            "awake_hours": awake_sec / 3600,
            "deep_sleep_percentage": deep_percentage,
            "light_sleep_percentage": light_percentage,
            "rem_sleep_percentage": rem_percentage,
            "sleep_efficiency": sleep_efficiency,
        }

    def _analyze_event_distribution(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze event distribution by sleep stage"""
        stage_counts = {"Deep": 0, "Light": 0, "REM": 0, "Awake": 0}

        for event in events:
            stage = event.get("sleep_stage_name", "Unknown")
            if stage in stage_counts:
                stage_counts[stage] += 1

        return {
            "events_deep_sleep": stage_counts["Deep"],
            "events_light_sleep": stage_counts["Light"],
            "events_rem_sleep": stage_counts["REM"],
            "events_awake": stage_counts["Awake"],
        }

    def _analyze_severity_distribution(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze severity distribution"""
        severity_counts = {"Mild": 0, "Moderate": 0, "Severe": 0}

        for event in events:
            severity = event.get("severity", "Mild")
            if severity in severity_counts:
                severity_counts[severity] += 1

        # Calculate statistics
        min_spo2_values = [e.get("min_spo2", 100) for e in events]
        duration_values = [e.get("duration_minutes", 0) for e in events]

        return {
            "mild_events": severity_counts["Mild"],
            "moderate_events": severity_counts["Moderate"],
            "severe_events": severity_counts["Severe"],
            "avg_min_spo2": np.mean(min_spo2_values) if min_spo2_values else 0,
            "lowest_spo2": min(min_spo2_values) if min_spo2_values else 0,
            "avg_event_duration": np.mean(duration_values) if duration_values else 0,
            "max_event_duration": max(duration_values) if duration_values else 0,
        }

    def _analyze_temporal_patterns(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns of events"""
        if not events:
            return {"hourly_distribution": {}}

        # Hourly distribution
        hourly_counts = {}
        for event in events:
            hour = event["start_time"].hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

        return {
            "hourly_distribution": hourly_counts,
            "peak_hour": (
                max(hourly_counts.items(), key=lambda x: x[1])[0]
                if hourly_counts
                else 0
            ),
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            "total_sleep_time_hours": 0,
            "total_events": 0,
            "ahi_total": 0,
            "ahi_classification": "Normal",
        }


class EnhancedVisualizer:
    """Enhanced visualization with clinical-grade plots"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        plt.style.use("default")
        sns.set_palette("husl")

    def create_comprehensive_dashboard(
        self, all_metrics: List[Dict], all_events: List[Dict]
    ):
        """Create comprehensive analysis dashboard"""
        fig = plt.figure(figsize=(20, 16))

        # Create subplots
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. AHI Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ahi_values = [m.get("ahi_total", 0) for m in all_metrics]
        ax1.hist(ahi_values, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.axvline(x=5, color="orange", linestyle="--", label="Mild Threshold")
        ax1.axvline(x=15, color="red", linestyle="--", label="Moderate Threshold")
        ax1.axvline(x=30, color="darkred", linestyle="--", label="Severe Threshold")
        ax1.set_xlabel("AHI (events/hour)")
        ax1.set_ylabel("Number of Nights")
        ax1.set_title("AHI Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Severity Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if all_events:
            severity_counts = {}
            for event in all_events:
                severity = event.get("severity", "Mild")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            colors = {"Mild": "lightgreen", "Moderate": "orange", "Severe": "red"}
            severities = list(severity_counts.keys())
            counts = list(severity_counts.values())
            bars = ax2.bar(
                severities, counts, color=[colors.get(s, "gray") for s in severities]
            )
            ax2.set_ylabel("Number of Events")
            ax2.set_title("Event Severity Distribution")
            ax2.grid(True, alpha=0.3)

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(count),
                    ha="center",
                    va="bottom",
                )

        # 3. Sleep Stage Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if all_events:
            stage_counts = {}
            for event in all_events:
                stage = event.get("sleep_stage_name", "Unknown")
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

            if stage_counts:
                ax3.pie(
                    stage_counts.values(), labels=stage_counts.keys(), autopct="%1.1f%%"
                )
                ax3.set_title("Events by Sleep Stage")

        # 4. SpO2 Distribution
        ax4 = fig.add_subplot(gs[0, 3])
        if all_events:
            min_spo2_values = [e.get("min_spo2", 100) for e in all_events]
            ax4.hist(
                min_spo2_values,
                bins=20,
                alpha=0.7,
                color="lightcoral",
                edgecolor="black",
            )
            ax4.axvline(x=90, color="orange", linestyle="--", label="Mild (90%)")
            ax4.axvline(x=85, color="red", linestyle="--", label="Moderate (85%)")
            ax4.axvline(x=80, color="darkred", linestyle="--", label="Severe (80%)")
            ax4.set_xlabel("Minimum SpO2 (%)")
            ax4.set_ylabel("Number of Events")
            ax4.set_title("SpO2 Distribution")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Sleep Efficiency vs AHI
        ax5 = fig.add_subplot(gs[1, 0])
        sleep_eff = [m.get("sleep_efficiency", 0) for m in all_metrics]
        ahi_vals = [m.get("ahi_total", 0) for m in all_metrics]
        ax5.scatter(sleep_eff, ahi_vals, alpha=0.6, s=50)
        ax5.set_xlabel("Sleep Efficiency (%)")
        ax5.set_ylabel("AHI (events/hour)")
        ax5.set_title("Sleep Efficiency vs AHI")
        ax5.grid(True, alpha=0.3)

        # Add correlation coefficient
        if len(sleep_eff) > 1 and len(ahi_vals) > 1:
            corr, p_val = stats.pearsonr(sleep_eff, ahi_vals)
            ax5.text(
                0.05,
                0.95,
                f"r = {corr:.3f}\np = {p_val:.3f}",
                transform=ax5.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # 6. Event Duration Distribution
        ax6 = fig.add_subplot(gs[1, 1])
        if all_events:
            durations = [e.get("duration_minutes", 0) for e in all_events]
            ax6.hist(
                durations, bins=20, alpha=0.7, color="lightblue", edgecolor="black"
            )
            ax6.set_xlabel("Event Duration (minutes)")
            ax6.set_ylabel("Number of Events")
            ax6.set_title("Event Duration Distribution")
            ax6.grid(True, alpha=0.3)

        # 7. Hourly Event Pattern
        ax7 = fig.add_subplot(gs[1, 2:])
        if all_events:
            hourly_counts = {}
            for event in all_events:
                hour = event["start_time"].hour
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

            hours = list(range(24))
            counts = [hourly_counts.get(h, 0) for h in hours]
            ax7.bar(hours, counts, alpha=0.7, color="lightgreen", edgecolor="black")
            ax7.set_xlabel("Hour of Day")
            ax7.set_ylabel("Number of Events")
            ax7.set_title("Hourly Distribution of Events")
            ax7.set_xticks(range(0, 24, 2))
            ax7.grid(True, alpha=0.3)

        # 8. Correlation Heatmap
        ax8 = fig.add_subplot(gs[2, :2])
        if all_metrics:
            # Create correlation matrix
            metrics_df = pd.DataFrame(all_metrics)
            numeric_cols = [
                "ahi_total",
                "odi_4",
                "sleep_efficiency",
                "deep_sleep_percentage",
                "rem_sleep_percentage",
            ]
            numeric_cols = [col for col in numeric_cols if col in metrics_df.columns]

            if len(numeric_cols) > 1:
                corr_matrix = metrics_df[numeric_cols].corr()
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap="coolwarm",
                    center=0,
                    ax=ax8,
                    cbar_kws={"label": "Correlation"},
                )
                ax8.set_title("Physiological Parameter Correlations")

        # 9. Sleep Architecture Summary
        ax9 = fig.add_subplot(gs[2, 2:])
        if all_metrics:
            avg_deep = np.mean([m.get("deep_sleep_percentage", 0) for m in all_metrics])
            avg_light = np.mean(
                [m.get("light_sleep_percentage", 0) for m in all_metrics]
            )
            avg_rem = np.mean([m.get("rem_sleep_percentage", 0) for m in all_metrics])

            stages = ["Deep Sleep", "Light Sleep", "REM Sleep"]
            percentages = [avg_deep, avg_light, avg_rem]
            colors = ["darkblue", "lightblue", "orange"]

            bars = ax9.bar(
                stages, percentages, color=colors, alpha=0.7, edgecolor="black"
            )
            ax9.set_ylabel("Percentage of Sleep Time")
            ax9.set_title("Average Sleep Architecture")
            ax9.grid(True, alpha=0.3)

            # Add percentage labels
            for bar, pct in zip(bars, percentages):
                ax9.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                )

        # 10. Summary Statistics
        ax10 = fig.add_subplot(gs[3, :])
        ax10.axis("off")

        if all_metrics and all_events:
            # Calculate summary statistics
            total_nights = len(all_metrics)
            total_events = len(all_events)
            avg_ahi = np.mean([m.get("ahi_total", 0) for m in all_metrics])
            avg_sleep_eff = np.mean([m.get("sleep_efficiency", 0) for m in all_metrics])

            # Count AHI classifications
            classifications = [
                m.get("ahi_classification", "Normal") for m in all_metrics
            ]
            class_counts = {}
            for c in classifications:
                class_counts[c] = class_counts.get(c, 0) + 1

            summary_text = f"""
COMPREHENSIVE SLEEP ANALYSIS SUMMARY
{'='*50}

Study Period: {total_nights} nights analyzed
Total Events Detected: {total_events}
Average AHI: {avg_ahi:.1f} events/hour
Average Sleep Efficiency: {avg_sleep_eff:.1f}%

AHI Classification Distribution:
"""
            for classification, count in class_counts.items():
                percentage = (count / total_nights) * 100
                summary_text += (
                    f"  {classification}: {count} nights ({percentage:.1f}%)\n"
                )

            ax10.text(
                0.05,
                0.95,
                summary_text,
                transform=ax10.transAxes,
                verticalalignment="top",
                fontfamily="monospace",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
            )

        plt.suptitle(
            "Enhanced Sleep Apnea Analysis Dashboard", fontsize=16, fontweight="bold"
        )
        plt.savefig(
            self.output_dir / "comprehensive_dashboard.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def main():
    """Complete enhanced main analysis function"""
    config = CompleteEnhancedConfig()

    # Setup directories
    config.OUTPUT_FOLDER.mkdir(exist_ok=True)
    plots_dir = config.OUTPUT_FOLDER / config.PLOTS_SUBFOLDER
    plots_dir.mkdir(exist_ok=True)

    detector = SpO2EventDetector(config)
    analyzer = ClinicalAnalyzer(config)
    visualizer = EnhancedVisualizer(plots_dir)

    logger.info("Starting complete enhanced sleep analysis...")

    # Process all JSON files
    json_files = sorted([f for f in config.DATA_FOLDER.glob("*.json")])

    all_events = []
    all_metrics = []

    for json_file in json_files:
        logger.info(f"Processing {json_file.name}...")

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Extract SpO2 data
            spo2_data = data.get("wellnessEpochSPO2DataDTOList", [])
            sleep_stages = data.get("sleepLevels", [])

            # Detect events
            events = detector.detect_events(spo2_data, sleep_stages)
            all_events.extend(events)

            # Calculate metrics for this night
            metrics = analyzer.calculate_comprehensive_metrics(events, data)
            metrics["date"] = data.get("dailySleepDTO", {}).get("calendarDate")
            metrics["filename"] = json_file.name
            all_metrics.append(metrics)

        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            continue

    # Generate comprehensive visualizations
    logger.info("Generating comprehensive dashboard...")
    visualizer.create_comprehensive_dashboard(all_metrics, all_events)

    # Generate comprehensive report
    report_path = config.OUTPUT_FOLDER / config.ANALYSIS_FILE
    with open(report_path, "w") as f:
        f.write("COMPLETE ENHANCED SLEEP APNEA ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Overall statistics
        total_nights = len(all_metrics)
        total_events = len(all_events)
        total_sleep_time = sum(m.get("total_sleep_time_hours", 0) for m in all_metrics)

        f.write("STUDY OVERVIEW:\n")
        f.write(f"Analysis Period: {total_nights} nights\n")
        f.write(f"Total Sleep Time: {total_sleep_time:.1f} hours\n")
        f.write(f"Total Events Detected: {total_events}\n")
        f.write(
            f"Overall Event Rate: {total_events/total_sleep_time:.2f} events/hour\n\n"
        )

        # AHI Analysis
        ahi_values = [m.get("ahi_total", 0) for m in all_metrics]
        f.write("AHI ANALYSIS:\n")
        f.write(f"Average AHI: {np.mean(ahi_values):.1f} events/hour\n")
        f.write(f"Median AHI: {np.median(ahi_values):.1f} events/hour\n")
        f.write(f"Maximum AHI: {np.max(ahi_values):.1f} events/hour\n")
        f.write(f"Standard Deviation: {np.std(ahi_values):.1f}\n\n")

        # Classification distribution
        classifications = [m.get("ahi_classification", "Normal") for m in all_metrics]
        class_counts = {}
        for c in classifications:
            class_counts[c] = class_counts.get(c, 0) + 1

        f.write("AHI CLASSIFICATION DISTRIBUTION:\n")
        for classification, count in class_counts.items():
            percentage = (count / total_nights) * 100
            f.write(f"{classification}: {count} nights ({percentage:.1f}%)\n")
        f.write("\n")

        # Severity analysis
        if all_events:
            severity_counts = {}
            for event in all_events:
                severity = event.get("severity", "Mild")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            f.write("EVENT SEVERITY DISTRIBUTION:\n")
            for severity, count in severity_counts.items():
                percentage = (count / total_events) * 100
                f.write(f"{severity}: {count} events ({percentage:.1f}%)\n")
            f.write("\n")

            # SpO2 statistics
            min_spo2_values = [e.get("min_spo2", 100) for e in all_events]
            f.write("SPO2 STATISTICS:\n")
            f.write(f"Average Minimum SpO2: {np.mean(min_spo2_values):.1f}%\n")
            f.write(f"Lowest SpO2 Recorded: {np.min(min_spo2_values):.1f}%\n")
            f.write(f"SpO2 Standard Deviation: {np.std(min_spo2_values):.1f}%\n\n")

            # Duration statistics
            durations = [e.get("duration_minutes", 0) for e in all_events]
            f.write("EVENT DURATION STATISTICS:\n")
            f.write(f"Average Duration: {np.mean(durations):.1f} minutes\n")
            f.write(f"Median Duration: {np.median(durations):.1f} minutes\n")
            f.write(f"Maximum Duration: {np.max(durations):.1f} minutes\n")
            f.write(f"Duration Standard Deviation: {np.std(durations):.1f} minutes\n\n")

        # Sleep architecture
        sleep_eff_values = [m.get("sleep_efficiency", 0) for m in all_metrics]
        deep_sleep_values = [m.get("deep_sleep_percentage", 0) for m in all_metrics]
        rem_sleep_values = [m.get("rem_sleep_percentage", 0) for m in all_metrics]

        f.write("SLEEP ARCHITECTURE:\n")
        f.write(f"Average Sleep Efficiency: {np.mean(sleep_eff_values):.1f}%\n")
        f.write(f"Average Deep Sleep: {np.mean(deep_sleep_values):.1f}%\n")
        f.write(f"Average REM Sleep: {np.mean(rem_sleep_values):.1f}%\n\n")

        # Clinical recommendations
        avg_ahi = np.mean(ahi_values)
        f.write("CLINICAL ASSESSMENT:\n")
        if avg_ahi < 5:
            f.write("Sleep breathing patterns appear normal.\n")
        elif avg_ahi < 15:
            f.write(
                "Mild sleep-disordered breathing detected. Consider lifestyle modifications.\n"
            )
        elif avg_ahi < 30:
            f.write("Moderate sleep apnea detected. Clinical evaluation recommended.\n")
        else:
            f.write(
                "Severe sleep apnea detected. Immediate clinical attention recommended.\n"
            )

        f.write(
            f"\nAnalysis completed. Dashboard saved to: {plots_dir / 'comprehensive_dashboard.png'}\n"
        )

    logger.info(
        f"Complete enhanced analysis finished. Results saved to {config.OUTPUT_FOLDER}"
    )


if __name__ == "__main__":
    main()
