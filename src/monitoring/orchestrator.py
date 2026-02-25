"""
Metric Orchestrator — coordinates execution of all four metrics
with configurable scheduling. Central entry point for the monitoring system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.config import AppConfig, load_config, DATA_DIR
from src.production.logging import LogStore
from src.monitoring.data_collection import DataCollectionLayer, MetricDataset
from src.monitoring.metrics.base import MetricResult
from src.monitoring.metrics.trci import TRCIEngine
from src.monitoring.metrics.cds import CDSEngine
from src.monitoring.metrics.fds import FDSEngine
from src.monitoring.metrics.ddi import DDIEngine


@dataclass
class MonitoringReport:
    """Report from a monitoring run."""
    run_type: str  # "daily" | "weekly" | "on_demand"
    timestamp: str
    results: list[MetricResult] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_type": self.run_type,
            "timestamp": self.timestamp,
            "results": [
                {
                    "metric_name": r.metric_name,
                    "value": r.value,
                    "status": r.status,
                    "explanation": r.explanation,
                    "details": r.details,
                }
                for r in self.results
            ],
            "alerts": self.alerts,
        }


class MetricOrchestrator:
    """
    Coordinates execution of all four metrics.
    This is the central entry point for the monitoring system.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        log_store: Optional[LogStore] = None,
        pipeline=None,
    ):
        self.config = config or load_config()
        self.log_store = log_store or LogStore(self.config.db_path)
        self.data_layer = DataCollectionLayer(self.log_store)
        self.trci = TRCIEngine(self.config.trci, pipeline=pipeline)
        self.cds = CDSEngine(self.config.cds)
        self.fds = FDSEngine(self.config.fds)
        self.ddi = DDIEngine(self.config.ddi)
        self.results_dir = DATA_DIR / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_daily(self) -> MonitoringReport:
        """
        Daily monitoring run — TRCI (canary probe) + CDS (distribution drift).
        """
        results: list[MetricResult] = []
        now = datetime.now(timezone.utc)

        # TRCI: canary probe
        trci_result = self.trci.run_probe()
        results.append(trci_result)

        # CDS: reference vs current window
        ref_dataset = self.data_layer.collect_window(
            now - timedelta(days=self.config.cds.reference_window_days), now
        )
        cur_dataset = self.data_layer.collect_window(
            now - timedelta(days=self.config.cds.current_window_days), now
        )

        if ref_dataset.descriptors and cur_dataset.descriptors:
            cds_result = self.cds.compute(ref_dataset.descriptors, cur_dataset.descriptors)
            results.append(cds_result)

        # Build report
        alerts = self._evaluate_alerts(results)
        report = MonitoringReport(
            run_type="daily",
            timestamp=now.isoformat(),
            results=results,
            alerts=alerts,
        )

        self._save_report(report)
        return report

    def run_weekly(self) -> MonitoringReport:
        """
        Weekly monitoring run — all four metrics.
        """
        results: list[MetricResult] = []
        now = datetime.now(timezone.utc)

        # TRCI + CDS from daily
        trci_result = self.trci.run_probe()
        results.append(trci_result)

        ref_dataset = self.data_layer.collect_window(
            now - timedelta(days=self.config.cds.reference_window_days), now
        )
        cur_dataset = self.data_layer.collect_window(
            now - timedelta(days=self.config.cds.current_window_days), now
        )

        if ref_dataset.descriptors and cur_dataset.descriptors:
            cds_result = self.cds.compute(ref_dataset.descriptors, cur_dataset.descriptors)
            results.append(cds_result)

        # FDS: faithfulness check on weekly window
        weekly_window = self.data_layer.collect_window(
            now - timedelta(weeks=1), now
        )
        if weekly_window.logs:
            # Use a synthetic reference baseline (in production, loaded from stored baseline)
            ref_faithfulness = np.array([0.85] * 50)  # placeholder baseline
            fds_result = self.fds.compute(weekly_window.logs, ref_faithfulness)
            results.append(fds_result)

        # DDI: fairness across segments
        if ref_dataset.logs and cur_dataset.logs:
            ddi_result = self.ddi.compute(ref_dataset, cur_dataset)
            results.append(ddi_result)

        alerts = self._evaluate_alerts(results)
        report = MonitoringReport(
            run_type="weekly",
            timestamp=now.isoformat(),
            results=results,
            alerts=alerts,
        )

        self._save_report(report)
        return report

    def run_single_metric(self, metric_name: str, **kwargs) -> MetricResult:
        """Run a single metric on demand."""
        now = datetime.now(timezone.utc)

        if metric_name.upper() == "TRCI":
            return self.trci.run_probe()

        elif metric_name.upper() == "CDS":
            ref = self.data_layer.collect_window(
                now - timedelta(days=self.config.cds.reference_window_days), now
            )
            cur = self.data_layer.collect_window(
                now - timedelta(days=self.config.cds.current_window_days), now
            )
            if ref.descriptors and cur.descriptors:
                return self.cds.compute(ref.descriptors, cur.descriptors)
            return MetricResult(
                metric_name="CDS",
                timestamp=now.isoformat(),
                value=0.0,
                status="GREEN",
                explanation="Insufficient data for CDS computation.",
            )

        elif metric_name.upper() == "FDS":
            window = self.data_layer.collect_window(
                now - timedelta(weeks=1), now
            )
            ref_faithfulness = kwargs.get(
                "reference_faithfulness", np.array([0.85] * 50)
            )
            if window.logs:
                return self.fds.compute(window.logs, ref_faithfulness)
            return MetricResult(
                metric_name="FDS",
                timestamp=now.isoformat(),
                value=0.0,
                status="GREEN",
                explanation="Insufficient data for FDS computation.",
            )

        elif metric_name.upper() == "DDI":
            ref = self.data_layer.collect_window(
                now - timedelta(days=self.config.cds.reference_window_days), now
            )
            cur = self.data_layer.collect_window(
                now - timedelta(days=self.config.cds.current_window_days), now
            )
            if ref.logs and cur.logs:
                return self.ddi.compute(ref, cur)
            return MetricResult(
                metric_name="DDI",
                timestamp=now.isoformat(),
                value=0.0,
                status="GREEN",
                explanation="Insufficient data for DDI computation.",
            )

        raise ValueError(f"Unknown metric: {metric_name}")

    def _evaluate_alerts(self, results: list[MetricResult]) -> list[dict]:
        """Evaluate alert conditions for all results."""
        alerts = []
        for r in results:
            if r.status == "RED":
                alerts.append({
                    "severity": "CRITICAL",
                    "metric": r.metric_name,
                    "value": r.value,
                    "status": r.status,
                    "explanation": r.explanation,
                    "timestamp": r.timestamp,
                })
            elif r.status == "AMBER":
                alerts.append({
                    "severity": "WARNING",
                    "metric": r.metric_name,
                    "value": r.value,
                    "status": r.status,
                    "explanation": r.explanation,
                    "timestamp": r.timestamp,
                })
        return alerts

    def _save_report(self, report: MonitoringReport) -> None:
        """Save monitoring report as JSON."""
        filename = f"{report.run_type}_{report.timestamp.replace(':', '-')}.json"
        filepath = self.results_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
