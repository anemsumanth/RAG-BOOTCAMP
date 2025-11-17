# splunk_tool.py
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import splunklib.client as client
import splunklib.results as results


@dataclass
class SplunkConfig:
    host: str = "localhost"
    port: int = 8000           # Splunk’s default management port
    username: str = "admin"
    password: str = "changeme"
    scheme: str = "https"
    validate: bool = False     # Set to True for prod to avoid self‑signed cert warnings


class SplunkTool:
    """
    A thin wrapper around Splunk SDK that can be plugged into a RAG pipeline.
    """

    def __init__(self, cfg: SplunkConfig):
        self.cfg = cfg
        self.service = client.connect(
            host=cfg.host,
            port=cfg.port,
            username=cfg.username,
            password=cfg.password,
            scheme=cfg.scheme,
            validate=cfg.validate,
        )

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Submit a search job and return results as list of dicts.
        :param query: SPL search string (e.g., `index=main sourcetype=access_combined | head 10`)
        :param kwargs: Optional search job parameters: `earliest_time`, `latest_time`,
                       `timeout`, `output_mode`, etc.
        :return: List of result rows as Python dicts
        """
        # 1. Create a search job
        job = self.service.jobs.create(
            query,
            **kwargs
        )
        # 2. Wait for it to finish
        while not job.is_done():
            job.refresh()
        # 3. Fetch results
        # Use 'json' output for easy parsing
        results_reader = results.ResultsReader(
            job.results(output_mode="json")
        )
        output = []
        for result in results_reader:
            if isinstance(result, dict):
                output.append(result)
        # 4. Clean up the job
        job.delete()
        return output

    def query_with_retry(self, query: str,
                         retries: int = 3,
                         backoff: float = 2.0,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Simple retry wrapper – useful if you hit rate limits or network hiccups.
        """
        last_exception = None
        for i in range(retries):
            try:
                return self.search(query, **kwargs)
            except Exception as e:
                last_exception = e
                time.sleep(backoff * (i + 1))
        raise last_exception

    def __repr__(self):
        return f"<SplunkTool host={self.cfg.host} port={self.cfg.port}>"
    
    from splunk_tool import SplunkConfig, SplunkTool

cfg = SplunkConfig(
    host="localhost",
    port=8000,
    username="admin",
    password="changeme",
    scheme="https",
    validate=False,
)
splunk = SplunkTool(cfg)

results = splunk.search("index=main | head 5")
print(json.dumps(results, indent=2))