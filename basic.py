import os, requests, json
webhook = "https://hooks.slack.com/services/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"
resp = requests.post(webhook, json={"text":"Hello from Python test!"}, timeout=10)
print(resp.status_code, resp.text)
