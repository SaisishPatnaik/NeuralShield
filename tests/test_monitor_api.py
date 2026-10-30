
import time
import requests

API_URL = "http://127.0.0.1:8000"


print("\n========== NEURALSHIELD LIVE MONITOR TEST ==========")

# Start monitoring
print("\nStarting network capture...")

response = requests.post(
    f"{API_URL}/monitor/start",
    json={"interface": None},
    timeout=10
)

print(response.json())

time.sleep(10)

# Check status
print("\nFetching monitoring statistics...")

response = requests.get(
    f"{API_URL}/monitor/status",
    timeout=10
)

status = response.json()

print("\n========== CAPTURE STATISTICS ==========")

for key, value in status.items():
    print(f"{key}: {value}")

# Stop monitoring
print("\nStopping network capture...")

response = requests.post(
    f"{API_URL}/monitor/stop",
    timeout=10
)

print(response.json())

print("\nMONITOR TEST FINISHED!")
