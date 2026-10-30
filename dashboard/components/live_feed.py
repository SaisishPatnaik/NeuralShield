import json
import queue
import threading
import time

import websocket


WS_URL = "ws://127.0.0.1:8000/ws/events"

event_queue = queue.Queue(maxsize=1000)
connection_state = {"connected": False}
connection_lock = threading.Lock()
ws_thread = None


def on_open(ws):
    with connection_lock:
        connection_state["connected"] = True


def on_message(ws, message):
    try:
        event = json.loads(message)

        if event.get("type") == "heartbeat":
            return

        try:
            event_queue.put_nowait(event)
        except queue.Full:
            pass

    except json.JSONDecodeError:
        pass


def on_error(ws, error):
    with connection_lock:
        connection_state["connected"] = False


def on_close(ws, close_status_code, close_message):
    with connection_lock:
        connection_state["connected"] = False


def websocket_worker():
    while True:
        ws = websocket.WebSocketApp(
            WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        try:
            ws.run_forever(
                ping_interval=20,
                ping_timeout=10
            )
        except Exception:
            with connection_lock:
                connection_state["connected"] = False

        time.sleep(3)


def start_websocket():
    global ws_thread

    if ws_thread is None or not ws_thread.is_alive():
        ws_thread = threading.Thread(
            target=websocket_worker,
            daemon=True
        )
        ws_thread.start()


def get_connection_status():
    with connection_lock:
        return connection_state["connected"]


def get_live_events():
    events = []

    while True:
        try:
            events.append(event_queue.get_nowait())
        except queue.Empty:
            break

    return events