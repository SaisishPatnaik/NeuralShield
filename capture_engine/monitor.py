
import threading
import queue
from datetime import datetime, timezone
from collections import Counter

from scapy.all import sniff, IP, TCP, UDP, ICMP


class NetworkMonitor:

    def __init__(self):

        self.running = False
        self.stop_event = threading.Event()
        self.capture_thread = None

        self.interface = None

        self.events = queue.Queue(maxsize=1000)

        self.lock = threading.Lock()

        self.stats = {
            "packets_captured": 0,
            "bytes_captured": 0,
            "protocols": {},
            "source_ips": 0,
            "destination_ips": 0
        }

        self.source_ips = set()
        self.destination_ips = set()

    def process_packet(self, packet):

        if IP not in packet:
            return

        ip = packet[IP]

        protocol = "OTHER"
        source_port = None
        destination_port = None

        if TCP in packet:

            protocol = "TCP"
            source_port = int(packet[TCP].sport)
            destination_port = int(packet[TCP].dport)

        elif UDP in packet:

            protocol = "UDP"
            source_port = int(packet[UDP].sport)
            destination_port = int(packet[UDP].dport)

        elif ICMP in packet:

            protocol = "ICMP"

        packet_size = len(packet)

        event = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "source_ip": ip.src,
            "destination_ip": ip.dst,

            "protocol": protocol,

            "source_port": source_port,
            "destination_port": destination_port,

            "packet_size": packet_size
        }

        with self.lock:

            self.stats["packets_captured"] += 1

            self.stats["bytes_captured"] += packet_size

            self.source_ips.add(ip.src)
            self.destination_ips.add(ip.dst)

            self.stats["source_ips"] = len(self.source_ips)
            self.stats["destination_ips"] = len(
                self.destination_ips
            )

            protocols = Counter(self.stats["protocols"])
            protocols[protocol] += 1

            self.stats["protocols"] = dict(protocols)

        # Keep only metadata, never packet payload.
        try:
            self.events.put_nowait(event)

        except queue.Full:
            pass

    def capture_loop(self):

        while not self.stop_event.is_set():

            try:

                sniff(
                    iface=self.interface,
                    filter="ip",
                    prn=self.process_packet,
                    store=False,
                    timeout=1
                )

            except Exception as error:

                with self.lock:
                    self.capture_error = str(error)

                self.stop_event.set()
                break

        self.running = False

    def start(self, interface=None):

        if self.running:
            return False

        self.interface = interface
        self.stop_event.clear()

        self.capture_error = None

        with self.lock:

            self.stats = {
                "packets_captured": 0,
                "bytes_captured": 0,
                "protocols": {},
                "source_ips": 0,
                "destination_ips": 0
            }

            self.source_ips.clear()
            self.destination_ips.clear()

        while not self.events.empty():

            try:
                self.events.get_nowait()
            except queue.Empty:
                break

        self.running = True

        self.capture_thread = threading.Thread(
            target=self.capture_loop,
            daemon=True
        )

        self.capture_thread.start()

        return True

    def stop(self):

        if not self.running:
            return False

        self.stop_event.set()

        if self.capture_thread:
            self.capture_thread.join(timeout=3)

        self.running = False

        return True

    def get_status(self):

        with self.lock:

            return {
                "running": self.running,
                "interface": self.interface,
                "capture_error": getattr(
                    self, "capture_error", None
                ),
                **self.stats
            }

    def get_event(self, timeout=1):

        try:
            return self.events.get(timeout=timeout)

        except queue.Empty:
            return None


network_monitor = NetworkMonitor()
