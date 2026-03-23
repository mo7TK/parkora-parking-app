import { useEffect, useRef, useState } from "react";
import { FlatList, StyleSheet, Text, View } from "react-native";

// ── Configuration ─────────────────────────────────────────────────────────────
const BACKEND_IP = "192.168.1.36";
const WS_URL = `ws://${BACKEND_IP}:8000/ws`;
// ─────────────────────────────────────────────────────────────────────────────

// This is the shape of one spot as it arrives from the backend
type Spot = {
  id: number;
  status: "free" | "occupied";
};

export default function MiniMap() {
  // spots holds the current status of all 12 parking spots
  // We start with an empty array — the WebSocket will fill it
  const [spots, setSpots] = useState<Spot[]>([]);

  // connectionStatus lets the user know what's happening
  const [connectionStatus, setConnectionStatus] = useState<
    "connecting" | "connected" | "disconnected"
  >("connecting");

  // useRef stores the WebSocket instance without causing re-renders
  // We need this to close the connection when the screen unmounts
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // This function opens the WebSocket and sets up all event handlers
    function connect() {
      setConnectionStatus("connecting");
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      // Called once the connection is established
      ws.onopen = () => {
        setConnectionStatus("connected");
        console.log("WebSocket connected");
      };

      // Called every time the backend sends a message
      // The message is a JSON string like:
      // { "spots": [{"id": 1, "status": "free"}, {"id": 2, "status": "occupied"}, ...] }
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setSpots(data.spots);
      };

      // Called when the connection is lost (backend stopped, WiFi issue, etc.)
      ws.onclose = () => {
        setConnectionStatus("disconnected");
        console.log("WebSocket disconnected, retrying in 3s...");
        // Try to reconnect after 3 seconds
        setTimeout(connect, 3000);
      };

      // Called on connection errors
      ws.onerror = (error) => {
        console.log("WebSocket error:", error);
        ws.close();
      };
    }

    connect();

    // Cleanup: close the WebSocket when the screen unmounts
    // Without this, old connections would pile up
    return () => {
      wsRef.current?.close();
    };
  }, []); // empty array = run once when the screen first loads

  // ── Render each spot as a colored box ──────────────────────────────────────
  const renderSpot = ({ item }: { item: Spot }) => {
    const isFree = item.status === "free";
    return (
      <View
        style={[styles.spot, isFree ? styles.spotFree : styles.spotOccupied]}
      >
        <Text style={styles.spotNumber}>{item.id}</Text>
        <Text style={styles.spotLabel}>{isFree ? "Free" : "Occupied"}</Text>
      </View>
    );
  };

  // ── Connection status indicator color ──────────────────────────────────────
  const statusColor = {
    connecting: "#f0a500",
    connected: "#2ecc71",
    disconnected: "#e74c3c",
  }[connectionStatus];

  const freeCount = spots.filter((s) => s.status === "free").length;
  const occupiedCount = spots.filter((s) => s.status === "occupied").length;

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Parking Mini </Text>
        <View style={styles.statusRow}>
          <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
          <Text style={styles.statusText}>{connectionStatus}</Text>
        </View>
      </View>

      {/* Summary */}
      {spots.length > 0 && (
        <View style={styles.summary}>
          <View style={styles.summaryItem}>
            <View style={[styles.summaryDot, styles.spotFree]} />
            <Text style={styles.summaryText}>Free: {freeCount}</Text>
          </View>
          <View style={styles.summaryItem}>
            <View style={[styles.summaryDot, styles.spotOccupied]} />
            <Text style={styles.summaryText}>Occupied: {occupiedCount}</Text>
          </View>
        </View>
      )}

      {/* Waiting message before first data arrives */}
      {spots.length === 0 && (
        <View style={styles.waiting}>
          <Text style={styles.waitingText}>
            Waiting for detection data...{"\n"}Make sure detect.py is running.
          </Text>
        </View>
      )}

      {/* The 12 parking spot boxes in a 3-column grid */}
      <FlatList
        data={spots}
        keyExtractor={(item) => item.id.toString()}
        renderItem={renderSpot}
        numColumns={3}
        columnWrapperStyle={styles.row}
        contentContainerStyle={styles.grid}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f4f4f4",
    paddingTop: 60,
    paddingHorizontal: 16,
  },

  // ── Header ────────────────────────────────────────────────────────────────
  header: {
    marginBottom: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: "600",
    color: "#1a1a2e",
    marginBottom: 6,
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  statusText: {
    fontSize: 13,
    color: "#555",
    textTransform: "capitalize",
  },

  // ── Summary ───────────────────────────────────────────────────────────────
  summary: {
    flexDirection: "row",
    gap: 20,
    marginBottom: 20,
  },
  summaryItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  summaryDot: {
    width: 14,
    height: 14,
    borderRadius: 3,
  },
  summaryText: {
    fontSize: 15,
    color: "#333",
    fontWeight: "500",
  },

  // ── Waiting state ─────────────────────────────────────────────────────────
  waiting: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  waitingText: {
    fontSize: 15,
    color: "#888",
    textAlign: "center",
    lineHeight: 24,
  },

  // ── Grid ──────────────────────────────────────────────────────────────────
  grid: {
    paddingBottom: 40,
  },
  spot: {
    height: 100,
    width: "30%",
    margin: 6,
    borderRadius: 10,
    justifyContent: "center",
    alignItems: "center",
  },
  row: {
    justifyContent: "flex-start",
  },

  spotFree: {
    backgroundColor: "#2ecc71",
  },
  spotOccupied: {
    backgroundColor: "#e74c3c",
  },
  spotNumber: {
    fontSize: 22,
    fontWeight: "700",
    color: "#fff",
  },
  spotLabel: {
    fontSize: 11,
    color: "rgba(255,255,255,0.85)",
    marginTop: 2,
  },
});
