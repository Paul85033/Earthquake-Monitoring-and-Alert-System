// Seismic AI Detector Dashboard JavaScript

let magnitudeChart, confidenceChart, map;
let detectionMarkers = [];
let predictionOverlays = [];
let showDetections = true;
let showPredictions = true;

// Initialize map
function initMap() {
  // Center on USA
  map = L.map("map").setView([39.8283, -98.5795], 4);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap contributors",
    maxZoom: 18,
  }).addTo(map);

  // Add toggle controls
  document
    .getElementById("toggleDetections")
    .addEventListener("click", function () {
      showDetections = !showDetections;
      this.classList.toggle("active");
      updateMapLayers();
    });

  document
    .getElementById("togglePredictions")
    .addEventListener("click", function () {
      showPredictions = !showPredictions;
      this.classList.toggle("active");
      updateMapLayers();
    });
}

// Update map layer visibility
function updateMapLayers() {
  // Toggle detection markers
  detectionMarkers.forEach((marker) => {
    if (showDetections) {
      marker.addTo(map);
    } else {
      map.removeLayer(marker);
    }
  });

  // Toggle prediction overlays
  predictionOverlays.forEach((overlay) => {
    if (showPredictions) {
      overlay.addTo(map);
    } else {
      map.removeLayer(overlay);
    }
  });
}

// Add earthquake detection marker
function addEarthquakeMarker(event) {
  if (!event.epicenter_lat || !event.epicenter_lon) {
    return;
  }

  const lat = event.epicenter_lat;
  const lon = event.epicenter_lon;
  const mag = event.magnitude;
  const prob = event.location_probability || 0.5;
  const distance = event.distance_km || 0;
  const depth = event.depth_km || 0;

  // Determine color based on probability
  let color;
  if (prob >= 0.7) color = "#ff4444";
  else if (prob >= 0.5) color = "#ffaa00";
  else color = "#44ff44";

  // Circle size based on magnitude
  const radius = Math.pow(2, mag) * 1000;

  // Create circle marker
  const circle = L.circle([lat, lon], {
    color: color,
    fillColor: color,
    fillOpacity: 0.5,
    radius: radius,
    weight: 2,
  });

  if (showDetections) {
    circle.addTo(map);
  }

  // Create popup
  const popupContent = `
        <div style="min-width: 200px;">
            <h3 style="margin: 0 0 10px 0;">🚨 DETECTED Magnitude ${mag.toFixed(
              1
            )}</h3>
            <p style="margin: 5px 0;"><strong>Time:</strong> ${new Date(
              event.timestamp
            ).toLocaleString()}</p>
            <p style="margin: 5px 0;"><strong>Location:</strong> ${lat.toFixed(
              4
            )}°, ${lon.toFixed(4)}°</p>
            <p style="margin: 5px 0;"><strong>Distance:</strong> ${distance.toFixed(
              1
            )} km</p>
            <p style="margin: 5px 0;"><strong>Depth:</strong> ${depth.toFixed(
              1
            )} km</p>
            <p style="margin: 5px 0;"><strong>Probability:</strong> ${(
              prob * 100
            ).toFixed(0)}%</p>
            <p style="margin: 5px 0;"><strong>Confidence:</strong> ${(
              event.confidence * 100
            ).toFixed(0)}%</p>
            ${
              event.nearest_zone
                ? `<p style="margin: 5px 0;"><strong>Near:</strong> ${event.nearest_zone}</p>`
                : ""
            }
        </div>
    `;

  circle.bindPopup(popupContent);

  detectionMarkers.push(circle);

  // Limit to 50 markers
  if (detectionMarkers.length > 50) {
    const oldMarker = detectionMarkers.shift();
    map.removeLayer(oldMarker);
  }
}

// Add prediction overlay
function addPredictionOverlay(prediction) {
  const lat = prediction.center_lat;
  const lon = prediction.center_lon;
  const radius = prediction.radius_km * 1000; // Convert to meters

  // Determine color and style based on risk level
  let color, fillOpacity;
  if (prediction.risk_level === "high") {
    color = "#ff4444";
    fillOpacity = 0.3;
  } else if (prediction.risk_level === "elevated") {
    color = "#ffaa00";
    fillOpacity = 0.25;
  } else if (prediction.risk_level === "moderate") {
    color = "#4444ff";
    fillOpacity = 0.2;
  } else {
    color = "#44ff44";
    fillOpacity = 0.15;
  }

  // Create circle overlay with dashed border
  const circle = L.circle([lat, lon], {
    color: color,
    fillColor: color,
    fillOpacity: fillOpacity,
    radius: radius,
    weight: 2,
    dashArray: "10, 5",
    opacity: 0.8,
  });

  if (showPredictions) {
    circle.addTo(map);
  }

  // Create popup with prediction details
  const magRange = prediction.magnitude_range;
  const popupContent = `
        <div style="min-width: 250px;">
            <h3 style="margin: 0 0 10px 0;">🔮 PREDICTION: ${
              prediction.region_name
            }</h3>
            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <p style="margin: 5px 0;"><strong>Risk Level:</strong> <span class="risk-${
                  prediction.risk_level
                }">${prediction.risk_level.toUpperCase()}</span></p>
                <p style="margin: 5px 0;"><strong>24h Probability:</strong> ${(
                  prediction.probability_24h * 100
                ).toFixed(1)}%</p>
                <p style="margin: 5px 0;"><strong>7d Probability:</strong> ${(
                  prediction.probability_7d * 100
                ).toFixed(1)}%</p>
                <p style="margin: 5px 0;"><strong>30d Probability:</strong> ${(
                  prediction.probability_30d * 100
                ).toFixed(1)}%</p>
            </div>
            <p style="margin: 5px 0;"><strong>Est. Magnitude:</strong> ${magRange[0].toFixed(
              1
            )} - ${magRange[1].toFixed(1)}</p>
            <p style="margin: 5px 0;"><strong>Confidence:</strong> ${(
              prediction.confidence * 100
            ).toFixed(0)}%</p>
            <p style="margin: 5px 0;"><strong>Fault Zones:</strong> ${prediction.fault_zones.join(
              ", "
            )}</p>
            <p style="margin: 5px 0;"><strong>Setting:</strong> ${
              prediction.tectonic_setting
            }</p>
            <p style="margin: 5px 0; font-size: 0.85em; color: #666;"><strong>Last Updated:</strong> ${new Date(
              prediction.last_updated
            ).toLocaleString()}</p>
        </div>
    `;

  circle.bindPopup(popupContent);

  // Add label
  const label = L.marker([lat, lon], {
    icon: L.divIcon({
      className: "prediction-label",
      html: `<div style="background: white; padding: 3px 8px; border-radius: 4px; border: 2px solid ${color}; font-weight: bold; font-size: 11px;">${prediction.region_name}</div>`,
      iconSize: [0, 0],
    }),
  });

  if (showPredictions) {
    label.addTo(map);
  }

  predictionOverlays.push(circle);
  predictionOverlays.push(label);
}

// Update predictions
async function updatePredictions() {
  try {
    const response = await fetch("/api/predictions");
    const predictions = await response.json();

    // Clear old prediction overlays
    predictionOverlays.forEach((overlay) => map.removeLayer(overlay));
    predictionOverlays = [];

    // Add new predictions
    predictions.forEach((prediction) => {
      addPredictionOverlay(prediction);
    });

    // Update predictions table
    updatePredictionsTable(predictions);
  } catch (error) {
    console.error("Error updating predictions:", error);
  }
}

// Update predictions table
function updatePredictionsTable(predictions) {
  const tbody = document.getElementById("predictionsTableBody");

  if (predictions.length === 0) {
    tbody.innerHTML =
      '<tr><td colspan="7" class="loading">No predictions available</td></tr>';
    return;
  }

  tbody.innerHTML = predictions
    .map((pred) => {
      const magRange = pred.magnitude_range;

      return `
            <tr>
                <td><strong>${pred.region_name}</strong></td>
                <td><span class="risk-${
                  pred.risk_level
                }">${pred.risk_level.toUpperCase()}</span></td>
                <td>${(pred.probability_24h * 100).toFixed(1)}%</td>
                <td>${(pred.probability_7d * 100).toFixed(1)}%</td>
                <td>${(pred.probability_30d * 100).toFixed(1)}%</td>
                <td>${magRange[0].toFixed(1)} - ${magRange[1].toFixed(1)}</td>
                <td>${(pred.confidence * 100).toFixed(0)}%</td>
            </tr>
        `;
    })
    .join("");
}

// Initialize charts
function initCharts() {
  // Magnitude timeline chart
  const magnitudeCtx = document
    .getElementById("magnitudeChart")
    .getContext("2d");
  magnitudeChart = new Chart(magnitudeCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Magnitude",
          data: [],
          borderColor: "#667eea",
          backgroundColor: "rgba(102, 126, 234, 0.1)",
          tension: 0.4,
          fill: true,
          pointRadius: 5,
          pointHoverRadius: 7,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          display: true,
          position: "top",
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Magnitude",
          },
        },
        x: {
          title: {
            display: true,
            text: "Time",
          },
        },
      },
    },
  });

  // Confidence distribution chart
  const confidenceCtx = document
    .getElementById("confidenceChart")
    .getContext("2d");
  confidenceChart = new Chart(confidenceCtx, {
    type: "bar",
    data: {
      labels: ["0-25%", "25-50%", "50-75%", "75-100%"],
      datasets: [
        {
          label: "Number of Events",
          data: [0, 0, 0, 0],
          backgroundColor: [
            "rgba(255, 99, 132, 0.8)",
            "rgba(255, 206, 86, 0.8)",
            "rgba(75, 192, 192, 0.8)",
            "rgba(54, 162, 235, 0.8)",
          ],
          borderColor: [
            "rgba(255, 99, 132, 1)",
            "rgba(255, 206, 86, 1)",
            "rgba(75, 192, 192, 1)",
            "rgba(54, 162, 235, 1)",
          ],
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          display: false,
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            stepSize: 1,
          },
          title: {
            display: true,
            text: "Count",
          },
        },
        x: {
          title: {
            display: true,
            text: "Confidence Range",
          },
        },
      },
    },
  });
}

// Update statistics
async function updateStatistics() {
  try {
    const response = await fetch("/api/stats");
    const stats = await response.json();

    document.getElementById("totalEvents").textContent =
      stats.total_events || 0;
    document.getElementById("avgMagnitude").textContent = stats.avg_magnitude
      ? stats.avg_magnitude.toFixed(1)
      : "-";
    document.getElementById("maxMagnitude").textContent = stats.max_magnitude
      ? stats.max_magnitude.toFixed(1)
      : "-";

    // Today's events
    const todayResponse = await fetch("/api/events/today");
    const todayEvents = await todayResponse.json();
    document.getElementById("todayEvents").textContent = todayEvents.length;
  } catch (error) {
    console.error("Error updating statistics:", error);
  }
}

// Update timeline chart
async function updateTimeline() {
  try {
    const response = await fetch("/api/timeline");
    const timeline = await response.json();

    // Update magnitude chart
    const labels = timeline.map((e) => {
      const date = new Date(e.time);
      return date.toLocaleTimeString();
    });
    const magnitudes = timeline.map((e) => e.magnitude);

    magnitudeChart.data.labels = labels;
    magnitudeChart.data.datasets[0].data = magnitudes;
    magnitudeChart.update();

    // Update confidence distribution
    const confidenceBins = [0, 0, 0, 0];
    timeline.forEach((e) => {
      const conf = e.confidence * 100;
      if (conf < 25) confidenceBins[0]++;
      else if (conf < 50) confidenceBins[1]++;
      else if (conf < 75) confidenceBins[2]++;
      else confidenceBins[3]++;
    });

    confidenceChart.data.datasets[0].data = confidenceBins;
    confidenceChart.update();
  } catch (error) {
    console.error("Error updating timeline:", error);
  }
}

// Update events table
async function updateEventsTable() {
  try {
    const response = await fetch("/api/events/recent");
    const events = await response.json();

    const tbody = document.getElementById("eventsTableBody");

    if (events.length === 0) {
      tbody.innerHTML =
        '<tr><td colspan="7" class="loading">No events detected yet</td></tr>';
      return;
    }

    tbody.innerHTML = events
      .map((event) => {
        const date = new Date(event.timestamp);
        const timeStr = date.toLocaleString();

        // Magnitude class
        let magClass = "magnitude-low";
        if (event.magnitude >= 5) magClass = "magnitude-high";
        else if (event.magnitude >= 3.5) magClass = "magnitude-medium";

        // Confidence class
        let confClass = "confidence-low";
        if (event.confidence >= 0.8) confClass = "confidence-high";
        else if (event.confidence >= 0.6) confClass = "confidence-medium";

        // Location info
        const location =
          event.epicenter_lat && event.epicenter_lon
            ? `${event.epicenter_lat.toFixed(
                2
              )}°, ${event.epicenter_lon.toFixed(2)}°`
            : "Unknown";

        const distance = event.distance_km
          ? `${event.distance_km.toFixed(1)} km`
          : "-";

        const probability = event.location_probability
          ? `${(event.location_probability * 100).toFixed(0)}%`
          : "-";

        return `
                <tr>
                    <td>${timeStr}</td>
                    <td><span class="magnitude ${magClass}">${event.magnitude.toFixed(
          1
        )}</span></td>
                    <td>${event.duration.toFixed(1)}s</td>
                    <td><span class="confidence ${confClass}">${(
          event.confidence * 100
        ).toFixed(0)}%</span></td>
                    <td>${location}</td>
                    <td>${distance}</td>
                    <td>${probability}</td>
                </tr>
            `;
      })
      .join("");

    // Add markers to map for events with location
    events.forEach((event) => {
      if (event.epicenter_lat && event.epicenter_lon) {
        // Check if marker already exists to avoid duplicates
        const exists = detectionMarkers.some((m) => {
          const latlng = m.getLatLng();
          return (
            latlng.lat === event.epicenter_lat &&
            latlng.lng === event.epicenter_lon
          );
        });
        if (!exists) {
          addEarthquakeMarker(event);
        }
      }
    });
  } catch (error) {
    console.error("Error updating events table:", error);
    document.getElementById("eventsTableBody").innerHTML =
      '<tr><td colspan="7" class="loading">Error loading events</td></tr>';
  }
}

// Update last update time
function updateLastUpdateTime() {
  const now = new Date();
  document.getElementById(
    "lastUpdate"
  ).textContent = `Last update: ${now.toLocaleTimeString()}`;
}

// Update all data
async function updateAll() {
  await updateStatistics();
  await updateTimeline();
  await updateEventsTable();
  await updatePredictions();
  updateLastUpdateTime();
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  initCharts();
  updateAll();

  // Auto-refresh every 5 seconds
  setInterval(updateAll, 5000);
});