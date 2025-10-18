// Seismic AI Detector Dashboard JavaScript

let magnitudeChart, confidenceChart;

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
        '<tr><td colspan="5" class="loading">No events detected yet</td></tr>';
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
                    <td>${event.pga.toFixed(3)} m/s²</td>
                </tr>
            `;
      })
      .join("");
  } catch (error) {
    console.error("Error updating events table:", error);
    document.getElementById("eventsTableBody").innerHTML =
      '<tr><td colspan="5" class="loading">Error loading events</td></tr>';
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
  updateLastUpdateTime();
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  updateAll();

  // Auto-refresh every 5 seconds
  setInterval(updateAll, 5000);
});
