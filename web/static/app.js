/**
 * app.js
 * EvoArena web frontend.
 *
 * Connects via WebSocket, renders the simulation on a Canvas,
 * and lets the user control parameters in real time.
 */

// ── Canvas setup ──────────────────────────────────────────────────────────────

const arenaCanvas = document.getElementById("arena");
const ctx         = arenaCanvas.getContext("2d");
const graphCanvas = document.getElementById("graph");
const gctx        = graphCanvas.getContext("2d");

// ── DOM refs ──────────────────────────────────────────────────────────────────

const connBadge   = document.getElementById("conn-badge");
const overlay     = document.getElementById("overlay");
const overlayText = overlay.querySelector(".overlay-text");
const btnStart    = document.getElementById("btn-startstop");

const hudGen    = document.getElementById("h-gen");
const hudStep   = document.getElementById("h-step");
const hudAgents = document.getElementById("h-agents");
const hudBest   = document.getElementById("h-best");
const hudAvg    = document.getElementById("h-avg");

const slPop   = document.getElementById("sl-pop");
const slMut   = document.getElementById("sl-mut");
const slElite = document.getElementById("sl-elite");
const valPop  = document.getElementById("val-pop");
const valMut  = document.getElementById("val-mut");
const valElite= document.getElementById("val-elite");

// ── State ─────────────────────────────────────────────────────────────────────

let ws          = null;
let isRunning   = false;
let obstacles   = [];
let arenaW      = 800;
let arenaH      = 600;
let agents      = [];
let bestHistory = [];
let avgHistory  = [];
let currentGen  = 0;
let currentStep = 0;

// ── Visual constants (match Pygame visualizer) ────────────────────────────────

const C = {
  BG:            "#0f0f19",
  OBSTACLE_FILL: "#3c3c50",
  OBSTACLE_EDGE: "#50506e",
  AGENT_FILL:    "#50c878",
  AGENT_OUTLINE: "#ffffff",
  ALPHA_FILL:    "#ffff64",
  ALPHA_RING:    "#ffd700",
  HEALTH_BG:     "#b40000",
  HEALTH_FILL:   "#00c800",
  HUD_TEXT:      "#c8c8dc",
  GRAPH_BEST:    "#ffdc32",
  GRAPH_AVG:     "#50c8c8",
  GRAPH_BG:      "#0f0f19",
  GRAPH_GRID:    "#1e1e30",
};

const AGENT_RADIUS = 10;
const MAX_HEALTH   = 100;

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    connBadge.textContent = "Connected";
    connBadge.className   = "badge connected";
    btnStart.disabled     = false;
  };

  ws.onclose = () => {
    connBadge.textContent = "Disconnected";
    connBadge.className   = "badge disconnected";
    btnStart.disabled     = true;
    isRunning = false;
    syncUI();
    // Reconnect after 2s
    setTimeout(connect, 2000);
  };

  ws.onerror = () => ws.close();

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    switch (msg.type) {
      case "state":    handleState(msg);   break;
      case "frame":    handleFrame(msg);   break;
      case "gen_end":  handleGenEnd(msg);  break;
    }
  };
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// ── Message handlers ──────────────────────────────────────────────────────────

function handleState(msg) {
  isRunning   = msg.running;
  obstacles   = msg.obstacles || obstacles;
  arenaW      = msg.arena?.width  || arenaW;
  arenaH      = msg.arena?.height || arenaH;
  bestHistory = msg.best_history || [];
  avgHistory  = msg.avg_history  || [];

  if (msg.config) {
    applyConfig(msg.config);
  }

  syncUI();
  drawArena();
  drawGraph();
}

function handleFrame(msg) {
  currentStep = msg.step;
  currentGen  = msg.gen;
  agents      = msg.agents;

  const alpha = agents.find(a => a.is_alpha);

  hudGen.textContent    = msg.gen;
  hudStep.textContent   = msg.step;
  hudAgents.textContent = agents.length;
  hudBest.textContent   = alpha ? alpha.fitness.toFixed(0) : "—";

  drawArena();
}

function handleGenEnd(msg) {
  bestHistory = msg.best_history;
  avgHistory  = msg.avg_history;

  hudAvg.textContent  = msg.avg.toFixed(0);
  hudBest.textContent = msg.best.toFixed(0);

  drawGraph();
}

// ── UI sync ───────────────────────────────────────────────────────────────────

function syncUI() {
  if (isRunning) {
    btnStart.textContent = "Stop";
    btnStart.classList.add("running");
    overlay.classList.add("hidden");
  } else {
    btnStart.textContent = "Start";
    btnStart.classList.remove("running");
    overlay.classList.remove("hidden");
    overlayText.textContent = "Stopped";
  }
}

function applyConfig(cfg) {
  if (cfg.population_size != null) {
    slPop.value         = cfg.population_size;
    valPop.textContent  = cfg.population_size;
  }
  if (cfg.mutation_rate != null) {
    slMut.value         = Math.round(cfg.mutation_rate * 100);
    valMut.textContent  = cfg.mutation_rate.toFixed(2);
  }
  if (cfg.elite_pct != null) {
    slElite.value       = Math.round(cfg.elite_pct * 100);
    valElite.textContent= Math.round(cfg.elite_pct * 100) + "%";
  }
}

function sendConfig() {
  send({
    type:            "config",
    population_size: parseInt(slPop.value),
    mutation_rate:   parseInt(slMut.value) / 100,
    elite_pct:       parseInt(slElite.value) / 100,
  });
}

// ── Controls ──────────────────────────────────────────────────────────────────

btnStart.addEventListener("click", () => {
  if (isRunning) {
    send({ type: "stop" });
    isRunning = false;
  } else {
    sendConfig(); // send latest slider values before starting
    send({ type: "start" });
    isRunning = true;
  }
  syncUI();
});

slPop.addEventListener("input", () => {
  valPop.textContent = slPop.value;
  sendConfig();
});

slMut.addEventListener("input", () => {
  valMut.textContent = (parseInt(slMut.value) / 100).toFixed(2);
  sendConfig();
});

slElite.addEventListener("input", () => {
  valElite.textContent = slElite.value + "%";
  sendConfig();
});

// ── Arena renderer ────────────────────────────────────────────────────────────

function drawArena() {
  ctx.fillStyle = C.BG;
  ctx.fillRect(0, 0, arenaCanvas.width, arenaCanvas.height);

  drawObstacles();
  drawAgents();
}

function drawObstacles() {
  for (const obs of obstacles) {
    ctx.fillStyle = C.OBSTACLE_FILL;
    ctx.fillRect(obs.x, obs.y, obs.w, obs.h);
    ctx.strokeStyle = C.OBSTACLE_EDGE;
    ctx.lineWidth   = 2;
    ctx.strokeRect(obs.x, obs.y, obs.w, obs.h);
  }
}

function drawAgents() {
  for (const agent of agents) {
    const x = agent.x;
    const y = agent.y;
    const r = AGENT_RADIUS;

    if (agent.is_alpha) {
      // Gold ring behind agent
      ctx.beginPath();
      ctx.arc(x, y, r + 4, 0, Math.PI * 2);
      ctx.strokeStyle = C.ALPHA_RING;
      ctx.lineWidth   = 3;
      ctx.stroke();
    }

    // Agent body
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = agent.is_alpha ? C.ALPHA_FILL : C.AGENT_FILL;
    ctx.fill();
    ctx.strokeStyle = C.AGENT_OUTLINE;
    ctx.lineWidth   = 1;
    ctx.stroke();

    // Health bar
    const barW    = 20;
    const barH    = 4;
    const barX    = x - barW / 2;
    const barY    = y - r - 8;
    const fillW   = Math.max(0, barW * (agent.health / MAX_HEALTH));

    ctx.fillStyle = C.HEALTH_BG;
    ctx.fillRect(barX, barY, barW, barH);
    ctx.fillStyle = C.HEALTH_FILL;
    ctx.fillRect(barX, barY, fillW, barH);

    // ID label
    ctx.fillStyle  = "#141414";
    ctx.font       = "bold 9px monospace";
    ctx.textAlign  = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(agent.id, x, y);
  }

  // Reset text alignment
  ctx.textAlign    = "left";
  ctx.textBaseline = "alphabetic";
}

// ── Fitness graph renderer ────────────────────────────────────────────────────

function drawGraph() {
  const W = graphCanvas.width;
  const H = graphCanvas.height;

  gctx.fillStyle = C.GRAPH_BG;
  gctx.fillRect(0, 0, W, H);

  if (bestHistory.length < 2) {
    gctx.fillStyle = "#3a3a55";
    gctx.font      = "13px monospace";
    gctx.textAlign = "center";
    gctx.fillText("Waiting for data…", W / 2, H / 2);
    gctx.textAlign = "left";
    return;
  }

  const PAD = { top: 20, right: 20, bottom: 30, left: 55 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top  - PAD.bottom;

  // Y range — include zero, pad 10%
  const allVals = [...bestHistory, ...avgHistory];
  let yMin = Math.min(0, ...allVals);
  let yMax = Math.max(...allVals);
  const yPad = (yMax - yMin) * 0.1 || 50;
  yMin -= yPad;
  yMax += yPad;

  const xScale = plotW / (bestHistory.length - 1);
  const yScale = plotH / (yMax - yMin);

  function toCanvas(genIdx, val) {
    return {
      x: PAD.left + genIdx * xScale,
      y: PAD.top  + plotH - (val - yMin) * yScale,
    };
  }

  // Grid lines
  const gridLines = 4;
  gctx.strokeStyle = C.GRAPH_GRID;
  gctx.lineWidth   = 1;
  for (let i = 0; i <= gridLines; i++) {
    const val = yMin + (yMax - yMin) * (i / gridLines);
    const y   = PAD.top + plotH - (val - yMin) * yScale;
    gctx.beginPath();
    gctx.moveTo(PAD.left, y);
    gctx.lineTo(PAD.left + plotW, y);
    gctx.stroke();

    // Y axis labels
    gctx.fillStyle   = "#6a6a88";
    gctx.font        = "10px monospace";
    gctx.textAlign   = "right";
    gctx.textBaseline= "middle";
    gctx.fillText(Math.round(val), PAD.left - 6, y);
  }

  // Zero line (if in range)
  if (yMin < 0 && yMax > 0) {
    const zeroY = PAD.top + plotH - (0 - yMin) * yScale;
    gctx.strokeStyle = "#303050";
    gctx.lineWidth   = 1;
    gctx.setLineDash([4, 4]);
    gctx.beginPath();
    gctx.moveTo(PAD.left, zeroY);
    gctx.lineTo(PAD.left + plotW, zeroY);
    gctx.stroke();
    gctx.setLineDash([]);
  }

  // Draw line helper
  function drawLine(history, color) {
    gctx.beginPath();
    gctx.strokeStyle = color;
    gctx.lineWidth   = 2;
    history.forEach((val, i) => {
      const { x, y } = toCanvas(i, val);
      i === 0 ? gctx.moveTo(x, y) : gctx.lineTo(x, y);
    });
    gctx.stroke();

    // Dot on latest point
    const last = toCanvas(history.length - 1, history[history.length - 1]);
    gctx.beginPath();
    gctx.arc(last.x, last.y, 4, 0, Math.PI * 2);
    gctx.fillStyle = color;
    gctx.fill();
  }

  drawLine(avgHistory,  C.GRAPH_AVG);
  drawLine(bestHistory, C.GRAPH_BEST);

  // X axis label
  gctx.fillStyle    = "#6a6a88";
  gctx.font         = "10px monospace";
  gctx.textAlign    = "center";
  gctx.textBaseline = "top";
  gctx.fillText(`Generation ${bestHistory.length}`, W / 2, H - PAD.bottom + 6);

  gctx.textAlign    = "left";
  gctx.textBaseline = "alphabetic";
}

// ── Init ──────────────────────────────────────────────────────────────────────

drawArena();
drawGraph();
connect();
