import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, Cell
} from "recharts";
import axios from "axios";

const API         = "http://localhost:8000";
const ALL_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "GLD", "TLT"];
const COLORS      = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#0891b2", "#7c3aed"];
const ARCH_COLORS = { LSTM: "#2563eb", TRANSFORMER: "#d97706", NBEATS: "#0891b2" };

const styles = {
  page:       { background: "#f8fafc", minHeight: "100vh", color: "#1e293b", fontFamily: "'Inter', 'Segoe UI', sans-serif", padding: "24px 32px", maxWidth: 1300, margin: "0 auto" },
  card:       { background: "#ffffff", borderRadius: 12, padding: 24, marginBottom: 20, boxShadow: "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)", border: "1px solid #e2e8f0" },
  cardTitle:  { margin: "0 0 18px 0", fontSize: 12, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 600 },
  label:      { fontSize: 12, color: "#64748b", display: "block", marginBottom: 6, fontWeight: 500 },
  input:      { width: "100%", background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 8, padding: "9px 12px", color: "#1e293b", fontFamily: "inherit", fontSize: 14, boxSizing: "border-box", outline: "none" },
  inputFocus: { border: "1px solid #2563eb" },
  btnPrimary: { background: "#2563eb", color: "#fff", border: "none", borderRadius: 8, padding: "10px 28px", cursor: "pointer", fontSize: 14, fontWeight: 600, fontFamily: "inherit" },
  btnGhost:   { background: "#f1f5f9", color: "#475569", border: "1px solid #e2e8f0", borderRadius: 8, padding: "8px 18px", cursor: "pointer", fontSize: 13, fontWeight: 500, fontFamily: "inherit" },
  th:         { textAlign: "left", padding: "10px 14px", fontSize: 11, color: "#94a3b8", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", borderBottom: "1px solid #f1f5f9", background: "#f8fafc" },
  td:         { padding: "12px 14px", fontSize: 13, color: "#334155", borderBottom: "1px solid #f8fafc" },
  tag:        (color) => ({ background: color + "15", color, padding: "3px 10px", borderRadius: 20, fontWeight: 600, fontSize: 12, display: "inline-block" }),
  stat:       { background: "#f8fafc", borderRadius: 10, padding: "16px 20px", border: "1px solid #e2e8f0" },
};

function Badge({ label, color }) {
  return <span style={styles.tag(color)}>{label}</span>;
}

function Card({ title, children, accent }) {
  return (
    <div style={{ ...styles.card, borderTop: accent ? `3px solid ${accent}` : "1px solid #e2e8f0" }}>
      {title && <h2 style={styles.cardTitle}>{title}</h2>}
      {children}
    </div>
  );
}

function TextInput({ label, value, onChange, hint }) {
  const [local, setLocal] = useState(String(value));
  const [error, setError] = useState("");
  const [focused, setFocused] = useState(false);
  useEffect(() => setLocal(String(value)), [value]);
  function handleBlur() {
    setFocused(false);
    const p = parseFloat(local);
    if (isNaN(p) || p <= 0) { setError("Must be a positive number"); setLocal(String(value)); }
    else { setError(""); onChange(p); }
  }
  return (
    <div>
      <label style={styles.label}>{label} {hint && <span style={{ color: "#94a3b8", fontWeight: 400 }}>— {hint}</span>}</label>
      <input
        type="text" value={local}
        onChange={e => { setLocal(e.target.value); setError(""); }}
        onFocus={() => setFocused(true)}
        onBlur={handleBlur}
        style={{ ...styles.input, ...(focused ? styles.inputFocus : {}), ...(error ? { border: "1px solid #dc2626" } : {}) }}
      />
      {error && <div style={{ color: "#dc2626", fontSize: 11, marginTop: 4 }}>{error}</div>}
    </div>
  );
}

function Toggle({ label, description, enabled, onChange }) {
  return (
    <div
      onClick={() => onChange(!enabled)}
      style={{ background: enabled ? "#eff6ff" : "#f8fafc", border: `1px solid ${enabled ? "#bfdbfe" : "#e2e8f0"}`, borderRadius: 10, padding: "14px 16px", cursor: "pointer", userSelect: "none", transition: "all 0.15s" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: enabled ? "#1d4ed8" : "#475569" }}>{label}</span>
        <div style={{ width: 36, height: 20, borderRadius: 10, background: enabled ? "#2563eb" : "#cbd5e1", position: "relative", transition: "background 0.2s", flexShrink: 0 }}>
          <div style={{ width: 16, height: 16, borderRadius: "50%", background: "#fff", position: "absolute", top: 2, left: enabled ? 18 : 2, transition: "left 0.2s", boxShadow: "0 1px 2px rgba(0,0,0,0.2)" }} />
        </div>
      </div>
      <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.5 }}>{description}</div>
    </div>
  );
}

function PulsingDot({ color = "#16a34a" }) {
  return (
    <span style={{ position: "relative", display: "inline-block", width: 10, height: 10, marginRight: 8, verticalAlign: "middle" }}>
      <span style={{ position: "absolute", width: 10, height: 10, borderRadius: "50%", background: color, opacity: 0.3, animation: "ping 1.2s ease-in-out infinite" }} />
      <span style={{ position: "absolute", width: 10, height: 10, borderRadius: "50%", background: color }} />
      <style>{`@keyframes ping{0%,100%{transform:scale(1);opacity:0.3}50%{transform:scale(1.9);opacity:0}}`}</style>
    </span>
  );
}

function TrainingAnimator({ progress, isTraining }) {
  if (!isTraining) return null;
  const { current, step, total, epoch, max_epochs } = progress || {};
  const modelPct = total > 0 ? Math.round((step / total) * 100) : 0;
  const epochPct = max_epochs > 0 ? Math.round((epoch / max_epochs) * 100) : 0;
  const archColor = current?.includes("LSTM") ? "#2563eb" : current?.includes("TRANSFORMER") ? "#d97706" : current?.includes("NBEATS") ? "#0891b2" : "#16a34a";

  return (
    <div style={{ background: "#f8fafc", borderRadius: 10, padding: 16, marginBottom: 16, border: `1px solid ${archColor}30` }}>
      <div style={{ display: "flex", alignItems: "center", marginBottom: 14 }}>
        <PulsingDot color={archColor} />
        <span style={{ fontSize: 13, color: archColor, fontWeight: 600 }}>{current || "Initializing..."}</span>
      </div>
      <div style={{ marginBottom: 10 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94a3b8", marginBottom: 5 }}>
          <span>Overall progress</span><span>{step}/{total} models — {modelPct}%</span>
        </div>
        <div style={{ background: "#e2e8f0", borderRadius: 4, height: 8, overflow: "hidden" }}>
          <div style={{ background: "#16a34a", height: "100%", width: `${modelPct}%`, transition: "width 0.4s ease", borderRadius: 4 }} />
        </div>
      </div>
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94a3b8", marginBottom: 5 }}>
          <span>Current model epoch</span><span>{epoch}/{max_epochs} — {epochPct}%</span>
        </div>
        <div style={{ background: "#e2e8f0", borderRadius: 4, height: 5, overflow: "hidden" }}>
          <div style={{ background: archColor, height: "100%", width: `${epochPct}%`, transition: "width 0.3s ease", borderRadius: 4 }} />
        </div>
      </div>
    </div>
  );
}

function ResultsTable({ results }) {
  if (!results || results.length === 0) return null;
  return (
    <div style={{ marginTop: 20 }}>
      <div style={styles.cardTitle}>Training Results</div>
      <div style={{ border: "1px solid #e2e8f0", borderRadius: 10, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              {["Model", "Arch", "MAE", "Dir Acc", "Precision", "Recall", "F1", "Stopped", "Status"].map(h => (
                <th key={h} style={styles.th}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => (
              <tr key={i} style={{ background: r.best ? "#f0fdf4" : i % 2 === 0 ? "#fff" : "#fafafa" }}>
                <td style={{ ...styles.td, fontWeight: r.best ? 700 : 400, color: r.best ? "#16a34a" : "#334155" }}>
                  {r.model} {r.best && "★"}
                </td>
                <td style={styles.td}><Badge label={r.arch} color={ARCH_COLORS[r.arch] || "#64748b"} /></td>
                <td style={{ ...styles.td, fontFamily: "monospace" }}>{r.mae ?? "—"}</td>
                <td style={styles.td}>
                  {r.dir_acc != null ? (
                    <span style={{ color: r.dir_acc >= 55 ? "#16a34a" : r.dir_acc >= 50 ? "#d97706" : "#dc2626", fontWeight: 700 }}>
                      {r.dir_acc}%
                    </span>
                  ) : "—"}
                </td>
                <td style={{ ...styles.td, fontFamily: "monospace" }}>{r.precision ?? "—"}</td>
                <td style={{ ...styles.td, fontFamily: "monospace" }}>{r.recall ?? "—"}</td>
                <td style={styles.td}>
                  {r.f1 != null ? (
                    <span style={{ color: r.f1 >= 0.6 ? "#16a34a" : r.f1 >= 0.5 ? "#d97706" : "#dc2626", fontWeight: 700 }}>
                      {r.f1}
                    </span>
                  ) : "—"}
                </td>
                <td style={{ ...styles.td, color: "#94a3b8" }}>ep {r.stopped ?? "—"}</td>
                <td style={styles.td}>
                  <Badge label={r.status} color={r.status === "done" ? "#16a34a" : r.status === "error" ? "#dc2626" : "#d97706"} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function App() {
  const [forecasts, setForecasts]       = useState([]);
  const [chartData, setChartData]       = useState([]);
  const [health, setHealth]             = useState(null);
  const [trainingLog, setTrainingLog]   = useState([]);
  const [results, setResults]           = useState([]);
  const [progress, setProgress]         = useState(null);
  const [isTraining, setIsTraining]     = useState(false);
  const [lastUpdated, setLastUpdated]   = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const logRef = useRef(null);

  const [settings, setSettings] = useState({
    epochs:          75,
    seq_len:         30,
    lr:              0.001,
    tickers:         ["SPY", "QQQ", "GLD"],
    use_ensemble:    true,
    use_conf_filter: true,
    conf_percentile: 40,
    early_stopping:  true,
    patience:        10,
  });

  function set(key, val) { setSettings(s => ({ ...s, [key]: val })); }

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [trainingLog]);

  async function fetchForecasts() {
    try {
      const res   = await Promise.all(ALL_TICKERS.map(t => axios.get(`${API}/forecast?ticker=${t}`).then(r => r.data).catch(() => null)));
      const valid = res.filter(Boolean);
      setForecasts(valid);
      setChartData(prev => {
        const point = { time: new Date().toLocaleTimeString() };
        valid.forEach(r => { point[r.ticker] = parseFloat((r.predicted_return * 100).toFixed(6)); });
        return [...prev, point].slice(-30);
      });
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (e) { console.error(e); }
  }

  async function fetchHealth() {
    try {
      const r = await axios.get(`${API}/health`);
      setHealth(r.data);
      setIsTraining(r.data.training);
    } catch (e) { console.error(e); }
  }

  async function fetchTrainingStatus() {
    try {
      const r  = await axios.get(`${API}/training_status`);
      setTrainingLog(r.data.log || []);
      setResults(r.data.results || []);
      setProgress(r.data.progress);
      const was = isTraining;
      setIsTraining(r.data.training);
      if (was && !r.data.training) fetchForecasts();
    } catch (e) { console.error(e); }
  }

  async function startTraining() {
    if (settings.tickers.length === 0) { alert("Select at least one ticker."); return; }
    try {
      const params = new URLSearchParams({
        epochs:          Math.round(settings.epochs),
        seq_len:         Math.round(settings.seq_len),
        lr:              settings.lr,
        tickers:         settings.tickers.join(","),
        use_ensemble:    settings.use_ensemble,
        use_conf_filter: settings.use_conf_filter,
        conf_percentile: settings.conf_percentile,
        early_stopping:  settings.early_stopping,
        patience:        Math.round(settings.patience),
      });
      await axios.post(`${API}/train?${params}`);
      setIsTraining(true);
      setTrainingLog(["Training started..."]);
      setResults([]);
    } catch (e) { alert("Failed to start training. Is the API running?"); }
  }

  useEffect(() => {
    fetchHealth(); fetchForecasts();
    const f = setInterval(fetchForecasts,     60000);
    const h = setInterval(fetchHealth,        10000);
    const t = setInterval(fetchTrainingStatus, 3000);
    return () => { clearInterval(f); clearInterval(h); clearInterval(t); };
  }, []);

  const confColor = c => c === "High" ? "#16a34a" : c === "Medium" ? "#d97706" : "#dc2626";

  return (
    <div style={styles.page}>

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, color: "#0f172a", letterSpacing: "-0.02em" }}>
            Adaptive Forecaster
          </h1>
          <p style={{ margin: "6px 0 0 0", fontSize: 13, color: "#64748b" }}>
            ML-powered equity direction forecasting
            {" · "}Model: <span style={{ color: "#2563eb", fontWeight: 600 }}>{health?.model_version || "none"}</span>
            {" · "}Retrained: <span style={{ color: "#2563eb", fontWeight: 600 }}>{health?.last_retrain || "never"}</span>
            {" · "}Updated: <span style={{ color: "#16a34a", fontWeight: 600 }}>{lastUpdated || "..."}</span>
            {isTraining && <span style={{ marginLeft: 10, color: "#d97706", fontWeight: 600 }}><PulsingDot color="#d97706" />Training active</span>}
          </p>
        </div>
        <button onClick={() => setShowSettings(s => !s)} style={showSettings ? { ...styles.btnGhost } : { ...styles.btnPrimary }}>
          {showSettings ? "Hide Settings" : "Training Settings"}
        </button>
      </div>

      {/* Stats row */}
      {forecasts.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14, marginBottom: 20 }}>
          <div style={styles.stat}>
            <div style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>Long Signals</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: "#16a34a" }}>
              {forecasts.filter(f => f.direction === "long").length}
              <span style={{ fontSize: 14, color: "#94a3b8", fontWeight: 400 }}> / {forecasts.length}</span>
            </div>
          </div>
          <div style={styles.stat}>
            <div style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>High Confidence</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: "#2563eb" }}>
              {forecasts.filter(f => f.confidence === "High").length}
              <span style={{ fontSize: 14, color: "#94a3b8", fontWeight: 400 }}> / {forecasts.length}</span>
            </div>
          </div>
          <div style={styles.stat}>
            <div style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>Avg Confidence Ratio</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: "#0f172a" }}>
              {(forecasts.reduce((a, f) => a + f.confidence_ratio, 0) / forecasts.length).toFixed(2)}x
            </div>
          </div>
          <div style={styles.stat}>
            <div style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>Tickers Loaded</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: "#0f172a" }}>
              {health?.tickers_loaded?.length || 0}
            </div>
          </div>
        </div>
      )}

      {/* Settings Panel */}
      {showSettings && (
        <Card title="Training Control Panel" accent="#2563eb">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 20 }}>
            <TextInput label="Epochs" hint="max iterations" value={settings.epochs} onChange={v => set("epochs", v)} />
            <TextInput label="Sequence Length (days)" hint="lookback window" value={settings.seq_len} onChange={v => set("seq_len", v)} />
            <TextInput label="Learning Rate" hint="recommended 0.001" value={settings.lr} onChange={v => set("lr", v)} />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 16, marginBottom: 20 }}>
            <TextInput label="Early Stop Patience" hint="epochs without improvement" value={settings.patience} onChange={v => set("patience", v)} />
            {settings.use_conf_filter && (
              <TextInput label="Confidence Percentile" hint="40 = trade top 60% signals" value={settings.conf_percentile} onChange={v => set("conf_percentile", v)} />
            )}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 12, marginBottom: 20 }}>
            <Toggle label="Early Stopping" description="Stops when validation accuracy plateaus. Prevents overfitting. Recommended." enabled={settings.early_stopping} onChange={v => set("early_stopping", v)} />
            <Toggle label="Ensemble Mode" description="Trains LSTM, Transformer, and N-BEATS then averages predictions. More accurate but 3x slower." enabled={settings.use_ensemble} onChange={v => set("use_ensemble", v)} />
            <Toggle label="Confidence Filter" description="Only signals LONG on high-conviction predictions. Raises directional accuracy 5-10%." enabled={settings.use_conf_filter} onChange={v => set("use_conf_filter", v)} />
            <div style={{ background: "#fffbeb", borderRadius: 10, padding: "14px 16px", border: "1px solid #fde68a" }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "#92400e", marginBottom: 4 }}>Volume + Volatility Features</div>
              <div style={{ fontSize: 12, color: "#a16207", lineHeight: 1.5 }}>Run python pipeline/features.py in terminal before training to activate realized_vol_10 and volume_trend.</div>
            </div>
          </div>

          <div style={{ marginBottom: 20 }}>
            <label style={styles.label}>Tickers to train on</label>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {ALL_TICKERS.map((t, i) => {
                const selected = settings.tickers.includes(t);
                return (
                  <button key={t}
                    onClick={() => set("tickers", selected ? settings.tickers.filter(x => x !== t) : [...settings.tickers, t])}
                    style={{ background: selected ? COLORS[i] + "15" : "#f8fafc", color: selected ? COLORS[i] : "#94a3b8", border: `1px solid ${selected ? COLORS[i] + "40" : "#e2e8f0"}`, borderRadius: 8, padding: "7px 18px", cursor: "pointer", fontFamily: "inherit", fontSize: 13, fontWeight: selected ? 700 : 400 }}>
                    {t}
                  </button>
                );
              })}
            </div>
          </div>

          <div style={{ background: "#f8fafc", borderRadius: 8, padding: "10px 14px", marginBottom: 20, fontSize: 12, color: "#64748b", border: "1px solid #e2e8f0" }}>
            <span style={{ fontWeight: 600, color: "#475569" }}>Config: </span>
            {Math.round(settings.epochs)} epochs · {Math.round(settings.seq_len)}d lookback · lr {settings.lr} · patience {Math.round(settings.patience)} · {settings.tickers.join(", ") || "none"}
            {settings.early_stopping  && <span style={{ color: "#16a34a", fontWeight: 600 }}> · early stop</span>}
            {settings.use_ensemble    && <span style={{ color: "#2563eb", fontWeight: 600 }}> · ensemble</span>}
            {settings.use_conf_filter && <span style={{ color: "#7c3aed", fontWeight: 600 }}> · conf filter @p{settings.conf_percentile}</span>}
          </div>

          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <button
              onClick={startTraining}
              disabled={isTraining}
              style={{ ...styles.btnPrimary, background: isTraining ? "#94a3b8" : "#16a34a", cursor: isTraining ? "not-allowed" : "pointer" }}>
              {isTraining ? "Training in progress..." : "Start Training"}
            </button>
            {isTraining && progress && (
              <span style={{ fontSize: 13, color: "#d97706", fontWeight: 600 }}>
                {progress.current} — {Math.round((progress.step / Math.max(progress.total, 1)) * 100)}%
              </span>
            )}
          </div>

          {(isTraining || progress?.current === "complete") && (
            <div style={{ marginTop: 16 }}>
              <TrainingAnimator progress={progress} isTraining={isTraining} />
            </div>
          )}

          {trainingLog.length > 0 && (
            <div ref={logRef} style={{ marginTop: 12, background: "#f8fafc", borderRadius: 8, padding: 14, maxHeight: 200, overflowY: "auto", border: "1px solid #e2e8f0", fontFamily: "monospace" }}>
              {trainingLog.map((line, i) => (
                <div key={i} style={{
                  fontSize: 12, lineHeight: 2,
                  color: line.includes("done") || line.includes("complete") || line.includes("Best") ? "#16a34a"
                       : line.includes("━━━") ? "#2563eb"
                       : line.includes("LSTM") ? "#2563eb"
                       : line.includes("TRANSFORMER") ? "#d97706"
                       : line.includes("NBEATS") ? "#0891b2"
                       : line.includes("error") || line.includes("failed") ? "#dc2626"
                       : line.includes("epoch") ? "#94a3b8"
                       : "#475569"
                }}>
                  {line}
                </div>
              ))}
            </div>
          )}

          <ResultsTable results={results} />
        </Card>
      )}

      {/* Forecast Table */}
      <Card title="Live Forecasts">
        {forecasts.length === 0 ? (
          <div style={{ color: "#94a3b8", fontSize: 14, padding: "20px 0" }}>Connecting to API...</div>
        ) : (
          <div style={{ border: "1px solid #e2e8f0", borderRadius: 10, overflow: "hidden" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  {["Ticker", "Direction", "Predicted Return", "Confidence", "Volatility"].map(h => (
                    <th key={h} style={styles.th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {forecasts.map((f, i) => (
                  <tr key={f.ticker} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                    <td style={{ ...styles.td, fontWeight: 700, color: COLORS[i] }}>{f.ticker}</td>
                    <td style={styles.td}>
                      <Badge label={f.direction === "long" ? "LONG" : "FLAT"} color={f.direction === "long" ? "#16a34a" : "#dc2626"} />
                    </td>
                    <td style={styles.td}>
                      <span style={{ color: f.predicted_return > 0 ? "#16a34a" : "#dc2626", fontWeight: 700, fontFamily: "monospace" }}>
                        {f.predicted_return > 0 ? "+" : ""}{(f.predicted_return * 100).toFixed(4)}%
                      </span>
                    </td>
                    <td style={styles.td}>
                      <Badge label={`${f.confidence} · ${f.confidence_ratio}x vol`} color={confColor(f.confidence)} />
                    </td>
                    <td style={{ ...styles.td, color: "#94a3b8", fontFamily: "monospace" }}>
                      {(f.ticker_volatility * 100).toFixed(4)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* Returns Chart */}
      <Card title="Predicted Returns Over Time (%)">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="time" stroke="#cbd5e1" tick={{ fontSize: 11, fill: "#94a3b8" }} />
            <YAxis stroke="#cbd5e1" tick={{ fontSize: 11, fill: "#94a3b8" }} tickFormatter={v => `${v}%`} />
            <Tooltip
              contentStyle={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 8, boxShadow: "0 4px 6px rgba(0,0,0,0.05)" }}
              labelStyle={{ color: "#64748b", fontSize: 12 }}
              formatter={v => [`${v}%`]}
            />
            <Legend wrapperStyle={{ fontSize: 12, color: "#64748b" }} />
            <ReferenceLine y={0} stroke="#e2e8f0" strokeWidth={1.5} />
            {ALL_TICKERS.map((t, i) => (
              <Line key={t} type="monotone" dataKey={t} stroke={COLORS[i]} dot={false} strokeWidth={2} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Signal Strength */}
      <Card title="Signal Strength by Ticker (confidence ratio vs volatility)">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={forecasts.map((f, i) => ({ ticker: f.ticker, ratio: f.confidence_ratio, color: COLORS[i] }))}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="ticker" stroke="#cbd5e1" tick={{ fontSize: 12, fill: "#64748b" }} />
            <YAxis stroke="#cbd5e1" tick={{ fontSize: 11, fill: "#94a3b8" }} />
            <Tooltip
              contentStyle={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 8 }}
              formatter={v => [`${v}x`]}
            />
            <ReferenceLine y={0.5} stroke="#16a34a" strokeDasharray="5 3"
              label={{ value: "High confidence threshold", fill: "#16a34a", fontSize: 11, position: "right" }} />
            <Bar dataKey="ratio" radius={[6, 6, 0, 0]}>
              {forecasts.map((f, i) => <Cell key={f.ticker} fill={COLORS[i]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

    </div>
  );
}