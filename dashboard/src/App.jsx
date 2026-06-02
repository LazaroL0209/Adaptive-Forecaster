import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, Cell
} from "recharts";
import axios from "axios";

const API          = "http://localhost:8000";
const ALL_TICKERS  = ["SPY", "QQQ", "AAPL", "MSFT", "GLD", "TLT"];
const COLORS       = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#14b8a6", "#f97316"];
const ARCH_COLORS  = { LSTM: "#6366f1", TRANSFORMER: "#f59e0b", NBEATS: "#14b8a6" };

function Badge({ label, color }) {
  return (
    <span style={{ background: color + "22", color, padding: "2px 10px", borderRadius: 6, fontWeight: "bold", fontSize: 12 }}>
      {label}
    </span>
  );
}

function Card({ title, children, accent }) {
  return (
    <div style={{ background: "#1a1a1a", borderRadius: 10, padding: 20, marginBottom: 20, borderTop: accent ? `2px solid ${accent}` : "none" }}>
      {title && <h2 style={{ margin: "0 0 16px 0", fontSize: 13, color: "#666", textTransform: "uppercase", letterSpacing: 1 }}>{title}</h2>}
      {children}
    </div>
  );
}

function TextInput({ label, value, onChange, hint }) {
  const [local, setLocal] = useState(String(value));
  const [error, setError] = useState("");
  useEffect(() => setLocal(String(value)), [value]);
  function handleBlur() {
    const p = parseFloat(local);
    if (isNaN(p) || p <= 0) { setError("Must be a positive number"); setLocal(String(value)); }
    else { setError(""); onChange(p); }
  }
  return (
    <div>
      <label style={{ fontSize: 12, color: "#aaa", display: "block", marginBottom: 6 }}>
        {label} {hint && <span style={{ color: "#555" }}>({hint})</span>}
      </label>
      <input type="text" value={local}
        onChange={e => { setLocal(e.target.value); setError(""); }}
        onBlur={handleBlur}
        style={{ width: "100%", background: "#111", border: `1px solid ${error ? "#ef4444" : "#333"}`, borderRadius: 6, padding: "8px 12px", color: "#fff", fontFamily: "monospace", fontSize: 14, boxSizing: "border-box", outline: "none" }}
      />
      {error && <div style={{ color: "#ef4444", fontSize: 11, marginTop: 4 }}>{error}</div>}
    </div>
  );
}

function Toggle({ label, description, enabled, onChange }) {
  return (
    <div onClick={() => onChange(!enabled)} style={{ background: enabled ? "#1a2a1a" : "#111", border: `1px solid ${enabled ? "#22c55e44" : "#2a2a2a"}`, borderRadius: 8, padding: "12px 16px", cursor: "pointer", userSelect: "none" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontSize: 13, fontWeight: "bold", color: enabled ? "#22c55e" : "#aaa" }}>{label}</span>
        <div style={{ width: 32, height: 18, borderRadius: 9, background: enabled ? "#22c55e" : "#333", position: "relative" }}>
          <div style={{ width: 14, height: 14, borderRadius: "50%", background: "#fff", position: "absolute", top: 2, left: enabled ? 16 : 2, transition: "left 0.2s" }} />
        </div>
      </div>
      <div style={{ fontSize: 11, color: "#555", lineHeight: 1.5 }}>{description}</div>
    </div>
  );
}

function PulsingDot({ color = "#22c55e" }) {
  return (
    <span style={{ position: "relative", display: "inline-block", width: 10, height: 10, marginRight: 8 }}>
      <span style={{ position: "absolute", width: 10, height: 10, borderRadius: "50%", background: color, opacity: 0.4, animation: "ping 1.2s ease-in-out infinite" }} />
      <span style={{ position: "absolute", width: 10, height: 10, borderRadius: "50%", background: color }} />
      <style>{`@keyframes ping { 0%,100%{transform:scale(1);opacity:0.4} 50%{transform:scale(1.8);opacity:0} }`}</style>
    </span>
  );
}

function TrainingAnimator({ progress, isTraining }) {
  if (!isTraining) return null;
  const { current, step, total, epoch, max_epochs } = progress || {};
  const modelPct  = total > 0 ? Math.round((step / total) * 100) : 0;
  const epochPct  = max_epochs > 0 ? Math.round((epoch / max_epochs) * 100) : 0;
  const archColor = current?.includes("LSTM") ? "#6366f1"
                  : current?.includes("TRANSFORMER") ? "#f59e0b"
                  : current?.includes("NBEATS") ? "#14b8a6" : "#22c55e";

  return (
    <div style={{ background: "#111", borderRadius: 8, padding: 16, marginBottom: 16, border: `1px solid ${archColor}33` }}>
      <div style={{ display: "flex", alignItems: "center", marginBottom: 12 }}>
        <PulsingDot color={archColor} />
        <span style={{ fontSize: 13, color: archColor, fontWeight: "bold" }}>{current || "Initializing..."}</span>
      </div>

      <div style={{ marginBottom: 10 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#555", marginBottom: 4 }}>
          <span>Overall progress</span>
          <span>{step}/{total} models — {modelPct}%</span>
        </div>
        <div style={{ background: "#222", borderRadius: 3, height: 8, overflow: "hidden" }}>
          <div style={{ background: "#22c55e", height: "100%", width: `${modelPct}%`, transition: "width 0.4s ease" }} />
        </div>
      </div>

      <div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#555", marginBottom: 4 }}>
          <span>Current model epoch</span>
          <span>{epoch}/{max_epochs} — {epochPct}%</span>
        </div>
        <div style={{ background: "#222", borderRadius: 3, height: 5, overflow: "hidden" }}>
          <div style={{ background: archColor, height: "100%", width: `${epochPct}%`, transition: "width 0.3s ease" }} />
        </div>
      </div>
    </div>
  );
}

function ResultsTable({ results }) {
  if (!results || results.length === 0) return null;
  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ fontSize: 12, color: "#666", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>Results</div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #2a2a2a", color: "#555" }}>
            <th style={{ textAlign: "left", padding: "6px 10px" }}>Model</th>
            <th style={{ textAlign: "left", padding: "6px 10px" }}>Arch</th>
            <th style={{ textAlign: "left", padding: "6px 10px" }}>MAE</th>
            <th style={{ textAlign: "left", padding: "6px 10px" }}>Dir Acc</th>
            <th style={{ textAlign: "left", padding: "6px 10px" }}>Precision</th>
            <th style={{ textAlign: "left", padding: "6px 10px" }}>Recall</th>
            <th style={{ textAlign: "left", padding: "6px 10px" }}>F1</th>
            <th style={{ textAlign: "left", padding: "6px 10px" }}>Stopped</th>
            <th style={{ textAlign: "left", padding: "6px 10px" }}>Status</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={i} style={{ borderBottom: "1px solid #1a1a1a", background: r.best ? "#1a2a1a" : "transparent" }}>
              <td style={{ padding: "8px 10px", color: r.best ? "#22c55e" : "#aaa", fontWeight: r.best ? "bold" : "normal" }}>
                {r.model} {r.best && "★"}
              </td>
              <td style={{ padding: "8px 10px" }}>
                <Badge label={r.arch} color={ARCH_COLORS[r.arch] || "#888"} />
              </td>
              <td style={{ padding: "8px 10px", color: "#fff" }}>{r.mae ?? "—"}</td>
              <td style={{ padding: "8px 10px" }}>
                {r.dir_acc != null ? (
                  <span style={{ color: r.dir_acc >= 55 ? "#22c55e" : r.dir_acc >= 50 ? "#f59e0b" : "#ef4444", fontWeight: "bold" }}>
                    {r.dir_acc}%
                  </span>
                ) : "—"}
              </td>
              <td style={{ padding: "8px 10px", color: "#aaa" }}>{r.precision ?? "—"}</td>
              <td style={{ padding: "8px 10px", color: "#aaa" }}>{r.recall ?? "—"}</td>
              <td style={{ padding: "8px 10px" }}>
                {r.f1 != null ? (
                  <span style={{ color: r.f1 >= 0.6 ? "#22c55e" : r.f1 >= 0.5 ? "#f59e0b" : "#ef4444", fontWeight: "bold" }}>
                    {r.f1}
                  </span>
                ) : "—"}
              </td>
              <td style={{ padding: "8px 10px", color: "#555" }}>ep {r.stopped ?? "—"}</td>
              <td style={{ padding: "8px 10px" }}>
                <Badge
                  label={r.status}
                  color={r.status === "done" ? "#22c55e" : r.status === "error" ? "#ef4444" : "#f59e0b"}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
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
      const results = await Promise.all(
        ALL_TICKERS.map(t => axios.get(`${API}/forecast?ticker=${t}`).then(r => r.data).catch(() => null))
      );
      const valid = results.filter(Boolean);
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
      const r = await axios.get(`${API}/training_status`);
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

  const confColor = c => c === "High" ? "#22c55e" : c === "Medium" ? "#f59e0b" : "#ef4444";
  const pct = progress ? Math.round((progress.step / Math.max(progress.total, 1)) * 100) : 0;

  return (
    <div style={{ background: "#0f0f0f", minHeight: "100vh", color: "#e5e5e5", fontFamily: "monospace", padding: 24, maxWidth: 1200, margin: "0 auto" }}>

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 22, color: "#fff" }}>Adaptive Forecaster</h1>
          <p style={{ margin: "4px 0 0 0", fontSize: 12, color: "#555" }}>
            Model: <span style={{ color: "#6366f1" }}>{health?.model_version || "none"}</span>
            {" · "}Retrained: <span style={{ color: "#6366f1" }}>{health?.last_retrain || "never"}</span>
            {" · "}Updated: <span style={{ color: "#22c55e" }}>{lastUpdated || "..."}</span>
            {isTraining && <span style={{ color: "#f59e0b", marginLeft: 8 }}><PulsingDot color="#f59e0b" />Training active</span>}
          </p>
        </div>
        <button onClick={() => setShowSettings(s => !s)}
          style={{ background: showSettings ? "#333" : "#6366f1", color: "#fff", border: "none", borderRadius: 8, padding: "8px 18px", cursor: "pointer", fontFamily: "monospace", fontSize: 13 }}>
          {showSettings ? "Hide Settings" : "Training Settings"}
        </button>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <Card title="Training Control Panel" accent="#6366f1">

          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 20 }}>
            <TextInput label="Epochs" hint="max iterations (early stop may exit sooner)" value={settings.epochs} onChange={v => set("epochs", v)} />
            <TextInput label="Sequence Length (days)" hint="lookback window, recommended 20-30" value={settings.seq_len} onChange={v => set("seq_len", v)} />
            <TextInput label="Learning Rate" hint="recommended 0.001" value={settings.lr} onChange={v => set("lr", v)} />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 16, marginBottom: 20 }}>
            <TextInput label="Early Stop Patience" hint="epochs without improvement before stopping" value={settings.patience} onChange={v => set("patience", v)} />
            {settings.use_conf_filter && (
              <TextInput label="Confidence Percentile" hint="40 = trade top 60% most confident signals" value={settings.conf_percentile} onChange={v => set("conf_percentile", v)} />
            )}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 10, marginBottom: 20 }}>
            <Toggle label="Early Stopping" description="Stops training when validation accuracy stops improving. Prevents overfitting. Highly recommended." enabled={settings.early_stopping} onChange={v => set("early_stopping", v)} />
            <Toggle label="Ensemble Mode" description="Trains LSTM, Transformer, and N-BEATS then averages their predictions. More accurate but 3x slower." enabled={settings.use_ensemble} onChange={v => set("use_ensemble", v)} />
            <Toggle label="Confidence Filter" description="Only signals LONG on high-conviction predictions. Reduces trade frequency but raises accuracy 5-10%." enabled={settings.use_conf_filter} onChange={v => set("use_conf_filter", v)} />
            <div style={{ background: "#111", borderRadius: 8, padding: "12px 16px", border: "1px solid #2a2a2a" }}>
              <div style={{ fontSize: 13, fontWeight: "bold", color: "#f59e0b", marginBottom: 4 }}>Volume + Volatility Features</div>
              <div style={{ fontSize: 11, color: "#555", lineHeight: 1.5 }}>
                Rerun python pipeline/features.py in your terminal to activate realized_vol_10 and volume_trend features before training.
              </div>
            </div>
          </div>

          <div style={{ marginBottom: 16 }}>
            <label style={{ fontSize: 12, color: "#666", textTransform: "uppercase", letterSpacing: 1, display: "block", marginBottom: 8 }}>Tickers</label>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {ALL_TICKERS.map((t, i) => {
                const selected = settings.tickers.includes(t);
                return (
                  <button key={t}
                    onClick={() => set("tickers", selected ? settings.tickers.filter(x => x !== t) : [...settings.tickers, t])}
                    style={{ background: selected ? COLORS[i] + "33" : "#222", color: selected ? COLORS[i] : "#555", border: `1px solid ${selected ? COLORS[i] : "#333"}`, borderRadius: 6, padding: "6px 16px", cursor: "pointer", fontFamily: "monospace", fontSize: 13 }}>
                    {t}
                  </button>
                );
              })}
            </div>
          </div>

          <div style={{ background: "#111", borderRadius: 6, padding: "10px 14px", marginBottom: 16, fontSize: 12, color: "#555" }}>
            <span style={{ color: "#777" }}>Config: </span>
            <span style={{ color: "#fff" }}>{Math.round(settings.epochs)} epochs</span>
            {" · "}<span style={{ color: "#fff" }}>{Math.round(settings.seq_len)}d lookback</span>
            {" · "}<span style={{ color: "#fff" }}>lr {settings.lr}</span>
            {" · "}<span style={{ color: "#fff" }}>patience {Math.round(settings.patience)}</span>
            {" · "}<span style={{ color: "#fff" }}>{settings.tickers.join(", ") || "none"}</span>
            {settings.early_stopping  && <span style={{ color: "#22c55e" }}> · early stop</span>}
            {settings.use_ensemble    && <span style={{ color: "#22c55e" }}> · ensemble</span>}
            {settings.use_conf_filter && <span style={{ color: "#22c55e" }}> · conf filter @p{settings.conf_percentile}</span>}
          </div>

          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <button onClick={startTraining} disabled={isTraining}
              style={{ background: isTraining ? "#1a1a1a" : "#22c55e", color: isTraining ? "#444" : "#000", border: `1px solid ${isTraining ? "#333" : "#22c55e"}`, borderRadius: 8, padding: "10px 28px", cursor: isTraining ? "not-allowed" : "pointer", fontFamily: "monospace", fontSize: 14, fontWeight: "bold" }}>
              {isTraining ? "Training..." : "Start Training"}
            </button>
          </div>

          {/* Animated training status */}
          {(isTraining || (progress?.current === "complete")) && (
            <div style={{ marginTop: 16 }}>
              <TrainingAnimator progress={progress} isTraining={isTraining} />
            </div>
          )}

          {/* Live log */}
          {trainingLog.length > 0 && (
            <div ref={logRef} style={{ marginTop: 12, background: "#080808", borderRadius: 6, padding: 12, maxHeight: 180, overflowY: "auto", border: "1px solid #222" }}>
              {trainingLog.map((line, i) => (
                <div key={i} style={{
                  fontSize: 11, lineHeight: 1.8, fontFamily: "monospace",
                  color: line.includes("done") || line.includes("complete") || line.includes("Best") ? "#22c55e"
                       : line.includes("━━━") ? "#6366f1"
                       : line.includes("LSTM") ? "#6366f1"
                       : line.includes("TRANSFORMER") ? "#f59e0b"
                       : line.includes("NBEATS") ? "#14b8a6"
                       : line.includes("error") || line.includes("failed") ? "#ef4444"
                       : line.includes("epoch") ? "#555"
                       : "#777"
                }}>
                  {line}
                </div>
              ))}
            </div>
          )}

          {/* Results table */}
          <ResultsTable results={results} />
        </Card>
      )}

      {/* Forecast Table */}
      <Card title="Live Forecasts">
        {forecasts.length === 0 ? (
          <div style={{ color: "#555", fontSize: 13 }}>Connecting to API...</div>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #2a2a2a", color: "#555", fontSize: 11 }}>
                <th style={{ textAlign: "left", padding: "6px 12px" }}>Ticker</th>
                <th style={{ textAlign: "left", padding: "6px 12px" }}>Direction</th>
                <th style={{ textAlign: "left", padding: "6px 12px" }}>Predicted Return</th>
                <th style={{ textAlign: "left", padding: "6px 12px" }}>Confidence</th>
                <th style={{ textAlign: "left", padding: "6px 12px" }}>Volatility</th>
              </tr>
            </thead>
            <tbody>
              {forecasts.map((f, i) => (
                <tr key={f.ticker} style={{ borderBottom: "1px solid #1f1f1f" }}>
                  <td style={{ padding: "12px", fontWeight: "bold", color: COLORS[i] }}>{f.ticker}</td>
                  <td style={{ padding: "12px" }}>
                    <Badge label={f.direction === "long" ? "LONG" : "FLAT"} color={f.direction === "long" ? "#22c55e" : "#ef4444"} />
                  </td>
                  <td style={{ padding: "12px", fontSize: 13 }}>
                    <span style={{ color: f.predicted_return > 0 ? "#22c55e" : "#ef4444" }}>
                      {f.predicted_return > 0 ? "+" : ""}{(f.predicted_return * 100).toFixed(4)}%
                    </span>
                  </td>
                  <td style={{ padding: "12px" }}>
                    <Badge label={`${f.confidence} (${f.confidence_ratio}x)`} color={confColor(f.confidence)} />
                  </td>
                  <td style={{ padding: "12px", fontSize: 12, color: "#555" }}>
                    {(f.ticker_volatility * 100).toFixed(4)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      {/* Returns Chart */}
      <Card title="Predicted Returns Over Time (%)">
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
            <XAxis dataKey="time" stroke="#333" tick={{ fontSize: 10 }} />
            <YAxis stroke="#333" tick={{ fontSize: 10 }} tickFormatter={v => `${v}%`} />
            <Tooltip contentStyle={{ background: "#111", border: "1px solid #333", borderRadius: 6 }} formatter={v => [`${v}%`]} />
            <Legend />
            <ReferenceLine y={0} stroke="#333" strokeDasharray="4 4" />
            {ALL_TICKERS.map((t, i) => (
              <Line key={t} type="monotone" dataKey={t} stroke={COLORS[i]} dot={false} strokeWidth={1.5} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Confidence Bar Chart */}
      <Card title="Signal Strength by Ticker">
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={forecasts.map((f, i) => ({ ticker: f.ticker, ratio: f.confidence_ratio, color: COLORS[i] }))}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
            <XAxis dataKey="ticker" stroke="#333" tick={{ fontSize: 11 }} />
            <YAxis stroke="#333" tick={{ fontSize: 10 }} />
            <Tooltip contentStyle={{ background: "#111", border: "1px solid #333", borderRadius: 6 }} />
            <ReferenceLine y={0.5} stroke="#22c55e" strokeDasharray="4 4"
              label={{ value: "High confidence threshold", fill: "#22c55e", fontSize: 10 }} />
            <Bar dataKey="ratio" radius={[4, 4, 0, 0]}>
              {forecasts.map((f, i) => <Cell key={f.ticker} fill={COLORS[i]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

    </div>
  );
}