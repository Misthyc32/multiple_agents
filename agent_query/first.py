import os, io, base64,re,binascii
from typing import Optional, List
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # <- sin GUI
import matplotlib.pyplot as plt
from uuid import uuid4
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import sqlglot
from sqlglot.errors import ParseError
from sqlalchemy import text
import traceback

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import StructuredTool
from langchain.schema import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from fastapi import FastAPI,HTTPException, Request

from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from fastapi.responses import Response
from fastapi.responses import JSONResponse

# ------------- ENV & ENGINE -------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required in .env")

PG_CONN = os.getenv("SUPABASE_PG_CONN")
if not PG_CONN:
    # Fail fast with clear guidance
    raise RuntimeError(
        "Missing SUPABASE_PG_CONN in .env\n"
        "Get it in Supabase → Project Settings → Database → Connection string (pooled). "
        "Use a READ-ONLY DB user. Example:\n"
        "SUPABASE_PG_CONN=postgresql+psycopg2://ro_user:password@host:6543/postgres"
    )

ENGINE = create_engine(PG_CONN, pool_pre_ping=True, future=True)

# Your table has hyphens, so it must be quoted in SQL.
TABLE_LOGICAL_NAME = "CTConsumoSAB-APD-MTY-CDT"
TABLE_SQL = 'public."CTConsumoSAB-APD-MTY-CDT"'  # quoted identifier for SQL

ROW_LIMIT_DEFAULT = 50000
STATEMENT_TIMEOUT_MS = 8000





# --- MAP CONFIG ---
MAP_CSV_PATH = r'C:\Users\cabal\Desktop\repos\multiple_agents\agent_query\Tabla_tiendas_ubicacion_CT.csv'

def _load_map_data() -> pd.DataFrame:
    df = pd.read_csv(MAP_CSV_PATH)

    # Renombra a los nombres estándar que espera el frontend
    df = df.rename(columns={
        'locationName': 'Tiendas',
        'locationCode': 'Codigo',
        'latitud': 'Latitud',
        'longitud': 'Longitud',
    })

    # Valida columnas mínimas
    required = ['Tiendas', 'Codigo', 'Latitud', 'Longitud']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV requiere columnas {missing}")

    # Quita filas sin coordenadas
    df = df.dropna(subset=['Latitud', 'Longitud']).copy()

    # Volumen (si no existe en el CSV)
    if 'Volumen' not in df.columns:
        df['Volumen'] = np.random.uniform(0, 1, len(df))
    else:
        # Normaliza Volumen a [0,1] para tamaño de burbuja
        s = pd.to_numeric(df['Volumen'], errors='coerce').fillna(0.0)
        rng = s.max() - s.min()
        df['Volumen'] = 0.0 if rng == 0 else (s - s.min())/rng

    return df[['Tiendas', 'Codigo', 'Latitud', 'Longitud', 'Volumen']]

# ------------- UTILS -------------
def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def _normalize_table_name(tbl_expr) -> str:
    """
    Return bare table name without schema/quotes for allowlisting checks.
    """
    # sqlglot Table may have catalog, db (schema), and this.name
    return (tbl_expr.name or "").replace('"', '')

def _is_select_only(sql: str) -> bool:
    try:
        trees = sqlglot.parse(sql, read="postgres")
    except ParseError:
        return False

    for t in trees:
        # Must contain a Select and no mutating/DDL statements
        if t.find(sqlglot.expressions.Select) is None:
            return False

        # Defensive banned-node list: only check nodes that exist in this sqlglot version
        banned_node_names = [
            "Insert", "Update", "Delete", "Create", "Drop", "Alter",
            "Truncate", "Grant", "Revoke",
        ]
        for name in banned_node_names:
            Node = getattr(sqlglot.expressions, name, None)
            if Node is not None and t.find(Node) is not None:
                return False

        # Banned functions
        for func in t.find_all(sqlglot.expressions.Func):
            if func.name and func.name.lower() in {"pg_sleep"}:
                return False

        # Table allowlist (only our CT table)
        for tbl in t.find_all(sqlglot.expressions.Table):
            if _normalize_table_name(tbl) != TABLE_LOGICAL_NAME:
                return False

    return True
def _ensure_limit(sql: str, default_limit: int) -> str:
    """
    Garantiza un LIMIT sin depender de parsear 'LIMIT 50000' como AST.
    Si no hay LIMIT, envuelve la consulta en un subquery y agrega LIMIT.
    """
    # Quitar ';' finales por seguridad
    sql = sql.strip().rstrip(";")

    try:
        trees = sqlglot.parse(sql, read="postgres")
    except ParseError:
        # Si no pudimos parsear, aplicamos wrapper seguro con LIMIT
        return f"SELECT * FROM ({sql}) AS _lim LIMIT {default_limit}"

    # Revisar si algún SELECT ya trae LIMIT
    has_limit = False
    for t in trees:
        for sel in t.find_all(sqlglot.expressions.Select):
            if sel.args.get("limit") is not None:
                has_limit = True
                break
        if has_limit:
            break

    if has_limit:
        return sql

    # Agregar LIMIT con wrapper (evita reescrituras AST frágiles)
    return f"SELECT * FROM ({sql}) AS _lim LIMIT {default_limit}"


def _rewrite_table_to_quoted(sql: str) -> str:
    sql = sql.strip()
    patterns = [
        (r'(?i)\bfrom\s+public\.CTConsumoSAB-APD-MTY-CDT\b', f'FROM {TABLE_SQL}'),
        (r'(?i)\bjoin\s+public\.CTConsumoSAB-APD-MTY-CDT\b', f'JOIN {TABLE_SQL}'),
        (r'(?i)\bfrom\s+CTConsumoSAB-APD-MTY-CDT\b',        f'FROM {TABLE_SQL}'),
        (r'(?i)\bjoin\s+CTConsumoSAB-APD-MTY-CDT\b',        f'JOIN {TABLE_SQL}'),
    ]
    for pat, repl in patterns:
        sql = re.sub(pat, repl, sql)
    return sql

def _run_sql_readonly(sql: str) -> pd.DataFrame:
    with ENGINE.connect() as conn:
        conn.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
        conn.execute(text(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT_MS}ms'"))
        return pd.read_sql(text(sql), conn)

# ------------- TOOLS -------------
class SQLArgs(BaseModel):
    sql: str = Field(..., description="A full SELECT (Postgres dialect). Use table public.\"CTConsumoSAB-APD-MTY-CDT\" only.")

def run_sql(sql: str) -> str:
    sql = _rewrite_table_to_quoted(sql.strip().rstrip(";"))
    if not _is_select_only(sql):
        return "SQL rejected: only read-only SELECTs on table public.\"CTConsumoSAB-APD-MTY-CDT\" are allowed."

    sql2 = _ensure_limit(sql, ROW_LIMIT_DEFAULT)
    try:
        df = _run_sql_readonly(sql2)
    except Exception as e:
        return f"SQL error: {e}"

    meta = f"rows={len(df)}, cols={list(df.columns)}"
    preview_md = df.head(10).to_markdown(index=False)
    preview_json = df.head(10).to_json(orient="records")  # 👈
    # 👇 añade el bloque JSON para que el agente lo lea fácil y *cierre* la respuesta
    return f"{meta}\n\nPREVIEW:\n{preview_md}\n\nDF_JSON_HEAD:\n{preview_json}"

RunSQLTool = StructuredTool.from_function(
    func=run_sql,
    name="run_sql",
    description="Execute a validated SELECT on Supabase/Postgres and return a preview.",
    args_schema=SQLArgs,
)

class SchemaArgs(BaseModel):
    pass

def get_schema() -> str:
    return (
        "Tables (allowlist):\n"
        f"{TABLE_SQL} (\n"
        "  row_id BIGINT (identity, PK),\n"
        "  sku BIGINT,\n"
        "  locationId BIGINT,\n"
        "  amount DOUBLE PRECISION,\n"
        "  cost DOUBLE PRECISION,\n"
        "  revenue DOUBLE PRECISION,\n"
        "  date DATE,\n"
        "  Margen DOUBLE PRECISION\n"
        ")\n"
        "Notes:\n"
        "- earnings = revenue - cost\n"
        "- 'most selling' means highest SUM(amount) unless otherwise stated\n"
        "- Use WHERE locationId = <storeId>\n"
        "- For weekly: date_trunc('week', date) as wk\n"
    )

SchemaTool = StructuredTool.from_function(
    func=get_schema,
    name="get_schema",
    description="Return the available tables/columns so you can write SQL.",
    args_schema=SchemaArgs,
)

class PlotArgs(BaseModel):
    data_sql: str = Field(..., description="SELECT that returns x + numeric y columns.")
    x: str = Field(..., description="x-axis column (e.g., date or wk).")
    y_cols: List[str] = Field(..., description="One or more numeric columns to plot.")
    title: Optional[str] = Field(None, description="Optional chart title.")

def plot_from_sql(data_sql: str, x: str, y_cols: List[str], title: Optional[str] = None) -> str:
    data_sql = _rewrite_table_to_quoted(data_sql.strip().rstrip(";"))
    if not _is_select_only(data_sql):
        return "SQL rejected in plot: only SELECT on the allowlisted table."

    # Limita filas para que no se tarde
    data_sql2 = _ensure_limit(data_sql, 5000)

    try:
        df = _run_sql_readonly(data_sql2)
    except Exception as e:
        return f"SQL error: {e}"

    if x not in df.columns or any(col not in df.columns for col in y_cols):
        return f"Missing columns. Available: {list(df.columns)}"

    # Ordena por x si aplica
    try:
        df = df.sort_values(by=[x])
    except Exception:
        pass

    # Downsample si hay demasiados puntos (para no generar PNG gigante)
    if len(df) > 3000:
        step = max(1, len(df) // 1500)
        df = df.iloc[::step].copy()

    # Layout sin cortes y tamaño moderado
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for col in y_cols:
        ax.plot(df[x].values, df[col].values, label=col)
    ax.set_xlabel(x); ax.set_ylabel("value")
    ax.set_title(title or "Chart")
    ax.legend(loc="best")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)  # sin bbox_inches="tight"
    plt.close(fig)
    buf.seek(0)

    # Guarda en cache y devuelve solo el ID
    pid = str(uuid4())
    _cache_put(pid, buf.getvalue())
    return f"PLOT_ID:{pid}"

PlotTool = StructuredTool.from_function(
    func=plot_from_sql,
    name="plot_from_sql",
    description="Run a SELECT, then plot y columns vs x. Returns base64 PNG.",
    args_schema=PlotArgs,
)

# ------------- AGENT -------------
SYSTEM = f"""You are the Treviño Data Agent. You can write your own SQL.
Workflow:
1) Call get_schema first if needed.
2) Plan briefly, then call run_sql with a single SELECT on {TABLE_SQL}.
3) If a graph is requested, call plot_from_sql with a SELECT that returns tidy columns.

Rules:
- Read-only SELECTs only. No DML/DDL. No pg_sleep.
- Only table allowed: {TABLE_SQL}.
- 'Most selling' => rank by SUM(amount). 'Most revenue' => SUM(revenue). 'Most earnings' => SUM(revenue - cost).
- If the user provides a store id, that maps to "locationId" (keep the double quotes around camelCase).
- If no dates given, use full history.
- For weekly series use: date_trunc('week', date) AS wk, GROUP BY wk ORDER BY wk.
- Return concise answers.
- When you return a plot DO NOT include any base64. The plot tool returns a line with the id. Output exactly:
  PLOT_ID:{{{{uuid}}}}
  (one single line, no markdown fences, no base64)
"""
def build_agent(model_name: str = "gpt-4o-mini") -> AgentExecutor:
    llm = ChatOpenAI(model=model_name, temperature=0)
    tools = [SchemaTool, RunSQLTool, PlotTool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    # 👇 agrega max_iterations y early_stopping_method
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,                 # útil para ver qué está haciendo
        max_iterations=8,
        early_stopping_method="generate",
    )


AGENT = build_agent()

# ------------- FASTAPI (optional) -------------
app = FastAPI(title="Treviño Free-SQL Agent")
PLOT_CACHE: dict[str, bytes] = {}
def _cache_put(pid: str, data: bytes) -> str:
    PLOT_CACHE[pid] = data
    return pid

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class AskPayload(BaseModel):
    prompt: str

@app.post("/ask")
def ask(p: AskPayload, request: Request):
    out = AGENT.invoke({"input": p.prompt})["output"]

    # 1) Preferimos PLOT_ID (robusto: multiline/insensitive)
    m_id = re.search(r"PLOT_ID:\s*([0-9a-fA-F-]{36})", out, flags=re.I | re.S)
    if m_id:
        text_part = out[:m_id.start()].strip()
        pid = m_id.group(1)
        url = str(request.url_for("get_plot", plot_id=pid))
        return {"text": text_part, "plot_url": url, "plot_id": pid}

    # 2) Fallback: si vino base64, lo cacheamos y devolvemos link
    m_b64 = re.search(r"PLOT_BASE64_PNG:\s*([A-Za-z0-9+/=\r\n]+)", out, flags=re.S)
    if m_b64:
        text_part = out[:m_b64.start()].strip()
        raw = m_b64.group(1)
        cleaned = re.sub(r"[^A-Za-z0-9+/=]", "", raw)
        try:
            img = base64.b64decode(cleaned, validate=True)
        except binascii.Error:
            img = None

        if img and img.startswith(b"\x89PNG\r\n\x1a\n"):
            pid = str(uuid4())
            PLOT_CACHE[pid] = img
            url = str(request.url_for("get_plot", plot_id=pid))
            return {"text": text_part, "plot_url": url, "plot_id": pid}

        return {"text": text_part}

    # 3) Sin gráfica
    return {"text": out}

@app.get("/plots/{plot_id}", name="get_plot")
def get_plot(plot_id: str):
    data = PLOT_CACHE.get(plot_id)
    if not data:
        raise HTTPException(status_code=404, detail="Plot not found or expired")
    return Response(content=data, media_type="image/png")

@app.get("/db_check")
def db_check():
    try:
        with ENGINE.connect() as conn:
            conn.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
            row = conn.execute(text("select current_user, current_database(), inet_server_addr(), inet_server_port()")).fetchone()
        return {"ok": True, "db": list(row)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llm_check")
def llm_check():
    try:
        _ = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke("ping")
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/map_data")
def map_data():
    try:
        df = _load_map_data()
        out = df.to_dict(orient="records")
        return JSONResponse(out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading map data: {e}")
HTML_UI = r"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Treviño Agent · Mapa + Chat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    :root { --gap: 12px; --bg:#0b0d10; --card:#14181d; --txt:#e9eef5; --muted:#9fb0c3; --brand:#46a6ff; }
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; background: var(--bg); color: var(--txt); font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans'; }
    .app { display: grid; grid-template-columns: 1fr 380px; grid-auto-rows: 100%; gap: var(--gap); height: 100%; padding: var(--gap); }
    .pane { background: var(--card); border-radius: 14px; overflow: hidden; display: flex; flex-direction: column; }
    .map-header, .chat-header { padding: 10px 14px; border-bottom: 1px solid #202733; display:flex; align-items:center; gap:10px; }
    .map-header h2, .chat-header h2 { font-size: 16px; margin: 0; font-weight: 600; }
    .map-wrap { flex: 1; min-height: 0; }
    #map { width: 100%; height: 100%; }
    .chat { display:flex; flex-direction:column; height:100%; }
    .chat-body { flex:1; overflow:auto; padding: 10px 14px; display:flex; flex-direction:column; gap:10px; }
    .msg { background:#0e1319; border:1px solid #1e2632; padding:10px 12px; border-radius:12px; max-width: 100%; white-space: pre-wrap; }
    .msg.agent { background:#0f1722; border-color:#1d2836; }
    .msg.user  { background:#101820; border-color:#1c242e; }
    .chat-input { border-top: 1px solid #202733; padding:10px; display:flex; gap:8px; }
    .chat-input textarea { resize: none; flex:1; height:74px; background:#0b1118; color:var(--txt); border:1px solid #1f2733; border-radius:10px; padding:10px; outline:none; }
    .btn { background: var(--brand); color: #00233f; border: none; border-radius: 10px; padding: 10px 14px; font-weight: 700; cursor: pointer; }
    .btn:disabled{ opacity:.6; cursor:not-allowed; }
    .tiny { color: var(--muted); font-size: 12px; }
    .pill { padding: 3px 8px; border-radius:999px; background:#0a121b; border:1px solid #1e2733; font-size:12px; color:#9fb0c3; }
    .plot-thumb { width:100%; border-radius:10px; border:1px solid #1f2733; margin-top:8px; }
    @media (max-width: 1024px) {
      .app { grid-template-columns: 1fr; }
      .pane.chat { height: 48vh; }
      .pane.map  { height: 52vh; }
    }
  </style>
</head>
<body>
  <div class="app">
    <section class="pane map">
      <div class="map-header">
        <span class="pill">Mapa</span>
        <h2>Comercial Treviño · Tiendas</h2>
        <span class="tiny" id="map-meta"></span>
      </div>
      <div class="map-wrap"><div id="map"></div></div>
    </section>

    <aside class="pane chat">
      <div class="chat-header">
        <span class="pill">Agente</span>
        <h2>Consultas SQL y Gráficas</h2>
      </div>
      <div class="chat-body" id="chat"></div>
      <div class="chat-input">
        <textarea id="prompt" placeholder="Pregúntame algo, p. ej.: 'Top 5 productos por ventas en locationId=123 esta semana'"></textarea>
        <button id="send" class="btn">Enviar</button>
      </div>
    </aside>
  </div>

<script>
const chatEl   = document.getElementById('chat');
const promptEl = document.getElementById('prompt');
const sendBtn  = document.getElementById('send');
const mapMeta  = document.getElementById('map-meta');
const mapEl    = document.getElementById('map');

function addMsg(text, who='agent', plotUrl=null) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.textContent = text;
  if (plotUrl) {
    const img = document.createElement('img');
    img.className = 'plot-thumb';
    img.src = plotUrl;
    img.alt = 'Gráfica';
    div.appendChild(img);
  }
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

async function sendPrompt() {
  const q = (promptEl.value || '').trim();
  if (!q) return;
  sendBtn.disabled = true;
  addMsg(q, 'user');
  promptEl.value = '';
  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'content-type':'application/json' },
      body: JSON.stringify({ prompt: q })
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const text = data.text || '(Sin texto)';
    const plot = data.plot_url || null;
    addMsg(text, 'agent', plot);
  } catch (err) {
    addMsg('Error: ' + err.message, 'agent');
  } finally {
    sendBtn.disabled = false;
  }
}
sendBtn.addEventListener('click', sendPrompt);
promptEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    sendPrompt();
  }
});

// Helper para enviar texto directo (lo usa el click del mapa)
async function sendPromptWithText(q) {
  try {
    sendBtn.disabled = true;
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'content-type':'application/json' },
      body: JSON.stringify({ prompt: q })
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    addMsg(data.text || '(Sin texto)', 'agent', data.plot_url || null);
  } catch (err) {
    addMsg('Error: ' + err.message, 'agent');
  } finally {
    sendBtn.disabled = false;
  }
}

// -------- MAPA (Plotly JS con scattermapbox estilo OpenStreetMap) --------
async function loadMap() {
  try {
    const res = await fetch('/map_data');
    if (!res.ok) throw new Error('No se pudo cargar /map_data');
    const rows = await res.json();
    mapMeta.textContent = rows.length + ' ubicaciones';

    const sizes = rows.map(r => 8 + 24 * ((r.Volumen ?? 0)));
    const trace = {
      type: 'scattermapbox',
      lat: rows.map(r => r.Latitud),
      lon: rows.map(r => r.Longitud),
      text: rows.map(r => r.Tiendas),
      customdata: rows.map(r => r.Codigo), // locationCode
      hovertemplate:
        '<b>%{text}</b><br>' +
        'Código: %{customdata}<br>' +
        'Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>',
      mode: 'markers',
      marker: { size: sizes }
    };

    const lats = rows.map(r => r.Latitud);
    const lons = rows.map(r => r.Longitud);
    const center = {
      lat: lats.reduce((a,b)=>a+b,0)/lats.length,
      lon: lons.reduce((a,b)=>a+b,0)/lons.length
    };

    const layout = {
      dragmode: 'zoom',
      mapbox: { style: 'open-street-map', center, zoom: 7 },
      margin: { l:0, r:0, t:0, b:0 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent'
    };

    await Plotly.newPlot(mapEl, [trace], layout, {displaylogo:false, responsive:true});
    window.addEventListener('resize', () => Plotly.Plots.resize(mapEl));

    // Click en marcador => pregunta automática al agente
    const storeRows = rows;
    mapEl.on('plotly_click', (ev) => {
      const pt = ev.points?.[0];
      if (!pt) return;
      const idx = pt.pointIndex;
      const row = storeRows[idx] || {};
      const nombre = row.Tiendas ?? '(sin nombre)';
      const code   = row.Codigo ?? '(sin código)';
      const locId  = row.locationId ?? row.LocationId ?? null; // si existiera

      let prompt = `Top 10 SKUs por revenue en toda la historia`;
      if (locId != null && String(locId).length) {
        prompt = `Top 10 SKUs por revenue para locationId=${locId} usando toda la historia`;
      }
      addMsg(`🗺️ ${nombre} [${code}]\n${prompt}`, 'user');
      sendPromptWithText(prompt);
    });
  } catch (e) {
    console.error(e);
    mapMeta.textContent = 'Error cargando mapa';
  }
}
loadMap();
</script>
</body>
</html>
"""
@app.get("/ui", name="ui")
def ui():
    return Response(content=HTML_UI, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
