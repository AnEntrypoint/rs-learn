import * as webjsx from "webjsx";
const h = webjsx.createElement;

const state = { data: null, error: null, tab: "overview", copied: null };
window.__debug = window.__debug || {};
window.__debug.app = { state, render: () => render(), reload: () => main() };

const fmtDate = (d) => `${d.getUTCFullYear()}.${String(d.getUTCMonth()+1).padStart(2,"0")}.${String(d.getUTCDate()).padStart(2,"0")}`;

const TABS = ["overview", "pipeline", "loops", "layers", "observability", "numbers", "ci"];

function Topbar() {
  return h("header", { class: "app-topbar" },
    h("span", { class: "brand" }, "247420", h("span", { class: "slash" }, " / "), "rs-learn"),
    h("nav", {},
      ...TABS.map(t => h("a", {
        href: "#", key: t,
        class: state.tab === t ? "active" : "",
        onclick: (e) => { e.preventDefault(); state.tab = t; render(); }
      }, t)),
      h("a", { href: "https://github.com/AnEntrypoint/rs-learn", target: "_blank", rel: "noopener" }, "source ↗"),
    ),
  );
}

function Crumb({ data }) {
  return h("div", { class: "app-crumb" },
    h("span", {}, "247420"), h("span", { class: "sep" }, "›"),
    h("span", {}, "rs-learn"), h("span", { class: "sep" }, "›"),
    h("span", { class: "leaf" }, state.tab),
    h("span", { style: "margin-left:auto;display:flex;gap:8px;align-items:center" },
      h("span", { class: "chip accent" }, "● " + (data?.project?.stamp || "probably emerging")),
      h("span", { class: "chip dim" }, "v0.1 · pre"),
    ),
  );
}

function Side() {
  const groups = [
    { group: "project", items: [["◆", "overview"], ["§", "pipeline"], ["§", "loops"], ["§", "layers"]] },
    { group: "runtime", items: [["›", "observability"], ["›", "numbers"], ["›", "ci"]] },
    { group: "links", items: [["↗", "source"], ["↗", "crates.io"], ["↗", "releases"]] },
  ];
  return h("aside", { class: "app-side" }, ...groups.flatMap(sec => [
    h("div", { class: "group", key: sec.group }, sec.group),
    ...sec.items.map(([glyph, label], i) => h("a", {
      key: sec.group + i, href: "#",
      class: state.tab === label ? "active" : "",
      onclick: (e) => { e.preventDefault(); if (TABS.includes(label)) { state.tab = label; render(); } },
    },
      h("span", { class: "glyph" }, glyph),
      h("span", {}, label),
    )),
  ]));
}

function Copyable({ text, id }) {
  const on = () => {
    navigator.clipboard?.writeText(text);
    state.copied = id; render();
    setTimeout(() => { state.copied = null; render(); }, 1200);
  };
  return h("div", { class: "cli" },
    h("span", { class: "prompt" }, "$"),
    h("span", { class: "cmd" }, text),
    h("span", { class: "copy", onclick: on }, state.copied === id ? "copied" : "copy"),
  );
}

function Receipt({ rows }) {
  return h("table", { class: "kv" },
    h("tbody", {}, ...rows.map(([k, v], i) =>
      h("tr", { key: i }, h("td", {}, k), h("td", {}, v))
    )),
  );
}

function Overview({ data }) {
  const p = data.project;
  return [
    h("h1", {}, p.title),
    h("p", { class: "lede" }, p.lede),
    h("h3", {}, "install"),
    Copyable({ text: "cargo install rs-learn", id: "install-cargo" }),
    h("h3", {}, "run"),
    Copyable({ text: 'RS_LEARN_BACKEND=claude-cli RS_LEARN_CLAUDE_MODEL=haiku rs-learn', id: "run-claude" }),
    Copyable({ text: 'RS_LEARN_ACP_COMMAND="opencode acp" rs-learn', id: "run-acp" }),
    h("h3", {}, "receipt"),
    Receipt({ rows: [
      ["status", p.stamp || "probably emerging"],
      ["lang", "rust · tokio · libsql"],
      ["license", "MIT"],
      ["backends", "acp stdio · claude-cli"],
      ["memory", "libsql vec + graph + fts5 · single file"],
      ["loops", "instant · background · deep"],
    ]}),
  ];
}

function PipelineView({ data }) {
  const s = data.pipeline;
  return [
    h("h2", {}, s.title),
    h("p", { class: "lede" }, s.intro),
    h("div", { class: "panel" },
      h("div", { class: "panel-body" }, ...s.stages.map((st, i) =>
        h("div", { key: i, class: "row", style: "grid-template-columns:48px 120px 1fr" },
          h("span", { class: "code" }, String(i+1).padStart(2, "0")),
          h("span", { style: "font-family:var(--ff-mono);color:var(--panel-accent);text-transform:uppercase;font-size:12px;letter-spacing:0.08em" }, st.phase),
          h("span", { class: "title" }, st.name || st.hint, st.hint && st.name ? h("span", { class: "meta", style: "display:block;font-family:var(--ff-mono);font-size:12px;color:var(--panel-text-3);margin-top:4px" }, st.hint) : null),
        )
      )),
    ),
  ];
}

function LoopsView({ data }) {
  const l = data.loops;
  return [
    h("h2", {}, l.title),
    h("p", { class: "lede" }, l.intro),
    h("div", { class: "panel" },
      h("div", { class: "panel-body" }, ...l.items.map((lp, i) =>
        h("div", { key: i, class: "row", style: "grid-template-columns:120px 160px 1fr" },
          h("span", { class: "code" }, lp.badge),
          h("span", { style: "font-family:var(--ff-mono);color:var(--panel-accent);font-size:12px" }, lp.freq + " · " + lp.method),
          h("span", { class: "title" }, lp.name,
            h("span", { style: "display:block;font-family:var(--ff-mono);font-size:12px;color:var(--panel-text-2);margin-top:4px;line-height:1.5" }, lp.body),
          ),
        )
      )),
    ),
  ];
}

function LayersView({ data }) {
  const l = data.layers;
  return [
    h("h2", {}, l.title),
    h("p", { class: "lede" }, l.intro),
    h("div", { class: "panel" },
      h("div", { class: "panel-body" }, ...l.items.map((it, i) =>
        h("div", { key: i, class: "row", style: "grid-template-columns:140px 1fr auto" },
          h("span", { class: "code" }, it.code),
          h("span", { class: "title" }, it.title),
          h("span", { class: "meta" }, it.meta),
        )
      )),
    ),
  ];
}

function DebugView({ data }) {
  const d = data.debug;
  return [
    h("h2", {}, d.title),
    h("p", { class: "lede" }, d.intro),
    h("pre", { class: "panel", style: "padding:16px;overflow-x:auto;font-family:var(--ff-mono);font-size:12px;line-height:1.5;color:var(--panel-text-2);white-space:pre" }, d.ascii),
    h("div", { style: "display:flex;flex-direction:column;gap:8px;margin-top:16px" },
      ...d.curls.map((c, i) => Copyable({ text: c, id: "curl-" + i })),
    ),
  ];
}

function NumbersView({ data }) {
  const n = data.numbers;
  return [
    h("h2", {}, n.title),
    h("div", { class: "panel" },
      h("div", { class: "panel-body" }, ...n.stats.map((s, i) =>
        h("div", { key: i, class: "row", style: "grid-template-columns:1fr auto" },
          h("span", { class: "title" }, s.label),
          h("span", { style: "font-family:var(--ff-mono);font-size:20px;color:var(--panel-accent);font-weight:600" }, s.value),
        )
      )),
    ),
  ];
}

function CIView({ data }) {
  const c = data.ci;
  return [
    h("h2", {}, c.title),
    h("p", { class: "lede" }, c.intro),
    h("div", { class: "panel" },
      h("div", { class: "panel-body" }, ...c.targets.map((t, i) =>
        h("div", { key: i, class: "row", style: "grid-template-columns:24px 1fr auto" },
          h("span", {}, t.status === "green" ? "●" : "○"),
          h("span", { class: "title" }, t.name),
          h("span", { class: "chip " + (t.status === "green" ? "accent" : "dim") }, t.status),
        )
      )),
    ),
    data.quote ? h("blockquote", { class: "panel", style: "padding:24px;margin-top:24px;font-family:var(--ff-display);font-size:20px;line-height:1.4;color:var(--panel-text)" }, "“", data.quote.text, "”") : null,
  ];
}

function Status({ data }) {
  return h("footer", { class: "app-status" },
    h("span", { class: "item" }, "main"),
    h("span", { class: "item" }, "• rust + tokio"),
    h("span", { class: "item" }, "• libsql"),
    h("span", { class: "item" }, "• acp + claude-cli"),
    h("span", { class: "spread" }),
    h("span", { class: "item" }, fmtDate(new Date())),
    h("span", { class: "item" }, "• 247420 / mmxxvi"),
    h("span", { class: "item" }, "• " + (data?.project?.stamp || "probably emerging")),
  );
}

function Body({ data }) {
  switch (state.tab) {
    case "overview":      return Overview({ data });
    case "pipeline":      return PipelineView({ data });
    case "loops":         return LoopsView({ data });
    case "layers":        return LayersView({ data });
    case "observability": return DebugView({ data });
    case "numbers":       return NumbersView({ data });
    case "ci":            return CIView({ data });
    default:              return Overview({ data });
  }
}

function App({ data, error }) {
  if (error) return h("div", { class: "app" }, h("main", { class: "app-main narrow" }, h("h1", {}, "error"), h("p", { class: "lede" }, error.message)));
  if (!data) return h("div", { class: "app" }, h("main", { class: "app-main narrow" }, h("p", { class: "lede" }, "loading // ...")));
  return h("div", { class: "app" },
    Topbar(),
    Crumb({ data }),
    h("div", { class: "app-body" },
      Side(),
      h("main", { class: "app-main narrow" }, ...Body({ data })),
    ),
    Status({ data }),
  );
}

function render() {
  webjsx.applyDiff(document.getElementById("root"), App({ data: state.data, error: state.error }));
}

async function main() {
  try {
    const res = await fetch("./data/index.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`fetch index.json: ${res.status}`);
    state.data = await res.json();
    render();
  } catch (e) {
    state.error = e;
    render();
    throw e;
  }
}

main();
