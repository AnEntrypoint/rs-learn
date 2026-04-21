import * as webjsx from "webjsx";
const h = webjsx.createElement;

const state = { data: null, error: null, now: new Date() };
window.__debug = window.__debug || {};
window.__debug.app = { state, render: () => render(), reload: () => main() };

const fmtDate = (d) => {
  const y = d.getUTCFullYear(); const m = String(d.getUTCMonth()+1).padStart(2,"0"); const da = String(d.getUTCDate()).padStart(2,"0");
  return `${y}.${m}.${da}`;
};

function Dateline({ now, buildSha }) {
  return h("div", { class: "dateline" },
    h("span", { class: "mono" }, "247420"),
    h("span", { class: "sep" }, "//"),
    h("span", { class: "mono" }, "an entrypoint"),
    h("span", { class: "sep" }, "//"),
    h("span", { class: "mono" }, "rs-learn"),
    h("span", { class: "sep" }, "//"),
    h("span", { class: "mono" }, fmtDate(now)),
    h("span", { class: "sep" }, "//"),
    h("span", { class: "mono live" }, "· live"),
    h("span", { style: "flex:1" }),
    h("a", { class: "mono", href: "https://github.com/AnEntrypoint/rs-learn", target: "_blank", rel: "noopener" }, "source ↗"),
  );
}

function Hero({ project }) {
  const p = project || { title: "rs-learn", tag: "// pure rust · continual learning · acp wrapper", lede: "persistent memory, adaptive routing, three background learning loops. wraps any agent client protocol stdio agent. ships as one binary.", stamp: "probably emerging" };
  return h("section", { class: "hero" },
    h("div", { class: "section-label" }, "project // rs-learn · v0.1 · pre"),
    h("h1", { class: "display" }, p.title),
    h("div", { class: "tag" }, p.tag),
    h("p", { class: "lede" }, p.lede),
    h("div", {},
      h("span", { class: "stamp" }, p.stamp),
    ),
  );
}

function Pipeline({ stages }) {
  const s = stages || [];
  return h("div", { class: "pipeline" },
    ...s.map((st, i) => h("div", { class: "stage" },
      h("span", { class: "num mono" }, String(i+1).padStart(2, "0")),
      h("div", { class: "caps" }, st.phase),
      h("div", { class: "name" }, st.name),
      h("div", { class: "hint" }, st.hint),
    )),
  );
}

function Loops({ loops }) {
  const l = loops || [];
  return h("div", { class: "grid-3" },
    ...l.map((lp) => h("article", { class: "loop-card" },
      h("div", { class: "accent-line" }),
      h("div", { class: "badge" }, lp.badge),
      h("h3", { class: "display" }, lp.name),
      h("div", { class: "freq" }, lp.freq + " · " + lp.method),
      h("p", {}, lp.body),
    )),
  );
}

function Rows({ items }) {
  const it = items || [];
  return h("div", { class: "rows" },
    ...it.map((r) => h("div", { class: "row" },
      h("div", { class: "code" }, r.code),
      h("div", { class: "title display" }, r.title),
      h("div", { class: "meta" }, r.meta),
    )),
  );
}

function Targets({ list }) {
  const tgts = list || [];
  return h("div", { style: "margin-top: 16px" },
    ...tgts.map((t) => h("span", { class: "target-chip " + (t.status === "green" ? "green" : "") }, t.name)),
  );
}

function Numbers({ stats }) {
  const s = stats || [];
  return h("div", { class: "grid-4" },
    ...s.map((n) => h("div", {},
      h("div", { class: "num-big" }, n.value),
      h("div", { class: "num-label" }, n.label),
    )),
  );
}

function Section({ label, title, children }) {
  return h("section", {},
    h("div", { class: "section-label" }, label),
    h("h2", { class: "section-title" }, title),
    children,
  );
}

function App({ data, now }) {
  if (!data) return h("div", { style: "padding: 64px; font-family: JetBrains Mono, monospace" }, "loading // ...");
  const d = data;
  return h("div", {},
    h(Dateline, { now }),
    h("main", {},
      h(Hero, { project: d.project }),
      h(Section, { label: d.pipeline.label, title: d.pipeline.title },
        h("p", { class: "prose" }, d.pipeline.intro),
        h(Pipeline, { stages: d.pipeline.stages }),
      ),
      h(Section, { label: d.loops.label, title: d.loops.title },
        h("p", { class: "prose" }, d.loops.intro),
        h(Loops, { loops: d.loops.items }),
      ),
      h(Section, { label: d.layers.label, title: d.layers.title },
        h("p", { class: "prose" }, d.layers.intro),
        h(Rows, { items: d.layers.items }),
      ),
      h(Section, { label: d.debug.label, title: d.debug.title },
        h("p", { class: "prose" }, d.debug.intro),
        h("div", { class: "diagram" }, d.debug.ascii),
        h("div", { class: "cli" },
          ...d.debug.curls.map((c) => h("div", {},
            h("span", { class: "prompt" }, "$ "),
            h("span", {}, c),
          )),
        ),
      ),
      h(Section, { label: d.numbers.label, title: d.numbers.title },
        h(Numbers, { stats: d.numbers.stats }),
      ),
      h(Section, { label: d.ci.label, title: d.ci.title },
        h("p", { class: "prose" }, d.ci.intro),
        h(Targets, { list: d.ci.targets }),
      ),
      h(Section, { label: d.quote.label, title: d.quote.title },
        h("blockquote", { class: "quote" }, d.quote.text),
      ),
    ),
    h("footer", {},
      h("span", {}, "247420 / mmxxvi"),
      h("span", {}, "probably emerging ", h("span", { class: "spin" }, "✱")),
      h("span", {}, "built in public"),
      h("a", { href: "https://github.com/AnEntrypoint/rs-learn" }, "source"),
    ),
  );
}

function render() {
  webjsx.applyDiff(document.getElementById("root"), App({ data: state.data, now: state.now }));
}

async function main() {
  try {
    const res = await fetch("./data/index.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`fetch index.json: ${res.status}`);
    state.data = await res.json();
    state.now = new Date();
    render();
  } catch (e) {
    state.error = e;
    document.getElementById("root").innerHTML = `<div style="padding:64px;font-family:JetBrains Mono,monospace">error // ${e.message}</div>`;
    throw e;
  }
}

main();
