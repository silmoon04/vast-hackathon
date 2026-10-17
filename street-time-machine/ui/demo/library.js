/* Local footage groups share a clock only within their own recording session. */
(function () {
  "use strict";
  const { el } = STM;
  STM.libraryScenes = {};
  let loading = null, polling = null;
  STM.loadLibrary = async () => {
    try {
      const r = await fetch("/api/find/library", { cache: "no-store" });
      if (!r.ok) return;
      const data = await r.json();
      STM.libraryScenes = Object.fromEntries((data.scenes || []).map(s => [s.name, s]));
      const built = STM.sceneList.filter(([name]) => !name.startsWith("library-"));
      STM.sceneList = built.concat((data.scenes || []).map(s => [s.name, s.title, s.kind]));
    } catch (e) { console.warn("Footage library unavailable", e); }
  };
  STM.openLibraryScene = async name => {
    if (loading) return;
    loading = name;
    try {
      await STM.loadScene(name);
      STM.sceneName = name;
      STM.desk.reset();
      const url = new URL(location.href); url.searchParams.set("scene", name);
      history.replaceState(null, "", url);
      await STM.go("desk");
      if (STM.chat) STM.chat.note(`Opened ${STM.scene().title}. ${STM.scene().cameras.length} cameras. ${STM.scene().clock_note || "Each recording group has its own clock."}`);
      if (name.startsWith("library-") && STM.scene().index_status !== "ready") STM.indexLibraryScene(name);
    } catch (e) { STM.toast(STM.esc(e.message)); }
    finally { loading = null; }
  };
  STM.indexLibraryScene = async name => {
    try {
      const r = await fetch(`/api/find/library/${encodeURIComponent(name)}/index`, { method: "POST" });
      if (!r.ok) throw new Error(`Index unavailable (${r.status})`);
      await STM.loadLibrary();
      if (!polling) polling = setInterval(refresh, 4000);
      if (STM.sceneName === name && STM.current === "desk") {
        STM.scene().index_status = (STM.libraryScenes[name] || {}).status || "queued";
        await STM.go("desk");
      }
    } catch (e) { STM.toast(STM.esc(e.message)); }
  };
  async function refresh() {
    await STM.loadLibrary();
    const name = STM.sceneName, info = STM.libraryScenes[name], current = STM.scene();
    if (info && current && info.status === "ready" && current.index_status !== "ready") {
      delete STM.scenes[name];
      await STM.loadScene(name);
      STM.desk.reset();
      await STM.go(STM.current || "desk");
      if (STM.chat) STM.chat.note(`${info.title} is indexed. Search its ${STM.scene().tracks.length} tracks now.`);
    } else {
      const node = document.querySelector("#library-index-status");
      if (node && info) node.textContent = statusText(info.status);
      if (info && current && info.status === "error" && current.index_status !== "error") {
        current.index_status = "error";
        if (STM.current === "desk") await STM.go("desk");
      }
    }
    if (!Object.values(STM.libraryScenes).some(s => ["queued", "indexing"].includes(s.status))) {
      clearInterval(polling); polling = null;
    }
  }
  function statusText(status) {
    return { ready: "Search ready", indexing: "Indexing footage…", queued: "Index queued", error: "Index needs retry" }[status] || "Playback ready · search not indexed";
  }
  STM.libraryPicker = fileIn => {
    const select = el("select", { id: "desk-scene-select", "aria-label": "Footage recording group", class: "btn btn-sm", style: { maxWidth: "310px", minWidth: "205px", background: "var(--card)" }, onchange: e => STM.openLibraryScene(e.target.value) });
    for (const [name, title, kind] of STM.sceneList) {
      const info = STM.libraryScenes[name];
      const count = info ? ` · ${info.cameras} cams` : "";
      const suffix = info ? ` · ${info.status === "ready" ? "search ready" : "playback"}` : "";
      select.append(el("option", { value: name, selected: name === STM.sceneName }, `${title}${count}${kind === "synthetic" ? " · synthetic" : ""}${suffix}`));
    }
    return el("div", { class: "row", style: { flexWrap: "wrap", justifyContent: "flex-end" } }, select,
      el("button", { class: "btn btn-sm", type: "button", id: "desk-add-video", onclick: () => fileIn.click() }, "Add a video"), fileIn);
  };
  STM.libraryNotice = () => {
    const s = STM.scene(), info = STM.libraryScenes[STM.sceneName];
    if (!info) return null;
    const state = info.status || s.index_status;
    if (["queued", "indexing"].includes(state) && !polling) polling = setInterval(refresh, 4000);
    const resolution = info.resolution_note || "Analysis uses up to 720p; lower-resolution sources retain their original detail.";
    const row = el("div", { class: "card", id: "library-status", style: { padding: "10px 14px", margin: "-4px 0 14px", fontSize: "12px", display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" } },
      el("b", { id: "library-index-status" }, statusText(state)),
      el("span", { class: "muted" }, `${info.kind === "synthetic" ? "Synthetic footage. " : ""}${s.clock_note || info.clock_note || "Elapsed time within this recording group."} ${resolution} ${info.route_note || s.route_note || (s.notices || []).join(" ")}`));
    const licenses = [...new Set(s.cameras.map(c => c.license).filter(Boolean))];
    if (licenses.length) row.append(el("span", { class: "muted" }, `Source licence: ${licenses.join(" / ")}.`));
    if (state !== "ready") row.append(el("button", { type: "button", class: "btn btn-sm", onclick: () => STM.indexLibraryScene(STM.sceneName), disabled: ["queued", "indexing"].includes(state) }, state === "error" ? "Retry indexing" : "Index for search"));
    return row;
  };
})();
