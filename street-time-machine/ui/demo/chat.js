/* Desk agent chat. The model (DeepSeek V4.1 Flash via /api/find/route) only picks ONE tool; this page runs the tool
   over the index and writes every fact itself. Offline, a small local router picks the tool and says so.
   Voice: hold the mic -> 16 kHz WAV -> /api/find/transcribe (Canary on the team's Find Desk).
   Photo: colours are read from the picture in this browser and searched like words. Video: indexed by the desk server. */
(function () {
  "use strict";
  const { el, esc, sleep, icons, $ } = STM;
  let log, busy = false;

  /* ---------- offline router ---------- */
  const T = "(\\d{1,2}(?::\\d{2}){0,2}\\s*(?:am|pm)?)";
  function localWindow(text) {
    const a = {}, s = STM.scene(); let m;
    if ((m = text.match(new RegExp(`\\bbetween\\s+${T}\\s+(?:and|to|-)\\s+${T}`, "i")))) { a.after = m[1]; a.before = m[2]; }
    else {
      if ((m = text.match(new RegExp(`\\b(?:after|since|from)\\s+${T}`, "i")))) a.after = m[1];
      if ((m = text.match(new RegExp(`\\b(?:before|until|till|by)\\s+${T}`, "i")))) a.before = m[1];
      if (!a.after && !a.before && (m = text.match(new RegExp(`\\b(?:around|about|at)\\s+${T}`, "i")))) {
        const t = s.toT(m[1]); if (t != null) { a.after = s.clock(Math.max(0, t - 60)); a.before = s.clock(t + 60); }
      }
    }
    for (const k of ["after", "before"]) if (a[k] && s.toT(a[k]) == null) delete a[k];
    if ((m = text.match(/\bcam(?:era)?\s+([a-z])\b/i))) a.camera = m[1].toUpperCase();
    return a;
  }
  const LOCAL = [
    [/\b(start over|reset|clear)\b/i, "start_over"],
    [/\b(who is|name|identify|face|is that)\b/i, "cannot_do", { reason: "The desk never identifies people. It only matches clothing." }],
    [/\b(how many|how busy|count|anyone|anybody)\b/i, "count_people", null, true],
    [/\b(when (was|were|did)|what time|how long)\b/i, "when_seen"],
    [/\b(not (me|him|her|them|that)|wrong|no\b)/i, "reject_candidate", { number: 1 }],
    [/\b(yes|that'?s (me|him|her|them)|first|number one|#1\b)/i, "pick_candidate", { number: 1 }],
    [/\b(second|number two|#2\b)/i, "pick_candidate", { number: 2 }],
    [/\b(third|number three|#3\b)/i, "pick_candidate", { number: 3 }],
    [/\b(who came|came near|who took|touched|near)\b/i, "who_came_near"],
    [/\b(where did (they|he|she) go|follow|where is)\b/i, "follow_lead"],
    [/\b(where did i go|my route)\b/i, "where_did_i_go"],
    [/\b(show|play).*(moment|when)\b/i, "show_moment"],
    [/\b(ghost|trails|everyone)\b/i, "ghost_trails"],
  ];
  function localRoute(text) {
    const win = localWindow(text), hasColour = (() => { const l = STM.desk.parseLook(text); return !!(l.top || l.bottom); })();
    const num = text.match(/(?:\bnumber\s*|#)(\d)\b/i);
    if (num && !/\b(not|wrong)\b/i.test(text)) return { tool: "pick_candidate", args: { number: +num[1] } };
    for (const [re, tool, args, takesWin] of LOCAL) {
      if (!re.test(text)) continue;
      if (takesWin && hasColour && !/\b(how many|count|how busy)\b/i.test(text)) break;     // "anyone in blue after 16:52" is a search
      return { tool, args: takesWin ? win : (args || {}) };
    }
    if (Object.keys(win).length && !hasColour && STM.state.look) return { tool: "narrow_search", args: win };
    if (Object.keys(win).length && !hasColour) return { tool: "count_people", args: win };
    return { tool: "search_people", args: { description: text, refine: !!(STM.state.candidates.length && /\b(also|with|and|carrying)\b/i.test(text)), ...win } };
  }
  const status = live => { const n = $("#chatSub"); if (n) n.textContent = live ? (STM.showcase ? "DeepSeek V4.1 Flash picks each step" : "") : "Offline · built-in router"; };
  async function route(text) {
    const s = STM.scene(), st = STM.state;
    const state = { candidates: st.candidates.map(id => s.track(id).desc), subject: !!st.subject, lead: !!st.lead,
      clock: `${s.span()} on ${s.clock0.slice(0, 10)}`, cameras: s.cameras.map(c => `${c.id} ${c.name}`).join(", "), window: STM.desk.winText(st.window) };
    const t0 = performance.now();
    if (STM.live.route !== false) {
      try {
        const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 7000);
        const r = await fetch("/api/find/route", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, state }), signal: ctl.signal });
        clearTimeout(timer);
        if (r.ok) { const o = await r.json(); status(true); return { ...o, model: (o.model || "").split("/").pop(), live: true }; }
      } catch (e) {}
    }
    await sleep(380);
    status(false);
    const o = localRoute(text);
    return { ...o, model: "local router (offline)", ms: Math.round(performance.now() - t0), live: false };
  }

  /* ---------- message helpers ---------- */
  const scroll = () => { log.scrollTop = log.scrollHeight; };
  function me(text, via, img) {
    log.append(el("div", { class: "msg me" }, img ? el("img", { class: "ref", src: img, alt: "Photo reference" }) : null, text, via ? el("span", { class: "via" }, via) : null)); scroll();
  }
  function bot() {
    const m = el("div", { class: "msg bot" }, el("div", { class: "bubble" }, el("span", { class: "dots" }, el("i"), el("i"), el("i"))));
    log.append(m); scroll(); return m;
  }
  const say = (m, html) => { const b = el("div", { class: "bubble", html }); m.append(b); scroll(); return b; };
  async function steps(m, list) {               // staged progress for the video; the app answers straight away
    if (!STM.showcase) return;
    const box = el("div", { class: "steps" }); m.append(box);
    for (const [label, ms] of list) {
      const row = el("div", {}, el("span", { class: "spin", style: { width: "14px", height: "14px" } }), el("span", {}, label)); box.append(row); scroll();
      const t0 = performance.now(); await sleep(STM.reduced ? 0 : ms);
      row.firstChild.replaceWith(el("span", { class: "ck", html: icons.check }));
      row.append(el("span", { class: "muted mono", style: { marginLeft: "auto", fontSize: "11px" } }, `${Math.round(performance.now() - t0)} ms`));
    }
  }
  const cards = (s, ids, clickable) => el("div", { class: "cand-row" }, ...ids.slice(0, 6).map((id, i) => {
    const t = s.track(id), sc = (STM.state.scores || {})[id];
    return el(clickable ? "button" : "div", { class: "cand", type: "button", id: clickable ? `chat-cand-${i + 1}` : null, onclick: clickable ? () => tap("pick_candidate", { number: i + 1 }, `That's them, number ${i + 1}`) : null },
      el("img", { src: s.cropUrl(t), alt: t.desc }), el("div", {}, el("b", {}, `#${i + 1} · Cam ${t.cam}`), el("span", { class: "mono", style: { fontSize: "10.5px", color: "var(--ink-3)" } }, s.clock(t.t0)), el("br"),
        t.desc.replace(" lower body", "")));
  }));

  /* a small player inside the conversation: loops one sighting, with its own play/pause, and can hand off to the big player */
  const PLAY = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
  const PAUSE = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>';
  function clip(s, t, from, to) {
    const c = s.cam(t.cam), off = c.offset || 0;
    const a = Math.max(0, from ?? t.t0 - 1), b = Math.min(s.duration, to ?? t.t1 + 1);
    const v = el("video", { muted: true, playsinline: true, preload: "auto", src: s.videoUrl(c) }); v.muted = true;
    const bar = el("i"), clock = el("span", { class: "mono" }, s.clock(a));
    const pp = el("button", { class: "cp", type: "button", "aria-label": "Play or pause", html: PAUSE, onclick: () => (v.paused ? v.play().catch(() => {}) : v.pause()) });
    const box = el("div", { class: "clip" }, v,
      el("div", { class: "clip-hud" }, pp, el("span", {}, `Camera ${t.cam}`), clock, el("div", { class: "grow" }),
        el("button", { class: "cp", type: "button", onclick: () => { v.pause(); STM.desk.seek(a, t.cam); } }, "Open in player")),
      el("div", { class: "clip-bar" }, bar));
    v.addEventListener("loadedmetadata", () => { v.currentTime = Math.max(0, a - off); v.play().catch(() => {}); });
    v.addEventListener("timeupdate", () => {
      const st = v.currentTime + off;
      if (st >= b) v.currentTime = Math.max(0, a - off);
      bar.style.width = Math.min(100, Math.max(0, (st - a) / Math.max(1, b - a) * 100)) + "%"; clock.textContent = s.clock(st);
    });
    v.addEventListener("play", () => { pp.innerHTML = PAUSE; document.querySelectorAll(".clip video").forEach(o => { if (o !== v) o.pause(); }); });
    v.addEventListener("pause", () => { pp.innerHTML = PLAY; });
    return box;
  }
  const lookText = l => [l.top && `top <b>${l.top}</b>`, l.bottom && `lower <b>${l.bottom}</b>`].filter(Boolean).join(", ") || "that";

  /* the action a tool call stands for, in words: no raw arguments on screen */
  const LABEL = { search_people: "Search", narrow_search: "Narrow", when_seen: "When seen", count_people: "Count", pick_candidate: "Confirm", reject_candidate: "Remove",
    who_came_near: "Who came near", follow_lead: "Follow route", where_did_i_go: "Follow route", show_moment: "Play moment", ghost_trails: "Open timeline",
    start_over: "Start over", send_robot: "Send Rover 1", cannot_do: "Declined", photo_search: "Photo search", add_video: "Index new video" };
  function actionChip(tool, a = {}, meta) {
    const bits = [];
    if (a.description) { const l = STM.desk.parseLook(a.description); if (l.top) bits.push(`${l.top} top`); if (l.bottom) bits.push(`${l.bottom} lower`); if (!l.top && !l.bottom) bits.push(`“${String(a.description).slice(0, 28)}”`); }
    if (a.look) { if (a.look.top) bits.push(`${a.look.top} top`); if (a.look.bottom) bits.push(`${a.look.bottom} lower`); }
    if (a.number) bits.push(`#${a.number}`);
    if (a.camera) bits.push(`Camera ${String(a.camera).toUpperCase()}`); else if (a.camera === "") bits.push("any camera");
    if (a.after === "" && a.before === "") bits.push("any time");
    if (a.after && a.before) bits.push(`${a.after}–${a.before}`); else if (a.after) bits.push(`after ${a.after}`); else if (a.before) bits.push(`before ${a.before}`);
    if (a.refine) bits.push("adds to last search");
    if (a.file) bits.push(a.file);
    return el("div", { class: "toolcall" }, el("span", { class: "fn" }, LABEL[tool] || "Answer"), ...bits.map(b => el("span", { class: "arg" }, b)), el("span", { class: "ms" }, meta));
  }

  /* a strip of when each match was seen, plus chips that narrow by time bucket or camera */
  function timeStrip(s, ids, w) {
    const strip = el("div", { class: "tstrip" });
    if (w && (w.a != null || w.b != null)) { const a = Math.max(0, w.a ?? 0), b = Math.min(s.duration, w.b ?? s.duration); strip.append(el("span", { class: "w", style: { left: a / s.duration * 100 + "%", width: Math.max(0, b - a) / s.duration * 100 + "%" } })); }
    ids.forEach((id, i) => { const t = s.track(id); if (!t) return; const sc = (STM.state.scores || {})[id];
      strip.append(el("i", { class: id === STM.state.subject ? "subj" : sc != null ? STM.desk.tier(sc) : "", title: `${t.id} · Camera ${t.cam} · ${s.clock(t.t0)}–${s.clock(t.t1)}`,
        style: { left: t.t0 / s.duration * 100 + "%", width: (t.t1 - t.t0) / s.duration * 100 + "%", animationDelay: i * 50 + "ms", top: (6 + (s.cameras.findIndex(c => c.id === t.cam) % 2) * 3) + "px" } })); });
    return el("div", {}, strip, el("div", { class: "tstrip-ax" }, el("span", {}, s.clock(0)), el("span", {}, s.clock(s.duration / 2)), el("span", {}, s.clock(s.duration))));
  }
  function narrowChips(s, ids, jump) {
    const B = STM.desk.buckets(ids), row = el("div", { class: "nchips" });
    const best = B.times.reduce((a, b, i) => (b.n >= (B.times[a] ? B.times[a].n : 0) ? i : a), -1);
    B.times.forEach((b, i) => {
      const args = { after: s.clock(b.a), before: s.clock(b.b) }; if (jump) args.camera = "";
      row.append(el("button", { class: "chip", type: "button", id: `chat-narrow-${i}`, "data-best": i === best ? "1" : null, onclick: () => tap("narrow_search", args, `Only ${b.label}`) }, b.label, el("b", {}, String(b.n))));
    });
    if (B.cams.length > 1 || jump) B.cams.forEach(c => {
      const args = { camera: c.cam }; if (jump) Object.assign(args, { after: "", before: "" });
      row.append(el("button", { class: "chip", type: "button", id: `chat-narrow-cam-${c.cam}`, onclick: () => tap("narrow_search", args, `Just camera ${c.cam}`) }, el("span", { html: icons.cam }), `Camera ${c.cam}`, el("b", {}, String(c.n))));
    });
    if (STM.desk.hasWin(STM.state.window)) row.append(el("button", { class: "chip", type: "button", id: "chat-narrow-clear", onclick: () => tap("narrow_search", { after: "", before: "", camera: "" }, "Any time, any camera") }, "Clear window"));
    return row;
  }

  /* what a search or narrow found, honestly: relaxed colours, assumed words, unreadable times, nothing in the window */
  function report(m, r, s, how) {
    const D = STM.desk, w = r.window, lk = lookText(r.look);
    if (w.bad) say(m, `I couldn't use ${esc(w.bad.join(", "))}. This footage runs <b>${s.span()}</b> on camera${s.cameras.length > 1 ? "s" : ""} ${s.cameras.map(c => c.id).join(", ")}.`);
    if (w.outside) say(m, `That time is outside the footage, which runs <b>${s.span()}</b>. Here is when they were seen instead:`);
    if (!r.look.top && !r.look.bottom) { say(m, "I need at least one clothing colour to search, for example “black top” or “blue jeans”. The index has clothing colours only, no faces."); setSugg(); return; }
    if (!r.results.length) {
      if (r.outside.length && D.hasWin(w)) {
        if (!w.outside) say(m, `Nobody matching ${lk} ${esc(D.winPhrase(w))}. ${r.outside.length} ${r.outside.length === 1 ? "match was" : "matches were"} seen at other times or cameras:`);
        const ids = r.outside.slice(0, 6).map(x => x.t.id); m.append(timeStrip(s, ids, w), narrowChips(s, ids, true));
      } else say(m, `Nobody on ${s.cameras.length} camera${s.cameras.length > 1 ? "s" : ""} matches ${lk}. Try one colour you're sure of. Colours are read from the video pixels, so lighting can shift them.`);
      scroll(); return;
    }
    const n = r.results.length, strong = r.results.filter(x => D.tier(x.score) === "strong").length;
    const tiers = "";
    let txt = how === "narrow"
      ? (D.hasWin(w) ? `Narrowed to <b>${esc(D.winPhrase(w))}</b>: ` : "Whole recording again: ") + `<b>${n}</b> ${n === 1 ? "match" : "matches"} for ${lk}${tiers}.`
      : `Found <b>${n}</b> ${n === 1 ? "person" : "people"} matching ${lk}${D.hasWin(w) ? ` <b>${esc(D.winPhrase(w))}</b>` : ""}${tiers}.`;
    if (r.relaxed) txt += `<br><span class="muted" style="font-size:12px">${esc(r.relaxed)}.</span>`;
    if (r.look.assumed && r.look.assumed.length) txt += `<br><span class="muted" style="font-size:12px">Assumed: ${esc(r.look.assumed.join(", "))}.</span>`;
    if (r.look.ignored && r.look.ignored.length) txt += `<br><span class="muted" style="font-size:12px">Not used: ${esc(r.look.ignored.join(", "))}. The index has clothing colours only.</span>`;
    say(m, txt);
    m.append(cards(s, STM.state.candidates, true), timeStrip(s, STM.state.candidates, w));
    const B = D.buckets(STM.state.candidates);
    if (n >= 3 && (B.times.length > 1 || B.cams.length > 1)) { say(m, strong >= 2 ? "Several look alike and were seen at different times. When or where did you last see them?" : "When or where did you last see them?"); m.append(narrowChips(s, STM.state.candidates)); }
    else say(m, n === 1 ? "Is this them?" : "Is one of them who you're looking for?");
    scroll();
  }
  function seenList(s, ids) {
    const list = el("div", { class: "seen" });
    ids.forEach(id => { const t = s.track(id);
      const btn = el("button", { class: "btn btn-sm", type: "button" }, "Play");
      const row = el("div", { class: "r" }, el("img", { src: s.cropUrl(t), alt: "" }),
        el("div", { style: { minWidth: 0 } }, el("b", {}, `Camera ${t.cam} · ${s.cam(t.cam).name}`), el("div", { class: "mono" }, `${s.clock(t.t0)} → ${s.clock(t.t1)} · ${Math.round(t.t1 - t.t0)} s`)), btn);
      btn.onclick = () => {
        const mine = row.nextElementSibling && row.nextElementSibling.classList.contains("clip") ? row.nextElementSibling : null;
        list.querySelectorAll(".clip").forEach(x => { x.querySelector("video").pause(); x.remove(); });
        list.querySelectorAll(".r .btn").forEach(x => { x.textContent = "Play"; });
        if (!mine) { row.after(clip(s, t)); btn.textContent = "Close"; scroll(); }
      };
      list.append(row); });
    return list;
  }

  /* ---------- tools ---------- */
  async function run(tool, args, m) {
    const s = STM.scene(), D = STM.desk;
    if (!m) m = bot();
    if (m.firstChild && m.firstChild.querySelector && m.firstChild.querySelector(".dots")) m.firstChild.remove();
    if (s.name.startsWith("library-") && s.index_status !== "ready" && !["cannot_do", "start_over", "add_video"].includes(tool)) {
      say(m, "This recording is ready to play. Its search index is still being prepared; matches and counts will appear when indexing finishes.");
      if (STM.indexLibraryScene) STM.indexLibraryScene(s.name);
      return;
    }
    if (STM.current !== "desk" && !["ghost_trails"].includes(tool)) await STM.go("desk");
    switch (tool) {
      case "search_people": {
        const desc = args.description || STM.state.query;
        const win = { after: args.after, before: args.before, camera: args.camera };
        await steps(m, [["Reading the clothing words", 260], [`Scanning ${s.cameras.length} camera${s.cameras.length > 1 ? "s" : ""} · ${s.tracks.length} tracks`, 520], ["Ranking by colour match", 300]]);
        report(m, D.search(desc, !!args.refine, win), s, "search");
        break;
      }
      case "photo_search": {
        await steps(m, [["Reading colours from your photo", 380], [`Scanning ${s.cameras.length} camera${s.cameras.length > 1 ? "s" : ""} · ${s.tracks.length} tracks`, 480]]);
        say(m, `From the photo: ${lookText(args.look)}. Colours only; the photo never leaves this browser.`);
        report(m, D.search({ ...args.look, photo: true }, false, {}), s, "search");
        break;
      }
      case "narrow_search": {
        if (!STM.state.look) { say(m, "Describe who you're looking for first, then I can narrow it by time or camera."); break; }
        await steps(m, [["Applying the time window", 240], ["Re-ranking matches", 260]]);
        report(m, D.narrow(args), s, "narrow");
        break;
      }
      case "when_seen": {
        const ids = STM.state.subject ? D.route(STM.state.subject).map(l => l.t.id) : STM.state.candidates.slice(0, 4);
        if (!ids.length) { say(m, "Describe who you're looking for first, then I can tell you when they were seen."); break; }
        await steps(m, [["Reading sightings from the index", 280]]);
        say(m, STM.state.subject ? "Every sighting of them, in order:" : `When the top ${ids.length} matches were seen:`);
        m.append(seenList(s, [...ids].sort((a, b) => s.track(a).t0 - s.track(b).t0)), timeStrip(s, ids, STM.state.window));
        if (!STM.state.subject && ids.length > 1) m.append(narrowChips(s, STM.state.candidates));
        break;
      }
      case "count_people": {
        await steps(m, [["Counting people tracks in the index", 320]]);
        const c = D.count(args), w = c.w;
        if (w.bad) say(m, `I couldn't use ${esc(w.bad.join(", "))}. This footage runs <b>${s.span()}</b>.`);
        if (w.outside) { say(m, `That time is outside the footage, which runs <b>${s.span()}</b>.`); break; }
        const where = D.hasWin(w) ? `<b>${esc(D.winPhrase(w))}</b>` : `across the whole recording (${s.span()})`;
        say(m, `<b>${c.tracks.length}</b> people tracks ${where}${c.people !== c.tracks.length ? `, about <b>${c.people}</b> people once camera hops are joined` : ""}.`);
        const max = Math.max(1, ...c.perCam.map(x => x.n));
        const bars = el("div", { class: "cbars" }, ...c.perCam.filter(x => !w.cam || x.cam === w.cam).map(x => { const p = el("div", { class: "prog" }, el("i"));
          requestAnimationFrame(() => setTimeout(() => p.firstChild.style.width = x.n / max * 100 + "%", 80)); return el("div", { class: "r" }, el("span", {}, `Camera ${x.cam}`), p, el("b", {}, String(x.n))); }));
        m.append(bars);
        if (c.busiest && c.busiest.n) { const b = say(m, `Busiest moment: <b>${s.clock(c.busiest.t)}</b> with ${c.busiest.n} in view. `);
          b.append(el("button", { class: "btn btn-sm", type: "button", onclick: () => D.seek(Math.max(0, c.busiest.t - 2)) }, "Play it")); }
        say(m, '<span class="muted" style="font-size:12px">A track can split when someone is hidden, so counts are approximate.</span>');
        break;
      }
      case "pick_candidate": {
        const t = D.pick(args.number || 1);
        if (!t) { say(m, STM.state.candidates.length ? `There are only ${STM.state.candidates.length} matches on screen.` : "There's no match on screen yet. Describe them first."); break; }
        await steps(m, [["Locking on to this person", 240], ["Retracing across cameras", 520]]);
        const legs = D.route(t.id), last = D.lastSighting(t.id);
        say(m, `Confirmed: <b>${esc(t.desc)}</b>. Seen on ${new Set(legs.map(l => l.t.cam)).size} camera${new Set(legs.map(l => l.t.cam)).size > 1 ? "s" : ""}, ${legs.length} sighting${legs.length > 1 ? "s" : ""}. Last seen on <b>Camera ${last.track.cam}</b> at <b>${s.clock(last.t)}</b>.`);
        m.append(seenList(s, legs.map(l => l.t.id)));
        setSugg(["Who came near after?", "When were they seen?", STM.showcase ? "Send the robot to check" : "Show the moment they left"]);
        break;
      }
      case "reject_candidate": {
        const id = D.reject(args.number || 1);
        say(m, id ? `Removed that one. ${STM.state.candidates.length} left.` : "Nothing to remove.");
        if (STM.state.candidates.length) m.append(cards(s, STM.state.candidates, true));
        break;
      }
      case "who_came_near": {
        if (!STM.state.subject) { say(m, "Confirm who you're looking for first, then I can check who came near."); break; }
        STM.state.followed = true;
        await steps(m, [["Finding where they were last seen", 260], ["Checking that camera 15 s either side", 480]]);
        const n = D.near(STM.state.subject);
        if (!n.length) say(m, "Nobody else was on that camera around then.");
        else { say(m, `${n.length} ${n.length === 1 ? "person was" : "people were"} on the same camera around then. A lead to check, not proof of anything.`); m.append(cards(s, n.map(x => x.t.id), false)); }
        STM.emit("desk", { kind: "near" });
        break;
      }
      case "follow_lead": case "where_did_i_go": {
        const id = STM.state.subject || STM.state.candidates[0];
        if (!id) { say(m, "Describe who you're looking for first."); break; }
        if (!STM.state.subject) D.pick(1);
        await steps(m, [["Following camera to camera", 520]]);
        const legs = D.route(id);
        say(m, `Their route, ${legs.length} sighting${legs.length > 1 ? "s" : ""}:`);
        m.append(seenList(s, legs.map(l => l.t.id)));
        break;
      }
      case "show_moment": {
        const id = STM.state.subject || STM.state.candidates[0];
        const last = id && D.lastSighting(id);
        if (!last) { say(m, "Nothing to show yet. Describe who you're looking for first."); break; }
        D.seek(Math.max(0, last.t - 4), last.track.cam);
        say(m, `Camera ${last.track.cam}, the last few seconds before ${s.clock(last.t)}:`);
        m.append(clip(s, last.track, Math.max(0, last.t - 6), last.t + 1)); scroll();
        break;
      }
      case "ghost_trails": { say(m, "Opening the timeline with everyone who passed."); STM.go("timeline"); break; }
      case "start_over": { D.reset(); say(m, "Cleared. Who are you looking for?"); setSugg(); break; }
      case "send_robot": {
        if (!STM.showcase) { say(m, "No robot is connected to this desk."); break; }
        if (!STM.state.subject) { say(m, "Confirm who you're looking for first, so Rover 1 knows where to go."); break; }
        say(m, "Opening Rover 1. It will drive to the last sighting and scan. (simulated)");
        await STM.go("robot", { autoSend: true });
        setTimeout(() => { const b = STM.$("#robot-send"); if (b && !b.disabled) b.click(); }, 1400);
        break;
      }
      default: say(m, esc(args.reason || "The desk can't do that. It finds and follows people by what they wore, never by face or name."));
    }
  }

  /* replies that need no model: answered instantly on the page */
  function quick(text) {
    const t = text.toLowerCase().replace(/[.!?]+$/, "").trim(); let m;
    if (/^(start over|reset|clear|new search)$/.test(t)) return { tool: "start_over", args: {}, model: "instant", ms: 0 };
    if (STM.state.candidates.length && (m = t.match(/^(?:number|no\.?|#)\s*(\d)$/))) return { tool: "pick_candidate", args: { number: +m[1] }, model: "instant", ms: 0 };
    if (STM.state.candidates.length && /^(yes|yep|yeah|that'?s (them|him|her|me))$/.test(t)) return { tool: "pick_candidate", args: { number: 1 }, model: "instant", ms: 0 };
    return null;
  }
  /* several people in one request: split into one look per person, client-side */
  function splitLooks(text) {
    let t = " " + text.replace(/^.*?:\s*/, m => /\b(top|shirt|jeans|trousers|in)\b/i.test(m) ? m : "") + " ";
    const parts = t.split(/\s*;\s*|,?\s+and\s+(?:the other|another|someone|somebody|a person|one)\b(?:\s+(?:is|was|in|wearing))*|\s+plus\s+|,\s*(?:the other|another)\b(?:\s+(?:is|was|in|wearing))*/i)
      .map(x => (x || "").replace(/^\s*(?:one|the first)\s+(?:is\s+|was\s+)?(?:in\s+|wearing\s+)?/i, "").trim()).filter(Boolean);
    const looks = parts.filter(p => { const l = STM.desk.parseLook(p); return !!(l.top || l.bottom); });
    return looks.length >= 2 && looks.length === parts.filter(p => /\w/.test(p)).length ? looks.slice(0, 3) : null;
  }
  async function multi(looks, m) {
    const s = STM.scene(), D = STM.desk;
    if (STM.current !== "desk") await STM.go("desk");
    await steps(m, [[`Splitting into ${looks.length} people`, 240], [`Scanning ${s.cameras.length} camera${s.cameras.length > 1 ? "s" : ""} · ${s.tracks.length} tracks`, 520]]);
    const res = [];
    for (let i = looks.length - 1; i >= 0; i--) { const r = D.search(looks[i], false, {}); res[i] = { r, ids: STM.state.candidates.slice() }; }   // person 1 last, so picks apply to them
    say(m, `Looking for <b>${looks.length}</b> people separately.`);
    res.forEach(({ r, ids }, i) => {
      const lk = lookText(r.look);
      m.append(el("div", { class: "chat-person", style: { fontWeight: "600", margin: "10px 0 4px" } }, `Person ${i + 1}: ${lk.replace(/<[^>]+>/g, "")}`));
      if (!ids.length) { say(m, `Nobody matches ${lk}.`); return; }
      say(m, `<b>${ids.length}</b> ${ids.length === 1 ? "match" : "matches"}${i ? "" : ". Tap a card to follow this person"}.`);
      m.append(cards(s, ids, i === 0), timeStrip(s, ids, r.window));
    });
    scroll();
  }
  async function ask(text, via = "typed", viaNote) {
    text = (text || "").trim(); if (!text || busy) return;
    busy = true; $("#q").value = ""; autosize();
    me(text, viaNote || (via === "voice" ? "spoken" : null));
    const m = bot();
    try {
      let r = quick(text);
      const looks = !r && splitLooks(text);
      if (looks) {
        m.firstChild.remove();
        looks.forEach(d => m.append(actionChip("search_people", { description: d }, "split on this page")));
        await multi(looks, m);
        busy = false; if (!STM.state.subject) setSugg(); scroll(); return;
      }
      if (!r && /\b(robot|rover)\b/i.test(text) && STM.showcase) r = { tool: "send_robot", args: {}, model: "desk shortcut", ms: 0 };
      if (!r) r = await route(text);
      m.firstChild.remove();
      m.append(actionChip(r.tool, r.args || {}, `${r.model} · ${r.ms} ms`));
      await run(r.tool, r.args || {}, m);
    } catch (e) { console.error(e); say(m, "Something went wrong running that. Try again, or start over."); }
    busy = false; if (!STM.state.subject) setSugg(); scroll();
  }
  async function tap(tool, args, label, meta = "desk action") {     // chips and buttons run a tool directly, no model call
    if (busy) return; busy = true;
    if (label) me(label, "tap");
    const m = bot(); await sleep(220); m.firstChild.remove();
    m.append(actionChip(tool, args, meta));
    try { await run(tool, args, m); } catch (e) { console.error(e); say(m, "Something went wrong running that."); }
    busy = false; if (!STM.state.subject) setSugg(); scroll();
  }

  function setSugg(list) {
    const st = STM.state, s = STM.scene(), D = STM.desk;
    if (!list && st.candidates.length && !st.subject) {
      const B = D.buckets(st.candidates), late = B.times[B.times.length - 1];
      list = [late && B.times.length > 1 ? `Only after ${s.clock(late.a).slice(0, 5)}` : null, B.cams.length > 1 ? `Just camera ${B.cams[B.cams.length - 1].cam}` : null, "When were they seen?", "That's them, number 1"].filter(Boolean);
    }
    list = list || (st.subject ? ["Who came near after?", "When were they seen?", STM.showcase ? "Send the robot to check" : "Show the moment they left"] :
      STM.sceneName === "meva" ? ["Black top and jeans", "Two friends: black top and jeans, and someone in a blue top", "Anyone in blue after 16:52?", "How busy was camera B?"].concat(STM.showcase ? ["Who is that man?"] : []) :
      STM.sceneName === "lab" ? ["Someone in a red top and brown trousers", "A person in a yellow top", "Who is that man?"] :
      ["Someone in a red top", "How many people were there?", "Grey top and grey trousers"]);
    const h = $("#chatSugg"); h.innerHTML = "";
    list.forEach((x, i) => h.append(el("button", { class: "chip", type: "button", id: "chat-sugg-" + i, onclick: () => ask(x, "tap") }, x)));
  }
  const autosize = () => { const q = $("#q"); q.style.height = "auto"; q.style.height = Math.min(120, q.scrollHeight) + "px"; };

  /* ---------- photo reference: upper and lower body colours, read here ---------- */
  function nameRGB(r, g, b) {
    const mx = Math.max(r, g, b) / 255, mn = Math.min(r, g, b) / 255, sat = mx ? (mx - mn) / mx : 0;
    if (mx < 0.2) return "black";
    if (sat < 0.2) return mx > 0.78 ? "white" : mx < 0.38 ? "black" : "grey";
    const d = (mx - mn) * 255, R = r, G = g, B = b, M = Math.max(R, G, B);
    let h = M === R ? ((G - B) / d) % 6 : M === G ? (B - R) / d + 2 : (R - G) / d + 4; h = (h * 60 + 360) % 360;
    if (h < 15 || h >= 345) return mx < 0.45 ? "brown" : "red";
    if (h < 40) return mx < 0.6 ? "brown" : "orange";
    if (h < 68) return "yellow"; if (h < 165) return "green"; if (h < 255) return "blue"; if (h < 290) return "purple"; return "pink";
  }
  async function readPhoto(file) {
    const url = URL.createObjectURL(file), img = new Image();
    await new Promise((ok, bad) => { img.onload = ok; img.onerror = bad; img.src = url; });
    const W = 60, H = 120, c = el("canvas"); c.width = W; c.height = H;
    const g = c.getContext("2d", { willReadFrequently: true }), k = Math.max(W / img.width, H / img.height);
    g.drawImage(img, (W - img.width * k) / 2, (H - img.height * k) / 2, img.width * k, img.height * k);
    const vote = (y0, y1) => { const px = g.getImageData(W * .25, H * y0, W * .5, H * (y1 - y0)).data, n = {};
      for (let i = 0; i < px.length; i += 4) { const w = nameRGB(px[i], px[i + 1], px[i + 2]); n[w] = (n[w] || 0) + 1; }
      return Object.entries(n).sort((a, b) => b[1] - a[1])[0][0]; };
    return { url, look: { top: vote(.2, .48), bottom: vote(.56, .86) } };
  }
  async function photo(file) {
    if (busy || !file) return;
    let p; try { p = await readPhoto(file); } catch (e) { STM.toast("That file isn't an image the browser can read."); return; }
    busy = true; me("Find someone dressed like this", "photo", p.url);
    const m = bot(); await sleep(200); m.firstChild.remove();
    m.append(actionChip("photo_search", { look: p.look }, "read in this browser"));
    try { await run("photo_search", { look: p.look }, m); } catch (e) { console.error(e); say(m, "Couldn't search with that photo."); }
    busy = false; if (!STM.state.subject) setSugg(); scroll();
  }

  /* ---------- new video: upload to the desk server, which indexes it with the same pipeline ---------- */
  async function upload(file) {
    if (busy) return; busy = true;
    const name = file.name.replace(/\.[^.]+$/, "").slice(0, 40), mb = (file.size / 1048576).toFixed(1);
    me(`Add “${name}” to the desk`, `${mb} MB video`);
    const m = bot(); await sleep(200); m.firstChild.remove();
    m.append(actionChip("add_video", { file: `${mb} MB` }, "indexed on this laptop"));
    const box = el("div", { class: "steps" }); m.append(box);
    const row = label => { const p = el("div", { class: "prog", style: { width: "70px", marginLeft: "auto" } }, el("i")), r = el("div", {}, el("span", { class: "spin", style: { width: "14px", height: "14px" } }), el("span", {}, label), p); box.append(r); scroll(); return { r, p, done(note) { r.firstChild.replaceWith(el("span", { class: "ck", html: icons.check })); p.replaceWith(el("span", { class: "muted mono", style: { marginLeft: "auto", fontSize: "11px" } }, note)); } }; };
    const fail = msg => { box.lastChild && box.lastChild.firstChild.replaceWith(el("span", { style: { color: "var(--bad)" } }, "✕")); say(m, msg); busy = false; scroll(); };
    const up = row(`Uploading ${mb} MB`); let idx = null, t0 = performance.now(), timer;
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/find/scene?name=${encodeURIComponent(name)}`);
    xhr.upload.onprogress = e => { if (e.lengthComputable) up.p.firstChild.style.width = (e.loaded / e.total * 100) + "%"; };
    xhr.upload.onload = () => { up.done(`${((performance.now() - t0) / 1000).toFixed(1)} s`); t0 = performance.now();
      idx = row("Indexing: decode → detect → track → colours"); idx.p.classList.add("skel");
      timer = setInterval(() => { idx.r.children[1].textContent = `Indexing: decode → detect → track → colours · ${((performance.now() - t0) / 1000).toFixed(0)} s`; }, 500); };
    xhr.onerror = () => { clearInterval(timer); fail("Couldn't reach the desk server. Adding a video needs it running on this machine (serve.cmd)."); };
    xhr.onload = async () => {
      clearInterval(timer);
      let out = {}; try { out = JSON.parse(xhr.responseText); } catch (e) {}
      if (xhr.status !== 200) { const d = out.detail || {}; return fail(`Indexing failed: ${esc(d.error || ("server said " + xhr.status))}.`); }
      if (idx) idx.done(`${out.seconds} s`);
      const ld = row("Loading the new scene");
      try {
        const s = await STM.loadScene(out.scene, out.base);
        ld.done(`${s.tracks.length} tracks`);
        STM.sceneList.push([out.scene, name.slice(0, 22), "uploaded"]);
        STM.sceneName = out.scene; STM.desk.reset(); await STM.go("desk");
        say(m, `Indexed <b>${esc(name)}</b> in <b>${out.seconds} s</b>: ${s.tracks.length} people tracks over ${Math.round(s.duration)} s. Times are video time unless the camera clock is known. Who are you looking for?`);
        setSugg(s.tracks.slice(0, 1).map(t => `Someone in a ${t.look.top} top`).concat(["How many people were there?"]));
      } catch (e) { fail("The scene was indexed but couldn't be loaded."); return; }
      busy = false; scroll();
    };
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.send(file);
  }

  /* ---------- voice (Canary push-to-talk) ---------- */
  const MAX_REC_S = 19;
  function wavFrom(chunks, rate) {
    const n = chunks.reduce((a, c) => a + c.length, 0), all = new Float32Array(n);
    let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
    const ratio = rate / 16000, len = Math.floor(n / ratio), pcm = new Int16Array(len);
    for (let i = 0; i < len; i++) {
      const a = Math.floor(i * ratio), b = Math.min(n, Math.floor((i + 1) * ratio)); let s = 0;
      for (let j = a; j < b; j++) s += all[j];
      const v = Math.max(-1, Math.min(1, s / Math.max(1, b - a))); pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
    }
    const buf = new ArrayBuffer(44 + pcm.length * 2), d = new DataView(buf), w = (p, s) => [...s].forEach((ch, i) => d.setUint8(p + i, ch.charCodeAt(0)));
    w(0, "RIFF"); d.setUint32(4, 36 + pcm.length * 2, true); w(8, "WAVE"); w(12, "fmt "); d.setUint32(16, 16, true);
    d.setUint16(20, 1, true); d.setUint16(22, 1, true); d.setUint32(24, 16000, true); d.setUint32(28, 32000, true);
    d.setUint16(32, 2, true); d.setUint16(34, 16, true); w(36, "data"); d.setUint32(40, pcm.length * 2, true);
    new Int16Array(buf, 44).set(pcm);
    return { blob: new Blob([buf], { type: "audio/wav" }), seconds: len / 16000 };
  }
  function setupMic() {
    const btn = $("#mic"), heard = $("#heard");
    const wave = '<span class="wave"><i></i><i></i><i></i><i></i><i></i></span>';
    let rec = null, downAt = 0;
    const start = async () => {
      if (rec || busy) return;
      if (!STM.live.speech || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { heard.textContent = "Speech is offline right now. Type instead."; return; }
      const mine = rec = { chunks: [], t0: performance.now() };
      let stream;
      try { stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } }); }
      catch (e) { if (rec === mine) rec = null; heard.textContent = "Microphone blocked. Allow it for this page."; return; }
      if (rec !== mine) { stream.getTracks().forEach(t => t.stop()); return; }
      rec.stream = stream; rec.t0 = performance.now(); rec.ctx = new AudioContext();
      const src = rec.ctx.createMediaStreamSource(stream), node = rec.ctx.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = e => { if (rec) rec.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
      src.connect(node); node.connect(rec.ctx.destination); rec.node = node; btn.classList.add("on");
      rec.timer = setInterval(() => { const s = (performance.now() - rec.t0) / 1000; heard.innerHTML = `${wave} Listening ${s.toFixed(1)} s · release to send`; if (s >= MAX_REC_S) stop(); }, 100);
    };
    const stop = async () => {
      if (!rec || !rec.ctx) { rec = null; return; }
      const r = rec; rec = null; clearInterval(r.timer); btn.classList.remove("on");
      r.node.disconnect(); r.stream.getTracks().forEach(t => t.stop());
      const rate = r.ctx.sampleRate; r.ctx.close();
      const { blob, seconds } = wavFrom(r.chunks, rate);
      if (seconds < 0.4) { heard.textContent = "Too short. Hold the mic while you talk."; return; }
      heard.innerHTML = `<span class="spin" style="width:12px;height:12px"></span> Transcribing ${seconds.toFixed(1)} s with Canary…`;
      try {
        const res = await fetch("/api/find/transcribe", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: blob });
        const out = await res.json();
        if (!res.ok) throw new Error((out.detail && out.detail.error) || res.status);
        if (!out.text) { heard.textContent = "Canary heard nothing. Try again or type."; return; }
        heard.textContent = `Heard by ${out.provider} in ${out.ms} ms`;
        ask(out.text, "voice", `spoken · ${out.provider} ${out.ms} ms`);
      } catch (e) { heard.textContent = `Speech failed (${e.message}). Type instead.`; }
    };
    btn.addEventListener("pointerdown", e => { e.preventDefault(); downAt = performance.now(); if (rec) stop(); else start(); });
    btn.addEventListener("pointerup", () => { if (rec && performance.now() - downAt > 350) stop(); });
    btn.addEventListener("pointerleave", () => { if (rec && performance.now() - downAt > 350) stop(); });
    btn.addEventListener("keydown", e => { if ((e.key === "Enter" || e.key === " ") && !e.repeat) { e.preventDefault(); e.stopPropagation(); rec ? stop() : start(); } });
  }

  STM.chat = {
    init() {
      log = $("#chatLog");
      const s = STM.scene();
      const m = el("div", { class: "msg bot" });
      log.append(m);
      say(m, `Hi. I search <b>${s.cameras.length} cameras</b> (${s.span()}) for someone by what they wore, then follow them camera to camera. I never use faces or names.`);
      say(m, "Who are you looking for? You can add a time, a camera or a photo.");
      setSugg();
      const q = $("#q");
      q.addEventListener("input", autosize);
      q.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(q.value); } });
      $("#send").onclick = () => ask(q.value);
      const af = $("#attachFile");
      $("#attach").onclick = () => af.click();
      af.onchange = () => { const f = af.files[0]; af.value = ""; photo(f); };
      setupMic();
      $("#chatNew").onclick = () => tap("start_over", {}, "New search", "instant");
      status(STM.live.route !== false); STM.checkLive().then(l => status(!!l.route)).catch(() => {});
      $("#heard").textContent = STM.showcase ? "The agent only picks a tool. Every time, camera and match comes from the index." : "Clothing colours only. Never faces or names.";
      STM.on("desk", d => { if (d && d.kind === "reset") setSugg(); });
    },
    ask, photo, upload, tap,
    run: (tool, args, label) => tap(tool, args, label, "desk button"),
    note(text) { const m = el("div", { class: "msg bot" }); log.append(m); say(m, esc(text)); setSugg(); scroll(); },
    type: async (text, cps = 28) => { const q = $("#q"); q.value = ""; q.focus(); for (const ch of text) { q.value += ch; autosize(); await sleep(1000 / cps); } },
    get busy() { return busy; },
  };
})();
