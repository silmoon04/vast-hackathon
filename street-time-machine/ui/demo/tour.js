/* Guided demo for recording: a virtual cursor clicks through the real app while the camera rig zooms and pans to
   what it touches (the screen-recording look of the reference video). Start: "Play guided demo" or open #tour. */
(function () {
  "use strict";
  const { $, $$, sleep } = STM;
  const rig = () => $("#rig");
  let cam = { s: 1, tx: 0, ty: 0 }, stopped = true, cur = { x: innerWidth / 2, y: innerHeight / 2 };
  class Stop extends Error {}
  const chk = () => { if (stopped) throw new Stop(); };
  const wait = async ms => { const end = performance.now() + ms; while (performance.now() < end) { chk(); await sleep(Math.min(120, end - performance.now())); } };
  const q = sel => typeof sel === "string" ? $(sel) : sel;
  const lastOf = sel => { const all = $$(sel); return all[all.length - 1] || null; };

  function setCam(s, tx, ty) { cam = { s, tx, ty }; rig().style.transform = s === 1 && !tx && !ty ? "" : `translate(${tx}px,${ty}px) scale(${s})`; }
  async function focus(sel, s = 1.6, settle = 1150) {
    const n = q(sel); if (!n) return; chk();
    n.scrollIntoView({ block: "nearest", behavior: "instant" });
    await sleep(60);
    const r = n.getBoundingClientRect(), W = innerWidth, H = innerHeight;
    const w = r.width / cam.s, h = r.height / cam.s;
    s = Math.max(1, Math.min(s, 0.92 * W / w, 0.88 * H / h));
    const cx = (r.left + r.width / 2 - cam.tx) / cam.s, cy = (r.top + r.height / 2 - cam.ty) / cam.s;
    let tx = W / 2 - cx * s, ty = H / 2 - cy * s;
    tx = Math.min(0, Math.max(W - W * s, tx)); ty = Math.min(0, Math.max(H - H * s, ty));
    setCam(s, tx, ty); await wait(settle);
  }
  const wide = async (ms = 1100) => { setCam(1, 0, 0); await wait(ms); };

  const cursor = () => $("#cursor");
  async function moveTo(sel, ms = 900) {
    const n = q(sel); if (!n) return null; chk();
    const r = n.getBoundingClientRect();
    cur = { x: r.left + Math.min(r.width / 2, 60), y: r.top + r.height / 2 };
    const c = cursor(); c.style.transitionDuration = ms + "ms"; c.style.transform = `translate(${cur.x}px,${cur.y}px)`;
    await wait(ms + 80); return n;
  }
  async function click(sel, opts = {}) {
    const n = await moveTo(sel, opts.ms); if (!n) return false;
    const c = cursor(); c.classList.remove("click"); void c.offsetWidth; c.classList.add("click");
    await wait(160); n.click(); await wait(opts.after ?? 350); return true;
  }
  function caption(title, sub) {
    const c = $("#caption");
    if (!title) { c.classList.remove("show"); return; }
    c.innerHTML = `${STM.esc(title)}${sub ? `<small>${STM.esc(sub)}</small>` : ""}`; c.classList.add("show");
  }
  async function untilIdle(max = 9000) { const end = performance.now() + max; await wait(250); while (STM.chat.busy && performance.now() < end) await wait(150); await wait(500); }
  const nav = async view => { await click(`#nav a[data-view="${view}"]`, { after: 900 }); };
  const near = sel => { const n = q(sel); return n ? (n.closest(".card") || n) : null; };

  async function script() {
    caption("Find & Follow", "Find someone by what they wore, then follow them camera to camera");
    if (STM.current !== "desk") await STM.go("desk");
    STM.desk.reset(); $("#chat").classList.remove("closed");
    await wait(900);
    await focus("#desk-player", 1.35); await wait(1600);
    await wide(900);

    caption("Ask the desk agent", "Typed here; hold the mic to speak (NVIDIA Canary)");
    await focus("#chat", 1.55);
    await click("#q", { after: 200 });
    await STM.chat.type("I lost two colleagues: black top and jeans, and someone in a blue top", 26);
    await click("#send", { after: 200 });
    caption("Two people, two searches", "Each description runs against the index on its own");
    await untilIdle();
    await wait(1200);

    caption("Ranked matches from every camera", "Real campus footage. Clothing colours only, no faces, no names.");
    await wide(700);
    await focus(near("#desk-tab-matches"), 1.45); await wait(1800);

    caption("Narrow by when you last saw them", "The desk hands back when each match was seen");
    await focus("#chat", 1.55);
    const bucket = lastOf(".nchips .chip[data-best]") || lastOf(".nchips .chip");
    if (bucket) { await click(bucket, { after: 300 }); await untilIdle(); await wait(900); }
    await wide(600);
    await focus("#desk-player", 1.3); caption("The window shows on the scrubber"); await wait(1800);
    await wide(600);

    caption("Confirm it's them");
    await focus("#chat", 1.55);
    await click(lastOf("[id='chat-cand-1']") || "#desk-confirm-1", { after: 300 });
    await untilIdle(); await wait(900);

    caption("Their route, camera by camera", "Played from the last sighting");
    await wide(700);
    await focus("#desk-player", 1.3); await wait(2200);
    await focus(near("#desk-tab-route"), 1.4); await wait(2000);
    await focus("#desk-steps", 1.6); caption("Search steps unlock as the desk confirms each one"); await wait(1800);

    caption("Who came near after they left?", "A lead to check, not a verdict");
    await focus("#chat", 1.5);
    await click("#chat-sugg-0", { after: 300 });
    await untilIdle(); await wait(1600);
    await wide(800);

    for (const [view, steps] of [
      ["timeline", async () => {
        caption("Timeline", "Every camera, every sighting, on one clock");
        await focus(near("#timeline-play"), 1.25);
        await click("#timeline-play"); await wait(3200);
      }],
      ["catalogue", async () => {
        caption("Catalogue", "Everything the index has seen, filterable");
        await wait(1200); await focus(".view-title", 1.0, 400);
        await click("#catalogue-first-card", { after: 1400 });
        if ($("#catalogue-similar")) { caption("Find similar people"); await click("#catalogue-similar", { after: 1200 }); }
      }],
      ["similar", async () => {
        caption("Find similar", "A colour signature computed in your browser from each crop");
        await click("#similar-run", { after: 400 });
        await focus("#similar-map", 1.3); await wait(3200);
      }],
      ["data", async () => {
        caption("Query the index", "Plain English becomes a query over the tracks table");
        await click("#data-preset-0", { after: 400 });
        await focus(near(".code") || "#data-run", 1.4); await wait(2200);
        await focus(near("#data-results") || "#data-results", 1.25); await wait(2200);
      }],
      ["pipeline", async () => {
        caption("The pipeline", "Real camera footage indexed in seconds on a laptop CPU");
        await click("#pipeline-run", { after: 300 });
        await focus(near("#pipeline-graph"), 1.25); await wait(5200);
      }],
      ["robot", async () => {
        caption("Find the robot, send it to the spot", "Simulated robot: Rover 1");
        await click("#robot-find", { after: 2400 });
        await click("#robot-send", { after: 300 });
        await focus(near("#robot-send") ? lastOf("#view svg") || "#robot-send" : "#robot-send", 1.2); await wait(6500);
      }],
    ]) {
      if (!STM.views[view]) continue;
      await wide(600);
      await nav(view);
      if (STM.current === view || view === "similar") await steps();
    }
    await wide(900);
    await nav("desk");
    caption("Street Time Machine · Find & Follow", "Real footage in, answers in seconds");
    await wait(3500);
  }

  STM.tour = {
    init() {
      $("#tourBtn").onclick = () => STM.tour.start();
      $("#tourStop").onclick = () => STM.tour.stop();
      document.addEventListener("keydown", e => { if (e.key === "Escape" && !stopped) STM.tour.stop(); });
      if (STM.startHash === "#tour") setTimeout(() => STM.tour.start(), 600);
    },
    async start() {
      if (!stopped) return;
      stopped = false; document.body.classList.add("touring");
      const c = cursor(); c.style.transitionDuration = "0ms"; c.style.transform = `translate(${innerWidth / 2}px,${innerHeight / 2}px)`; c.classList.add("show");
      try { await script(); } catch (e) { if (!(e instanceof Stop)) console.error(e); }
      STM.tour.stop();
    },
    stop() { stopped = true; document.body.classList.remove("touring"); setCam(1, 0, 0); caption(null); cursor().classList.remove("show"); },
    focus, wide, click, caption,
  };
})();
