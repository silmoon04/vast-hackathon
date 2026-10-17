/* Illustrated Find & Follow assets. Load after core.js and before the views. */
(function () {
  "use strict";
  const STM = window.STM;
  if (!STM || STM.visualAssets) return;

  const assetBase = new URL("assets/illustrated/", document.currentScript.src).href;
  const original = { ...STM.icons };
  const files = {
    desk: "find-follow-v2", timeline: "timeline-v2", catalogue: "catalogue-v2", similar: "similar-v2",
    pipeline: "pipeline-v2", data: "data-v2", robot: "robot-v2", agent: "agent", camera: "camera",
    voice: "voice", privacy: "privacy", route: "route",
  };
  const esc = STM.esc;
  const svg = (name, paths, size = 16) => `<svg class="ff-icon ff-icon--${name}" data-icon="${name}" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${paths}</svg>`;
  const fallback = {
    ...original,
    agent: svg("agent", '<rect x="4" y="5" width="16" height="13" rx="4"/><path d="m8 18-2 3M8 10h.01M16 10h.01M9 14h6"/>'),
    camera: original.cam,
    voice: svg("mic", '<rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10a7 7 0 0 0 14 0M12 17v5M9 22h6"/>'),
    privacy: original.lock,
    route: svg("route", '<circle cx="5" cy="5" r="2"/><circle cx="19" cy="19" r="2"/><path d="M7 5h7a4 4 0 0 1 0 8h-4a4 4 0 0 0 0 8h7"/>'),
  };

  function artMarkup(name, extra = "") {
    if (!files[name]) name = "desk";
    return `<span class="ico ff-art ff-art--${name} ${extra}" data-art="${name}" aria-hidden="true"><img src="${esc(assetBase + files[name] + ".webp")}" alt="" width="128" height="128" decoding="async" draggable="false"><span class="ff-art-fallback">${fallback[name] || original.desk}</span></span>`;
  }

  /* Keep the shared object: several view modules capture its reference. */
  for (const name of Object.keys(STM.icons)) {
    STM.icons[name] = STM.icons[name].replace(/<svg\b/, `<svg class="ff-icon ff-icon--${name}" data-icon="${name}" aria-hidden="true" focusable="false"`).replace(/class="ff-icon([^"\n]*)"([^>]*?) class="ico"/, 'class="ico ff-icon$1"$2');
  }
  STM.icons.clock = svg("clock", '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>', 14);
  for (const name of ["desk", "timeline", "catalogue", "similar", "pipeline", "data", "robot"]) STM.icons[name] = artMarkup(name);

  const buildNav = STM.buildNav;
  STM.buildNav = function (...args) {
    const result = buildNav.apply(this, args);
    document.querySelectorAll("#nav a[data-view]").forEach(link => {
      const name = link.dataset.view;
      if (files[name] && !link.querySelector(":scope > .ff-art")) link.prepend(makeArt(name, ""));
    });
    return result;
  };

  const pageHead = STM.pageHead;
  STM.pageHead = function (root, title, sub, right) {
    const head = pageHead(root, title, sub, right);
    const name = files[STM.current] ? STM.current : "desk";
    head.classList.add("ff-illustrated-title");
    head.insertAdjacentHTML("afterbegin", artMarkup(name, "ff-title-art ff-entrance"));
    return head;
  };

  const observed = new WeakSet();
  const intersection = "IntersectionObserver" in window ? new IntersectionObserver(entries => {
    for (const entry of entries) {
      entry.target.classList.toggle("ff-in-view", entry.isIntersecting);
      if (entry.isIntersecting) entry.target.classList.add("ff-entered");
      if (!entry.target.isConnected) intersection.unobserve(entry.target);
    }
  }, { threshold: 0.08 }) : null;

  function observe(node) {
    if (observed.has(node)) return;
    observed.add(node);
    if (intersection) intersection.observe(node);
    else node.classList.add("ff-in-view", "ff-entered");
  }

  function makeArt(name, extra) {
    const box = document.createElement("span");
    box.innerHTML = artMarkup(name, extra);
    return box.firstElementChild;
  }

  function replaceGraphic(selector, name, className) {
    const host = document.querySelector(selector);
    if (!host || host.dataset.ffArt) return;
    host.dataset.ffArt = name;
    host.classList.add(className);
    host.replaceChildren(makeArt(name, ""));
  }

  const shellRoles = {
    cmdk: "search", bell: "bell", themeBtn: "theme", chatToggle: "chat",
    chatNew: "plus", chatClose: "close", attach: "attach", mic: "mic", send: "arrow",
    tourBtn: "play", "desk-play": "play", "robot-find": "search", "catalogue-drawer-close": "close",
  };
  const signatures = new Map(Object.entries(original).map(([name, markup]) => {
    const tmp = document.createElement("span"); tmp.innerHTML = markup;
    return [tmp.firstElementChild.innerHTML.replace(/\s+/g, ""), name];
  }));

  function markIcon(node) {
    if (node.dataset.icon) { observe(node); return; }
    /* A 24-unit glyph inside a known control is safe. Charts, paths, maps and pipeline SVGs stay intact. */
    if (node.getAttribute("viewBox") !== "0 0 24 24" && !node.closest(".brand")) return;
    const host = node.closest("[id]");
    let name = host && shellRoles[host.id];
    if (!name) name = signatures.get(node.innerHTML.replace(/\s+/g, ""));
    if (!name) {
      const paths = [...node.querySelectorAll("path")].map(p => p.getAttribute("d")).join(" ");
      if (node.closest(".brand")) name = "brand";
      else if (/M18 6L6 18|M6 6l12 12/.test(paths)) name = "close";
      else if (/M20 20l/.test(paths)) name = "search";
      else if (/M12 5v14M5 12/.test(paths)) name = "plus";
      else if (/M8 5v14/.test(paths)) name = "play";
      else if (/M5 12h14M13 6/.test(paths)) name = "arrow";
      else if (/M6 9l6 6/.test(paths)) name = "chev";
      else if (node.querySelector('circle[cx="12"][cy="12"]') && /M12/.test(paths)) name = "clock";
      else {
        const label = node.closest("button")?.getAttribute("aria-label") || node.closest("button")?.title || "";
        name = /sort/i.test(label) ? "sort" : /filter/i.test(label) ? "filter" : /close|remove|clear/i.test(label) ? "close" : "detail";
      }
    }
    node.dataset.icon = name;
    node.classList.add("ff-icon", "ff-icon--" + name);
    node.setAttribute("aria-hidden", "true");
    node.setAttribute("focusable", "false");
    observe(node);
  }

  function enhanceDeskEmpty() {
    const activeTab = document.querySelector("#desk-tab-matches.on, #desk-tab-route.on, #desk-tab-moments.on");
    if (!activeTab) return;
    const pane = activeTab.closest(".card")?.querySelector(".tab-pane > .empty");
    if (!pane || pane.dataset.ffIllustrated) return;
    pane.dataset.ffIllustrated = "true";
    pane.classList.add("ff-empty-state");
    const name = activeTab.id.endsWith("route") ? "route" : activeTab.id.endsWith("moments") ? "timeline" : "desk";
    pane.prepend(makeArt(name, "ff-empty-art ff-entrance"));
  }

  function enhanceChat() {
    const log = document.querySelector("#chatLog");
    if (!log) return;
    const conversation = !!log.querySelector(".msg.me");
    log.classList.toggle("ff-has-conversation", conversation);
    const first = log.querySelector(":scope > .msg.bot:first-child");
    if (first && !first.dataset.ffWelcome && !conversation) {
      first.dataset.ffWelcome = "true";
      first.prepend(makeArt("agent", "ff-chat-welcome ff-entrance"));
    }
  }

  const iconSelector = "button > svg, button > span > svg, .brand > svg, .st > svg, .sst > svg, .ck > svg, .spark > svg, .chev > svg, .vs-note > svg, .vd-note > span > svg, .tl-ev-ico > svg";
  const pending = new Set();
  let frame = 0;

  function scan(scope) {
    if (scope.nodeType !== 1) return;
    if (scope.matches(".ff-art, .ff-icon")) observe(scope);
    scope.querySelectorAll(".ff-art, .ff-icon").forEach(observe);
    if (scope.matches(iconSelector)) markIcon(scope);
    scope.querySelectorAll(iconSelector).forEach(markIcon);
  }

  function flush() {
    frame = 0;
    const scopes = [...pending]; pending.clear();
    for (const scope of scopes) if (scope.isConnected) scan(scope);
    enhanceDeskEmpty();
    enhanceChat();
  }

  /* New icons arrive through innerHTML in the existing views. Observe only added nodes, never animation attributes. */
  const mutations = new MutationObserver(records => {
    for (const record of records) {
      for (const node of record.addedNodes) if (node.nodeType === 1) pending.add(node);
      for (const node of record.removedNodes) {
        if (node.nodeType !== 1 || !intersection) continue;
        if (node.matches(".ff-art, .ff-icon")) { intersection.unobserve(node); observed.delete(node); }
        node.querySelectorAll(".ff-art, .ff-icon").forEach(icon => { intersection.unobserve(icon); observed.delete(icon); });
      }
    }
    if (!frame) frame = requestAnimationFrame(flush);
  });

  function init() {
    replaceGraphic("#agentCard .orb", "agent", "ff-agent-orb");
    replaceGraphic("#chat > .chat-h > .orb", "agent", "ff-agent-orb");
    replaceGraphic("#boot .boot-logo", "desk", "ff-boot-art");
    replaceGraphic("#attach", "camera", "ff-illustrated-control");
    replaceGraphic("#mic", "voice", "ff-illustrated-control");

    const heard = document.querySelector("#heard");
    if (heard && !heard.closest(".ff-privacy-note")) {
      const privacy = document.createElement("div");
      privacy.className = "ff-privacy-note";
      heard.before(privacy);
      privacy.append(makeArt("privacy", ""), heard);
    }

    document.addEventListener("error", event => {
      if (!(event.target instanceof HTMLImageElement)) return;
      const art = event.target.closest(".ff-art");
      if (art) art.classList.add("ff-art-failed");
    }, true);
    scan(document.body);
    /* Cached failures can finish before the capture listener was installed. */
    document.querySelectorAll(".ff-art > img").forEach(img => {
      if (img.complete && !img.naturalWidth) img.parentElement.classList.add("ff-art-failed");
    });
    mutations.observe(document.body, { childList: true, subtree: true });
    const visibility = () => document.documentElement.classList.toggle("ff-page-hidden", document.hidden);
    document.addEventListener("visibilitychange", visibility);
    visibility();
    STM.on("view", () => { pending.add(document.querySelector("#view")); if (!frame) frame = requestAnimationFrame(flush); });
    STM.on("desk", () => { if (!frame) frame = requestAnimationFrame(flush); });
  }

  STM.visualAssets = { files: { ...files }, base: assetBase, art: artMarkup };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, { once: true });
  else init();
})();
