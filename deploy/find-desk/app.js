"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    dataset: null,
    camera: "",
    description: "",
    refinements: [],
    candidates: [],
    selected: null,
    route: [],
    busy: false,
    request: 0,
    pollTimer: null,
    datasetTimer: null,
    noticeTimer: null,
    playingRoute: false,
    playlistIndex: -1,
    waitingForClip: false,
    storageKey: "",
    storageAvailable: true,
  };
  const voice = {
    available: false,
    phase: "idle",
    recorder: null,
    stream: null,
    chunks: [],
    started: 0,
    stopTimer: null,
    clockTimer: null,
    token: 0,
    target: "description",
  };

  function node(tag, className, text) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = text;
    return item;
  }

  function finite(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function clock(seconds) {
    const whole = Math.max(0, Math.floor(finite(seconds)));
    return [Math.floor(whole / 3600), Math.floor(whole / 60) % 60, whole % 60]
      .map((part) => String(part).padStart(2, "0")).join(":");
  }

  function cameraById(id) {
    return (state.dataset?.cameras || []).find((camera) => String(camera.id) === String(id));
  }

  function sceneStart(candidate) {
    const camera = cameraById(candidate.camera_id);
    return finite(candidate.clock_start_seconds, finite(camera?.clock_start_seconds)) + finite(candidate.confirmed_seconds, finite(candidate.start_seconds));
  }

  function keyFor(candidate) {
    return `${candidate.camera_id}|${candidate.id}|${candidate.start_seconds}`;
  }

  function confirmed(candidate) {
    return state.route.some((stop) => keyFor(stop) === keyFor(candidate));
  }

  function safeUrl(value, localOnly = false) {
    try {
      const url = new URL(String(value || ""), document.baseURI);
      if (!/^https?:$/.test(url.protocol) || (localOnly && url.origin !== location.origin)) return "";
      return url.href;
    } catch (_) {
      return "";
    }
  }

  async function api(path, options = {}) {
    let response;
    try {
      response = await fetch(new URL(path, document.baseURI), {
        credentials: "same-origin",
        cache: "no-store",
        ...options,
        headers: { "Content-Type": "application/json", ...options.headers },
      });
    } catch (_) {
      throw new Error("The search service could not be reached. Try again.");
    }
    let body;
    try {
      body = await response.json();
    } catch (_) {
      throw new Error(`The search service returned an unreadable response (${response.status}).`);
    }
    if (!response.ok) {
      const detail = typeof body.detail === "string" ? body.detail : body.error;
      throw new Error(detail || `The request failed (${response.status}). Try again.`);
    }
    return body;
  }

  function notice(message) {
    clearTimeout(state.noticeTimer);
    $("notice").textContent = message;
    $("notice").hidden = false;
    state.noticeTimer = setTimeout(() => { $("notice").hidden = true; }, 7000);
  }

  function status(message, working = false) {
    $("search-status").textContent = message;
    $("search-status").classList.toggle("working", working);
  }

  function setBusy(busy) {
    state.busy = busy;
    const editingLocked = busy || voice.phase !== "idle";
    $("search-button").disabled = editingLocked || !state.dataset?.ready;
    $("refine-button").disabled = editingLocked || !state.dataset?.ready || state.refinements.length >= 6;
    $("description").disabled = editingLocked;
    $("refinement").disabled = editingLocked;
    $("restart-search").disabled = editingLocked;
    $("search-form").setAttribute("aria-busy", String(editingLocked));
    $("refine-form").setAttribute("aria-busy", String(editingLocked));
    $("search-button").textContent = busy ? "Searching footage..." : "Find possible sightings";
    updateVoiceControls();
    renderCameras();
  }

  function addMessage(text, kind) {
    $("conversation").append(node("p", `search-message ${kind}`, text));
    $("conversation").scrollTop = $("conversation").scrollHeight;
  }

  function addModelNotes(text) {
    const details = node("details", "model-notes");
    details.append(node("summary", "", "Model notes"), node("p", "", text));
    $("conversation").append(details);
    $("conversation").scrollTop = $("conversation").scrollHeight;
  }

  function sourceCredit(dataset) {
    const target = $("source-credit");
    target.replaceChildren();
    const licenses = Array.isArray(dataset.licenses) ? dataset.licenses : [dataset.license].filter(Boolean);
    if (!licenses.length) {
      target.textContent = dataset.attribution || "Footage license information is being prepared.";
      return;
    }
    licenses.forEach((license, index) => {
      if (index) target.append(document.createTextNode("; "));
      if (typeof license === "string") {
        target.append(document.createTextNode(license));
        return;
      }
      if (license.attribution) target.append(document.createTextNode(`${license.attribution.replace(/[.\s]+$/, "")}. `));
      const title = license.title || license.name || "Source and license";
      const href = safeUrl(license.url || license.license_url);
      if (href) {
        const link = node("a", "", title);
        link.href = href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        target.append(link);
      } else {
        target.append(document.createTextNode(title));
      }
      if (license.summary) target.append(document.createTextNode(`. ${license.summary.replace(/\.+$/, "")}`));
      if (license.rights_note && license.rights_note !== license.summary) target.append(document.createTextNode(`. ${license.rights_note}`));
    });
  }

  function renderCameras() {
    const strip = $("camera-strip");
    strip.replaceChildren();
    const cameras = state.dataset?.cameras || [];
    if (!cameras.length) {
      strip.append(node("span", "muted", "Camera views are being prepared."));
      return;
    }
    [{ id: "", label: "All cameras" }, ...cameras].forEach((camera) => {
      const id = String(camera.id);
      const button = node("button", "camera-button");
      button.type = "button";
      button.classList.toggle("active", state.camera === id);
      button.classList.toggle("has-sighting", state.route.some((stop) => String(stop.camera_id) === id));
      button.setAttribute("aria-pressed", String(state.camera === id));
      button.disabled = state.busy || voice.phase !== "idle";
      button.append(node("span", "camera-label", camera.label || camera.id));
      button.append(node("span", "camera-time", id ? `${clock(camera.clock_start_seconds)} / ${Math.round(finite(camera.duration_seconds))}s` : `${cameras.length} views`));
      button.addEventListener("click", () => {
        state.camera = id;
        renderCameras();
        if (state.description && state.dataset?.ready) runSearch(false);
        else status(id ? `Search limited to ${camera.label || camera.id}.` : "Search across all cameras.");
      });
      strip.append(button);
    });
  }

  function loadRoute() {
    if (!state.storageKey) return;
    try {
      const data = JSON.parse(localStorage.getItem(state.storageKey) || "{}");
      const route = Array.isArray(data) ? data : data.route;
      state.route = Array.isArray(route) ? route.filter((candidate) => candidate && candidate.id && cameraById(candidate.camera_id) && safeUrl(candidate.video_url, true)).slice(0, 100) : [];
      if (state.route.length && typeof data.description === "string") {
        state.description = data.description.slice(0, 400);
        state.refinements = Array.isArray(data.refinements) ? data.refinements.filter((item) => typeof item === "string").slice(0, 6).map((item) => item.slice(0, 200)) : [];
        $("description").value = state.description;
        $("refine-form").hidden = false;
        $("restart-search").hidden = false;
        voice.target = "refinement";
      }
    } catch (_) {
      state.storageAvailable = false;
      state.route = [];
    }
  }

  function saveRoute() {
    try {
      localStorage.setItem(state.storageKey, JSON.stringify({ route: state.route, description: state.description, refinements: state.refinements }));
    } catch (_) {
      state.storageAvailable = false;
    }
    $("route-status").textContent = state.storageAvailable ? "Saved in this browser." : "Kept for this visit. Browser storage is unavailable.";
  }

  async function loadDataset() {
    clearTimeout(state.datasetTimer);
    try {
      const dataset = await api("api/dataset");
      const firstLoad = !state.dataset;
      state.dataset = dataset;
      $("dataset-title").textContent = dataset.title || "Connected camera footage";
      $("footage-kind").textContent = /synthetic/i.test(dataset.title || "") ? "Synthetic footage. Scene times are source offsets." : "Recorded footage. Scene time comes from the source.";
      $("connection").textContent = dataset.ready ? "Footage ready" : "Preparing footage";
      $("connection").className = `connection${dataset.ready ? " ready" : ""}`;
      const message = dataset.message || dataset.detail || (dataset.ready ? "" : dataset.status || "Footage is being prepared for search. This page will update when it is ready.");
      $("dataset-notice").textContent = typeof message === "string" ? message : "Footage is being prepared for search.";
      $("dataset-notice").hidden = !message;
      sourceCredit(dataset);
      if (dataset.method) $("method-description").textContent = dataset.method;
      if (firstLoad) {
        state.storageKey = `atlas-find-route:${dataset.id || dataset.title || "default"}:${(dataset.cameras || []).map((camera) => camera.id).join(",")}`;
        loadRoute();
        renderRoute();
      }
      setBusy(state.busy);
      if (!state.description) status(dataset.ready ? "Search across the available camera views." : "Waiting for footage.");
      if (!dataset.ready) state.datasetTimer = setTimeout(loadDataset, 10000);
    } catch (error) {
      $("connection").textContent = "Service unavailable";
      $("connection").className = "connection error";
      $("dataset-notice").textContent = `${error.message} Retrying shortly.`;
      $("dataset-notice").hidden = false;
      $("search-button").disabled = true;
      state.datasetTimer = setTimeout(loadDataset, 10000);
    }
  }

  function normalizeCandidate(candidate) {
    if (!candidate || !candidate.id || !candidate.camera_id) return null;
    const url = safeUrl(candidate.video_url, true);
    if (!url) return null;
    const camera = cameraById(candidate.camera_id);
    return {
      id: String(candidate.id),
      camera_id: String(candidate.camera_id),
      camera_label: String(candidate.camera_label || camera?.label || candidate.camera_id),
      start_seconds: Math.max(0, finite(candidate.start_seconds)),
      end_seconds: Math.max(0, finite(candidate.end_seconds, finite(candidate.start_seconds) + 5)),
      clock_start_seconds: finite(candidate.clock_start_seconds, finite(camera?.clock_start_seconds)),
      caption: String(candidate.caption || "Open the clip to check the person."),
      video_url: url,
    };
  }

  function renderCandidates() {
    const list = $("candidates");
    list.replaceChildren();
    if (!state.candidates.length) {
      list.append(node("p", "empty", state.description ? "No possible sightings found. Try fewer clothing details or choose all cameras." : "Results will appear here, with a clip for each sighting."));
      return;
    }
    state.candidates.forEach((candidate) => {
      const button = node("button", "candidate");
      button.type = "button";
      const selected = state.selected && keyFor(state.selected) === keyFor(candidate);
      button.classList.toggle("selected", Boolean(selected));
      button.setAttribute("aria-pressed", String(Boolean(selected)));
      const heading = node("span", "candidate-heading");
      heading.append(node("span", "", candidate.camera_label), node("time", "", clock(sceneStart(candidate))));
      button.append(heading, node("span", "candidate-text", candidate.caption));
      button.append(node("span", `candidate-state${confirmed(candidate) ? " confirmed" : ""}`, confirmed(candidate) ? "You confirmed" : "Possible sighting"));
      button.addEventListener("click", () => { stopPlaylist(); selectCandidate(candidate); });
      list.append(button);
    });
  }

  function updateClock() {
    if (!state.selected) {
      $("scene-clock").textContent = "--:--:--";
      return;
    }
    $("scene-clock").textContent = clock(state.selected.clock_start_seconds + finite($("video").currentTime, state.selected.start_seconds));
  }

  function updateConfirmation() {
    if (!state.selected) return;
    const yes = confirmed(state.selected);
    $("confirm-sighting").textContent = yes ? "You confirmed this sighting" : "Confirm sighting";
    $("confirm-sighting").disabled = yes;
    $("sighting-tag").textContent = yes ? "You confirmed" : "Possible sighting";
    $("sighting-tag").classList.toggle("confirmed", yes);
  }

  function selectCandidate(candidate, play = false) {
    state.selected = candidate;
    state.waitingForClip = true;
    const video = $("video");
    $("evidence-camera").textContent = candidate.camera_label;
    $("evidence-heading").textContent = "Check the person in the clip";
    $("candidate-caption").textContent = candidate.caption;
    const windowStart = candidate.clock_start_seconds + candidate.start_seconds;
    const confirmation = candidate.confirmed_seconds === undefined ? "Confirm the frame where you see the person." : `You confirmed the frame at ${clock(sceneStart(candidate))}.`;
    $("candidate-range").textContent = `Search window ${clock(windowStart)} to ${clock(candidate.clock_start_seconds + candidate.end_seconds)}. ${confirmation}`;
    $("evidence-detail").hidden = false;
    $("sighting-tag").hidden = false;
    $("stage-message").replaceChildren(node("p", "", "Describe someone to find a moment you can check."));
    $("stage-message").hidden = true;
    const seek = () => {
      if (state.selected !== candidate) return;
      const anchor = finite(candidate.confirmed_seconds, candidate.start_seconds);
      const end = Number.isFinite(video.duration) ? Math.max(0, video.duration - .1) : anchor;
      video.currentTime = Math.min(anchor, end);
      state.waitingForClip = false;
      updateClock();
      if (play) video.play().catch(() => {
        $("route-status").textContent = "Press play in the video to continue the route.";
      });
    };
    if (video.src !== candidate.video_url) {
      video.pause();
      video.src = candidate.video_url;
      video.addEventListener("loadedmetadata", seek, { once: true });
      video.load();
    } else if (video.readyState >= 1) {
      seek();
    } else {
      video.addEventListener("loadedmetadata", seek, { once: true });
    }
    $("scene-clock").textContent = clock(sceneStart(candidate));
    updateConfirmation();
    renderCandidates();
  }

  function renderRoute() {
    state.route.sort((a, b) => sceneStart(a) - sceneStart(b) || a.camera_label.localeCompare(b.camera_label));
    const list = $("route");
    list.replaceChildren();
    if (!state.route.length) list.append(node("li", "route-empty", "Start with a clear view of their clothes. Your confirmed sightings will connect here."));
    state.route.forEach((candidate) => {
      const item = node("li", "route-stop");
      const open = node("button", "open-stop");
      open.type = "button";
      open.setAttribute("aria-label", `Open confirmed sighting at ${clock(sceneStart(candidate))}, ${candidate.camera_label}`);
      open.append(node("time", "", clock(sceneStart(candidate))), node("span", "stop-camera", candidate.camera_label));
      open.addEventListener("click", () => { stopPlaylist(); selectCandidate(candidate); });
      const meta = node("div", "stop-meta");
      const remove = node("button", "remove-stop", "Remove");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove ${candidate.camera_label} sighting at ${clock(sceneStart(candidate))}`);
      remove.addEventListener("click", () => {
        stopPlaylist();
        state.route = state.route.filter((stop) => keyFor(stop) !== keyFor(candidate));
        saveRoute(); renderRoute(); renderCandidates(); updateConfirmation(); renderCameras();
      });
      meta.append(node("span", "", "You confirmed"), remove);
      item.append(open, meta);
      list.append(item);
    });
    $("route-count").textContent = state.route.length ? `${state.route.length} confirmed sighting${state.route.length === 1 ? "" : "s"}` : "No sightings confirmed";
    $("clear-route").disabled = !state.route.length;
    $("play-route").disabled = !state.route.length;
    $("route-status").textContent = state.storageAvailable ? "Saved in this browser." : "Kept for this visit. Browser storage is unavailable.";
  }

  function stopPlaylist() {
    state.playingRoute = false;
    state.playlistIndex = -1;
    $("play-route").textContent = "Play route";
  }

  function nextRouteStop() {
    state.playlistIndex += 1;
    if (state.playlistIndex >= state.route.length) {
      $("video").pause();
      stopPlaylist();
      $("route-status").textContent = "Route playback finished.";
      return;
    }
    const candidate = state.route[state.playlistIndex];
    $("route-status").textContent = `Playing ${state.playlistIndex + 1} of ${state.route.length}: ${candidate.camera_label}.`;
    selectCandidate(candidate, true);
  }

  function finishSearch(result) {
    state.candidates = (Array.isArray(result.candidates) ? result.candidates : []).map(normalizeCandidate).filter(Boolean);
    const count = state.candidates.length;
    $("result-count").textContent = `${count} possible sighting${count === 1 ? "" : "s"}`;
    const elapsed = finite(result.latency_ms);
    status(`${count} possible sighting${count === 1 ? "" : "s"} found${elapsed ? ` in ${(elapsed / 1000).toFixed(1)}s` : ""}.`);
    if (result.answer) addModelNotes(String(result.answer));
    else addMessage(count ? "Open a clip and check their clothes before confirming the sighting." : "Try fewer clothing details or search another camera.", "result");
    const limits = Array.isArray(result.limitations) ? result.limitations.filter(Boolean).join(" ") : result.limitations;
    const resultNote = [result.answer_error, limits].filter(Boolean).join(" ");
    $("result-limitations").textContent = resultNote;
    $("result-limitations").hidden = !resultNote;
    $("refine-form").hidden = false;
    $("restart-search").hidden = false;
    voice.target = "refinement";
    renderCandidates();
    if (count) selectCandidate(state.candidates[0]);
    setBusy(false);
    if (state.refinements.length >= 6) {
      $("refine-button").textContent = "Six details added. Start a new search to change them.";
    }
  }

  async function pollJob(id, request, started) {
    if (request !== state.request) return;
    try {
      const job = await api(`api/jobs/${encodeURIComponent(id)}`);
      if (request !== state.request) return;
      if (job.status === "done") {
        finishSearch(job.result || {});
      } else if (job.status === "error") {
        throw new Error(typeof job.detail === "string" ? job.detail : typeof job.error === "string" ? job.error : "The search failed. Try again with a shorter description.");
      } else if (Date.now() - started > 360000) {
        throw new Error("The search is taking longer than six minutes. Try again or check the service.");
      } else {
        const seconds = Math.round((Date.now() - started) / 1000);
        status(seconds > 8 ? `Searching footage. ${seconds}s elapsed.` : "Searching footage...", true);
        state.pollTimer = setTimeout(() => pollJob(id, request, started), 1200);
      }
    } catch (error) {
      if (request !== state.request) return;
      status(error.message);
      notice(error.message);
      setBusy(false);
    }
  }

  async function runSearch(addBaseMessage) {
    if (state.busy || !state.dataset?.ready) return;
    stopPlaylist();
    const request = ++state.request;
    clearTimeout(state.pollTimer);
    setBusy(true);
    status("Searching footage...", true);
    if (addBaseMessage) addMessage(state.description, "user");
    try {
      const body = { description: state.description, refinements: state.refinements };
      if (state.camera) body.camera_id = state.camera;
      const job = await api("api/search", { method: "POST", body: JSON.stringify(body) });
      if (request !== state.request) return;
      if (job.result && job.status === "done") finishSearch(job.result);
      else if (job.job_id) pollJob(job.job_id, request, Date.now());
      else throw new Error("The search service did not return a job. Try again.");
    } catch (error) {
      if (request !== state.request) return;
      status(error.message);
      notice(error.message);
      setBusy(false);
    }
  }

  $("search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const description = $("description").value.trim();
    if (!description || state.busy) return;
    if (state.description && state.description !== description) {
      state.refinements = [];
      state.route = [];
      saveRoute(); renderRoute();
      $("conversation").replaceChildren();
      $("refine-button").textContent = "Search with this detail";
      notice("New description. The previous route was cleared.");
    }
    state.description = description;
    runSearch(true);
  });

  $("refine-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const refinement = $("refinement").value.trim();
    if (!refinement || state.busy || state.refinements.length >= 6) return;
    state.refinements.push(refinement);
    addMessage(refinement, "user");
    $("refinement").value = "";
    runSearch(false);
  });

  $("restart-search").addEventListener("click", () => {
    state.request += 1;
    clearTimeout(state.pollTimer);
    stopPlaylist();
    state.description = "";
    state.refinements = [];
    state.candidates = [];
    state.selected = null;
    state.route = [];
    voice.target = "description";
    saveRoute(); renderRoute(); renderCameras();
    $("conversation").replaceChildren();
    $("description").value = "";
    $("refinement").value = "";
    $("refine-form").hidden = true;
    $("restart-search").hidden = true;
    $("refine-button").textContent = "Search with this detail";
    $("candidates").replaceChildren(node("p", "empty", "Results will appear here, with a clip for each sighting."));
    $("result-count").textContent = "No search yet";
    $("result-limitations").hidden = true;
    $("evidence-detail").hidden = true;
    $("sighting-tag").hidden = true;
    $("stage-message").replaceChildren(node("p", "", "Describe someone to find a moment you can check."));
    $("stage-message").hidden = false;
    $("video").pause();
    $("video").removeAttribute("src");
    $("video").load();
    $("evidence-camera").textContent = "Video evidence";
    updateClock();
    status("Describe an adult to start a new route.");
    $("description").focus();
  });

  $("confirm-sighting").addEventListener("click", () => {
    if (!state.selected || confirmed(state.selected)) return;
    stopPlaylist();
    const frame = $("video").readyState >= 1 ? finite($("video").currentTime, state.selected.start_seconds) : state.selected.start_seconds;
    const stop = { ...state.selected, confirmed_seconds: Math.max(0, frame) };
    state.route.push(stop);
    saveRoute(); renderRoute(); renderCameras(); renderCandidates(); updateConfirmation();
    $("route-status").textContent = `Sighting added at ${clock(sceneStart(stop))}. Search another view to continue.`;
  });

  $("clear-route").addEventListener("click", () => {
    stopPlaylist();
    state.route = [];
    saveRoute(); renderRoute(); renderCameras(); renderCandidates(); updateConfirmation();
    $("route-status").textContent = "Route cleared. Search results are still available.";
  });

  $("play-route").addEventListener("click", () => {
    if (state.playingRoute) { $("video").pause(); stopPlaylist(); return; }
    if (!state.route.length) return;
    state.playingRoute = true;
    state.playlistIndex = -1;
    $("play-route").textContent = "Stop route playback";
    nextRouteStop();
  });

  $("video").addEventListener("timeupdate", () => {
    updateClock();
    const playbackEnd = state.selected ? Math.max(state.selected.end_seconds, finite(state.selected.confirmed_seconds, state.selected.start_seconds) + 3) : 0;
    if (state.playingRoute && !state.waitingForClip && state.selected && $("video").currentTime >= playbackEnd) nextRouteStop();
  });
  $("video").addEventListener("ended", () => { if (state.playingRoute) nextRouteStop(); });
  $("video").addEventListener("error", () => {
    if (!state.selected) return;
    stopPlaylist();
    const stage = $("stage-message");
    stage.replaceChildren(node("p", "", "This recording could not be played. Choose another sighting or try the search again."));
    stage.hidden = false;
    notice("The video could not be loaded. The sighting has not been removed.");
  });

  function updateVoiceControls() {
    const button = $("record-voice");
    button.setAttribute("aria-pressed", String(voice.phase === "recording"));
    button.disabled = state.busy || !voice.available || !["idle", "recording", "permission"].includes(voice.phase);
    button.textContent = voice.phase === "recording" ? "Stop recording" : voice.phase === "permission" ? "Cancel" : voice.phase === "processing" ? "Transcribing..." : voice.target === "refinement" ? "Record detail" : "Record voice";
  }

  async function loadSpeech() {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      $("voice-status").textContent = "Voice needs localhost or HTTPS. Type your description here.";
      return;
    }
    if (!window.MediaRecorder || !window.AudioContext || !window.OfflineAudioContext) {
      $("voice-status").textContent = "Voice recording is unavailable in this browser. Type your description here.";
      return;
    }
    try {
      const speech = await api("api/speech");
      voice.available = speech.available === true;
      $("voice-status").textContent = voice.available ? "Canary transcription. Record up to 20s, then check the words." : "Voice is unavailable. Type your description here.";
    } catch (_) {
      $("voice-status").textContent = "Voice is unavailable. Type your description here.";
    }
    updateVoiceControls();
  }

  function releaseMicrophone() {
    clearTimeout(voice.stopTimer);
    clearInterval(voice.clockTimer);
    voice.stream?.getTracks().forEach((track) => track.stop());
    voice.stream = null;
  }

  function validateTranscript(target) {
    const tooLong = target.value.length > target.maxLength;
    target.setCustomValidity(tooLong ? `Shorten this text to ${target.maxLength} characters before searching.` : "");
    target.setAttribute("aria-invalid", String(tooLong));
    return tooLong;
  }

  async function wavFromRecording(blob) {
    const context = new AudioContext();
    let decoded;
    try {
      decoded = await context.decodeAudioData(await blob.arrayBuffer());
    } finally {
      await context.close().catch(() => {});
    }
    if (!Number.isFinite(decoded.duration) || decoded.duration < .2) throw new Error("The recording was too short. Record a few words or type your description.");
    const length = Math.max(1, Math.min(320000, Math.floor(decoded.duration * 16000)));
    const offline = new OfflineAudioContext(1, length, 16000);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start(0, 0, length / 16000);
    const samples = (await offline.startRendering()).getChannelData(0);
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const write = (offset, value) => {
      for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
    };
    write(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true);
    write(8, "WAVE"); write(12, "fmt "); view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, 16000, true); view.setUint32(28, 32000, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    write(36, "data"); view.setUint32(40, samples.length * 2, true);
    samples.forEach((sample, index) => {
      const value = Math.max(-1, Math.min(1, sample));
      view.setInt16(44 + index * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
    });
    return new Blob([buffer], { type: "audio/wav" });
  }

  async function transcribeRecording(token, mimeType, targetId) {
    if (token !== voice.token) return;
    releaseMicrophone();
    voice.phase = "processing";
    setBusy(state.busy);
    $("voice-status").textContent = "Canary is transcribing the recording...";
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 45000);
    try {
      const wav = await wavFromRecording(new Blob(voice.chunks, { type: mimeType }));
      voice.chunks = [];
      if (token !== voice.token) return;
      const result = await api("api/transcribe", { method: "POST", body: wav, headers: { "Content-Type": "audio/wav" }, signal: controller.signal });
      if (token !== voice.token) return;
      if (typeof result.text !== "string" || !result.text.trim()) throw new Error("Canary did not hear any words. Try again or type your description.");
      const target = $(targetId);
      target.value = result.text.trim();
      const tooLong = validateTranscript(target);
      const elapsed = finite(result.latency_ms);
      $("voice-status").textContent = tooLong ? `Canary returned ${target.value.length} characters. All words are shown; shorten the text to ${target.maxLength} characters before searching.` : `Canary transcription${elapsed ? ` in ${(elapsed / 1000).toFixed(2)}s` : ""}. Check the words, then search.`;
      voice.phase = "idle";
      setBusy(state.busy);
      target.focus();
    } catch (error) {
      if (token === voice.token) {
        voice.phase = "idle";
        $("voice-status").textContent = controller.signal.aborted ? "Canary took too long to respond. Try again or type your description." : error.message;
        setBusy(state.busy);
      }
    } finally {
      clearTimeout(timeout);
      voice.chunks = [];
      voice.recorder = null;
    }
  }

  function stopRecording() {
    if (voice.phase !== "recording" || !voice.recorder) return;
    clearTimeout(voice.stopTimer);
    clearInterval(voice.clockTimer);
    voice.phase = "processing";
    setBusy(state.busy);
    $("voice-status").textContent = "Preparing the recording for Canary...";
    voice.recorder.stop();
  }

  async function recordVoice() {
    if (voice.phase === "recording") { stopRecording(); return; }
    if (voice.phase === "permission") {
      voice.token += 1;
      voice.phase = "idle";
      $("voice-status").textContent = "Recording cancelled. You can type or record again.";
      setBusy(state.busy);
      return;
    }
    if (voice.phase !== "idle" || state.busy || !voice.available) return;
    const token = ++voice.token;
    const targetId = voice.target === "refinement" && !$("refine-form").hidden ? "refinement" : "description";
    voice.phase = "permission";
    setBusy(state.busy);
    $("voice-status").textContent = "Allow microphone access to record for Canary.";
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }, video: false });
      if (token !== voice.token) { stream.getTracks().forEach((track) => track.stop()); return; }
      voice.stream = stream;
      const mimeType = ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus"].find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      voice.recorder = recorder;
      voice.chunks = [];
      recorder.ondataavailable = (event) => { if (event.data.size && token === voice.token) voice.chunks.push(event.data); };
      recorder.onstop = () => transcribeRecording(token, recorder.mimeType, targetId);
      recorder.onerror = () => {
        if (token !== voice.token) return;
        voice.token += 1;
        releaseMicrophone();
        voice.phase = "idle";
        voice.chunks = [];
        $("voice-status").textContent = "The microphone recording failed. Try again or type your description.";
        setBusy(state.busy);
      };
      recorder.start();
      voice.started = performance.now();
      voice.phase = "recording";
      setBusy(state.busy);
      const update = () => {
        const elapsed = Math.min(20, Math.floor((performance.now() - voice.started) / 1000));
        $("voice-status").textContent = `Recording ${elapsed}s of 20s for Canary. Stop when finished.`;
      };
      update();
      voice.clockTimer = setInterval(update, 250);
      voice.stopTimer = setTimeout(stopRecording, 20000);
    } catch (error) {
      if (token !== voice.token) return;
      releaseMicrophone();
      voice.phase = "idle";
      const detail = error.name === "NotAllowedError" ? "Microphone access was denied. Allow it in the browser or type your description." : error.name === "NotFoundError" ? "No microphone was found. Connect one or type your description." : "The microphone could not start. Try again or type your description.";
      $("voice-status").textContent = detail;
      setBusy(state.busy);
    }
  }

  $("record-voice").addEventListener("click", recordVoice);
  ["description", "refinement"].forEach((id) => {
    $(id).addEventListener("focus", () => { if (voice.phase === "idle") { voice.target = id; updateVoiceControls(); } });
    $(id).addEventListener("input", () => validateTranscript($(id)));
  });
  window.addEventListener("pagehide", () => {
    voice.token += 1;
    if (voice.recorder?.state === "recording") voice.recorder.stop();
    releaseMicrophone();
  });

  loadDataset();
  loadSpeech();
})();
