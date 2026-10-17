(() => {
  "use strict";

  const guide = document.getElementById("project-guide");
  const content = document.getElementById("guide-content");
  const loadStatus = document.getElementById("guide-load-status");
  const workspace = document.getElementById("field-workspace");
  const inspectTab = document.getElementById("inspect-tab");
  const guideTab = document.getElementById("guide-tab");
  if (!guide || !content || !workspace || !inspectTab || !guideTab) return;

  const tabs = [inspectTab, guideTab];
  let loading = false;
  let loaded = false;
  if (loadStatus) {
    loadStatus.setAttribute("role", "status");
    loadStatus.setAttribute("aria-live", "polite");
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function paragraph(text, className) {
    return element("p", className, text);
  }

  function changeView(showGuide, updateHash = true) {
    workspace.hidden = showGuide;
    guide.hidden = !showGuide;
    tabs.forEach((tab, index) => {
      const selected = showGuide === (index === 1);
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    if (updateHash) {
      const url = new URL(window.location.href);
      url.hash = showGuide ? "projects" : "";
      window.history.replaceState(null, "", url);
    }
    if (showGuide) loadGuide();
  }

  tabs.forEach((tab, index) => {
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-controls", index === 0 ? "field-workspace" : "project-guide");
    tab.addEventListener("click", () => changeView(index === 1));
    tab.addEventListener("keydown", event => {
      let target;
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") target = 1 - index;
      else if (event.key === "Home") target = 0;
      else if (event.key === "End") target = 1;
      else return;
      event.preventDefault();
      tabs[target].focus();
      changeView(target === 1);
    });
  });
  workspace.setAttribute("role", "tabpanel");
  workspace.setAttribute("aria-labelledby", "inspect-tab");
  guide.setAttribute("role", "tabpanel");
  guide.setAttribute("aria-labelledby", "guide-tab");
  window.addEventListener("hashchange", () => changeView(window.location.hash === "#projects", false));

  function addSection(parent, heading, text, className = "") {
    if (!text) return;
    const section = element("section", `guide-section ${className}`.trim());
    section.append(element("h3", "", heading), paragraph(text));
    parent.append(section);
  }

  function addLink(parent, label, value) {
    if (!value) return;
    let url;
    try { url = new URL(value); } catch { return; }
    if (url.protocol !== "https:") return;
    const link = element("a", "guide-link", label);
    link.href = url.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    parent.append(link);
  }

  function projectRow(project) {
    const row = element("details", "guide-project");
    row.id = `guide-project-${project.slug}`;
    const summary = element("summary", "guide-project-summary");
    const overview = element("span", "guide-project-overview");
    const title = element("span", "guide-project-title", project.title);
    overview.append(title, element("span", "guide-project-what", project.what));
    const state = element("span", `guide-project-state${project.status === "connected" ? " is-connected" : ""}`,
      project.statusLabel || (project.status === "connected" ? "Connected here" : "Local prototype"));
    const toggle = element("span", "guide-project-toggle");
    toggle.setAttribute("aria-hidden", "true");
    summary.append(overview, state, toggle);

    const body = element("div", "guide-project-body");
    addSection(body, "When it helps", project.usefulness);
    if (project.example) addSection(body, "For example", project.example, "guide-example");

    const pipeline = element("section", "guide-section guide-project-pipeline");
    pipeline.append(element("h3", "", "Using it with the VAST pipeline"));
    const supplied = element("div", "guide-pipeline-part");
    supplied.append(element("h4", "", "VAST supplies"), paragraph(project.pipeline?.inputs || ""));
    const custom = element("div", "guide-pipeline-part");
    custom.append(element("h4", "", "This app adds"), paragraph(project.pipeline?.custom || ""));
    pipeline.append(supplied, custom);
    if (Array.isArray(project.pipeline?.steps) && project.pipeline.steps.length) {
      const steps = element("ol", "guide-project-steps");
      project.pipeline.steps.forEach(step => steps.append(element("li", "", step)));
      pipeline.append(steps);
    }
    body.append(pipeline);

    if (Array.isArray(project.features) && project.features.length) {
      const featureSection = element("section", "guide-section guide-features");
      featureSection.append(element("h3", "", `The ${project.features.length} tools inside ${project.title}`));
      const list = element("ul", "");
      project.features.forEach(feature => {
        const item = element("li", "");
        if (typeof feature === "string") item.textContent = feature;
        else {
          item.append(element("strong", "", feature.title || feature.name || ""));
          const description = feature.what || feature.description || feature.usefulness;
          if (description) item.append(document.createTextNode(`. ${description}`));
        }
        list.append(item);
      });
      featureSection.append(list);
      body.append(featureSection);
    }

    addSection(body, "What works today", project.statusDetail || project.integrationStatus || project.currentStatus);
    addSection(body, "Current limits", project.limit, "guide-limits");
    const actions = element("div", "guide-project-actions");
    if (project.id === "13" && project.status === "connected") {
      const tryCounts = element("button", "primary", "Try counts in Field Desk");
      tryCounts.type = "button";
      tryCounts.addEventListener("click", () => {
        changeView(false);
        inspectTab.focus();
        const mode = document.getElementById("mode");
        if (mode) {
          mode.value = "people";
          mode.dispatchEvent(new Event("change", { bubbles: true }));
        }
      });
      actions.append(tryCounts);
    }
    addLink(actions, "Watch the animated demo", project.demoUrl);
    addLink(actions, "Read the project and code", project.sourceUrl);
    if (actions.childElementCount) body.append(actions);
    row.append(summary, body);
    return row;
  }

  function searchText(value) {
    if (Array.isArray(value)) return value.map(searchText).join(" ");
    if (value && typeof value === "object") return Object.values(value).map(searchText).join(" ");
    return String(value || "");
  }

  function render(projects) {
    content.replaceChildren();
    const header = element("div", "guide-header");
    header.append(element("h1", "", "What each project does"),
      paragraph(`Explore our ${projects.length} apps and prototypes: what they do, when they help, and how they can use the VAST pipeline.`, "guide-intro"),
      paragraph("Pinch Point's counts are connected here. The other projects run separately; each entry explains its VAST connection.", "guide-connection-note"));

    const sharedFlow = element("details", "guide-shared-flow");
    sharedFlow.setAttribute("aria-labelledby", "guide-flow-title");
    sharedFlow.append(element("summary", "", "How the VAST pipeline fits"));
    sharedFlow.firstElementChild.id = "guide-flow-title";
    const flow = element("ol", "guide-flow-steps");
    [
      ["Store video", "VAST stores the footage in video chunks."],
      ["Describe and index", "Cosmos describes scenes, YOLO finds objects and Embed1 makes search vectors. VastDB stores the index."],
      ["Run a project", "Our app adds checks such as tracking, masking or camera health."],
      ["Review evidence", "Open the source timestamp and check the result before acting."]
    ].forEach(([label, description]) => {
      const step = element("li", "");
      step.append(element("strong", "", label), paragraph(description));
      flow.append(step);
    });
    sharedFlow.append(flow);

    const filters = element("div", "guide-filters");
    const searchLabel = element("label", "guide-search-label", "Find a project or use case");
    searchLabel.htmlFor = "guide-search";
    const search = element("input", "");
    search.id = "guide-search";
    search.type = "search";
    search.placeholder = "e.g. redaction, parking, camera health";
    search.autocomplete = "off";
    const searchGroup = element("div", "guide-search-group");
    searchGroup.append(searchLabel, search);
    const connectedLabel = element("label", "guide-connected-filter");
    const connected = element("input", "");
    connected.type = "checkbox";
    connected.id = "guide-connected-only";
    connectedLabel.append(connected, document.createTextNode("Connected in Field Desk"));
    filters.append(searchGroup, connectedLabel);

    const count = paragraph("", "guide-result-count");
    count.id = "guide-result-count";
    count.setAttribute("role", "status");
    count.setAttribute("aria-live", "polite");
    count.setAttribute("aria-atomic", "true");
    const list = element("div", "guide-project-list");
    list.setAttribute("aria-label", "Project explanations");
    const entries = projects.map(project => {
      const row = projectRow(project);
      list.append(row);
      return { project, row, text: searchText(project).toLocaleLowerCase() };
    });
    const empty = element("div", "guide-empty");
    empty.hidden = true;
    empty.append(element("h2", "", "No projects match those filters"), paragraph("Try a shorter search or show all projects."));
    const reset = element("button", "", "Show all projects");
    reset.type = "button";
    empty.append(reset);

    function filterProjects() {
      const terms = search.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
      let visible = 0;
      entries.forEach(({ project, row, text }) => {
        const matches = (!connected.checked || project.status === "connected") && terms.every(term => text.includes(term));
        row.hidden = !matches;
        if (matches) visible += 1;
      });
      count.textContent = visible === projects.length ? `${visible} projects. Open a row for its use, VAST connection and limits.` : `${visible} of ${projects.length} projects shown.`;
      empty.hidden = visible !== 0;
    }
    search.addEventListener("input", filterProjects);
    connected.addEventListener("change", filterProjects);
    reset.addEventListener("click", () => {
      search.value = "";
      connected.checked = false;
      filterProjects();
      search.focus();
    });
    content.append(header, sharedFlow, filters, count, list, empty);
    filterProjects();
  }

  async function loadGuide() {
    if (loaded || loading) return;
    loading = true;
    guide.setAttribute("aria-busy", "true");
    if (loadStatus) {
      loadStatus.hidden = false;
      loadStatus.textContent = "Loading the project guide…";
    }
    try {
      const response = await fetch("project-guide.json", { cache: "no-cache" });
      if (!response.ok) throw new Error("Guide unavailable");
      const data = await response.json();
      if (!Array.isArray(data.projects) || !data.projects.length) throw new Error("Empty guide");
      render(data.projects);
      loaded = true;
      if (loadStatus) loadStatus.hidden = true;
    } catch {
      if (loadStatus) loadStatus.textContent = "The project guide could not load. You can still use the footage inspector.";
      const retry = element("button", "guide-retry", "Retry project guide");
      retry.type = "button";
      retry.addEventListener("click", () => {
        content.replaceChildren();
        loadGuide();
      });
      content.replaceChildren(retry);
    } finally {
      loading = false;
      guide.removeAttribute("aria-busy");
    }
  }

  changeView(window.location.hash === "#projects", false);
})();
