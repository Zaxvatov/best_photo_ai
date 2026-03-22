const state = {
  mediaMode: "photo",
  mergeSceneMode: true,
  showAllMetrics: false,
  filterPeople: true,
  filterLandscapes: true,
  filterDocuments: true,
  showSingleVideos: false,
  filterLivePhotoVideos: true,
  filterRegularVideos: true,
  groups: [],
  currentGroupId: null,
  selectedAssetIds: new Set(),
  metricLabels: {},
  metricHelp: {},
};

const STORAGE_KEY = "best-photo-ai-review-state";

const el = {
  groupList: document.getElementById("group-list"),
  groupTitle: document.getElementById("group-title"),
  groupView: document.getElementById("group-view"),
  status: document.getElementById("status"),
  groupSummary: document.getElementById("group-summary"),
  selectionSummary: document.getElementById("selection-summary"),
  mergeSceneMode: document.getElementById("merge-scene-mode"),
  showAllMetrics: document.getElementById("show-all-metrics"),
  filterPeople: document.getElementById("filter-people"),
  filterLandscapes: document.getElementById("filter-landscapes"),
  filterDocuments: document.getElementById("filter-documents"),
  showSingleVideos: document.getElementById("show-single-videos"),
  filterLivePhotoVideos: document.getElementById("filter-live-photo-videos"),
  filterRegularVideos: document.getElementById("filter-regular-videos"),
  photoFilters: document.getElementById("photo-filters"),
  videoFilters: document.getElementById("video-filters"),
  prevGroup: document.getElementById("prev-group"),
  nextGroup: document.getElementById("next-group"),
  deleteSelected: document.getElementById("delete-selected"),
  modeButtons: Array.from(document.querySelectorAll(".mode-btn")),
};

function queryString() {
  const params = new URLSearchParams({
    media_mode: state.mediaMode,
    merge_scene_mode: String(state.mergeSceneMode),
    show_all_metrics: String(state.showAllMetrics),
    filter_people: String(state.filterPeople),
    filter_landscapes: String(state.filterLandscapes),
    filter_documents: String(state.filterDocuments),
    show_single_videos: String(state.showSingleVideos),
    filter_live_photo_videos: String(state.filterLivePhotoVideos),
    filter_regular_videos: String(state.filterRegularVideos),
  });
  return params.toString();
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

function setStatus(text, isError = false) {
  el.status.textContent = text || "";
  el.status.style.color = isError ? "#ff7b7b" : "";
}

function setLoading(isLoading, text = "Загрузка...") {
  document.body.style.cursor = isLoading ? "progress" : "";
  if (isLoading) {
    setStatus(text, false);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function loadMeta() {
  const meta = await fetchJson("/api/meta");
  state.metricLabels = meta.metricLabels || {};
  state.metricHelp = meta.metricHelp || {};
}

function saveState() {
  const serializable = {
    mediaMode: state.mediaMode,
    mergeSceneMode: state.mergeSceneMode,
    showAllMetrics: state.showAllMetrics,
    filterPeople: state.filterPeople,
    filterLandscapes: state.filterLandscapes,
    filterDocuments: state.filterDocuments,
    showSingleVideos: state.showSingleVideos,
    filterLivePhotoVideos: state.filterLivePhotoVideos,
    filterRegularVideos: state.filterRegularVideos,
    currentGroupId: state.currentGroupId,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
}

function updateSelectionSummary() {
  el.selectionSummary.textContent = `Выбрано: ${state.selectedAssetIds.size}`;
}

function restoreState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    Object.assign(state, saved || {});
  } catch (_) {}
  el.mergeSceneMode.checked = !!state.mergeSceneMode;
  el.showAllMetrics.checked = !!state.showAllMetrics;
  el.filterPeople.checked = !!state.filterPeople;
  el.filterLandscapes.checked = !!state.filterLandscapes;
  el.filterDocuments.checked = !!state.filterDocuments;
  el.showSingleVideos.checked = !!state.showSingleVideos;
  el.filterLivePhotoVideos.checked = !!state.filterLivePhotoVideos;
  el.filterRegularVideos.checked = !!state.filterRegularVideos;
}

async function loadGroups() {
  setLoading(true, "Загрузка списка групп...");
  try {
    const data = await fetchJson(`/api/groups?${queryString()}`);
    state.groups = data.groups || [];
    el.groupSummary.textContent = `${state.mediaMode === "photo" ? "Сцен" : "Видео-групп"}: ${state.groups.length}`;
    if (!state.groups.length) {
      state.currentGroupId = null;
      renderGroupList();
      el.groupTitle.textContent = "Нет групп";
      el.groupView.innerHTML = '<div class="empty-state">По текущим фильтрам ничего не найдено.</div>';
      updateSelectionSummary();
      setStatus("");
      return;
    }
    if (!state.groups.some((item) => item.id === state.currentGroupId)) {
      state.currentGroupId = state.groups[0].id;
    }
    saveState();
    renderGroupList();
    await loadGroup(state.currentGroupId);
    setStatus("");
  } finally {
    setLoading(false);
  }
}

function renderGroupList() {
  el.groupList.innerHTML = "";
  for (const item of state.groups) {
    const fallbackParts = String(item.label || "").split(" ");
    const commonLabel = item.commonLabel ?? fallbackParts[0] ?? "";
    const privateLabel = item.privateLabel ?? String(item.label || "").slice(String(commonLabel).length).trim();
    const row = document.createElement("label");
    row.className = `group-item ${item.id === state.currentGroupId ? "active" : ""}`;
    row.title = item.label || "";
    row.innerHTML = `
      <input type="radio" name="group" ${item.id === state.currentGroupId ? "checked" : ""} />
      <span class="group-common">${escapeHtml(commonLabel)}</span>
      <span class="group-private">${escapeHtml(privateLabel)}${item.hasLivePhoto ? '<span class="live-dot"></span>' : ""}</span>
      <span class="group-size" title="Число элементов в группе">${escapeHtml(item.size ?? "")}</span>
    `;
    row.addEventListener("click", async () => {
      state.currentGroupId = item.id;
      state.selectedAssetIds.clear();
      saveState();
      renderGroupList();
      updateSelectionSummary();
      await loadGroup(item.id);
    });
    el.groupList.appendChild(row);
    if (item.id === state.currentGroupId) {
      requestAnimationFrame(() => {
        row.scrollIntoView({ block: "nearest" });
      });
    }
  }
}

function metricMaxValues(rows, metricOrder) {
  const maxMap = {};
  for (const metric of metricOrder) {
    let max = null;
    for (const row of rows) {
      const raw = row.metrics?.[metric]?.raw;
      const num = Number(raw);
      if (!Number.isNaN(num)) {
        max = max === null ? num : Math.max(max, num);
      }
    }
    maxMap[metric] = max;
  }
  return maxMap;
}

async function loadGroup(groupId) {
  if (groupId == null) return;
  setLoading(true, "Загрузка группы...");
  try {
    const data = await fetchJson(`/api/group/${groupId}?${queryString()}`);
    el.groupTitle.textContent = data.title || "";
    saveState();
    renderGroup(data);
  } finally {
    setLoading(false);
  }
}

function renderGroup(data) {
  const rows = data.rows || [];
  const metricOrder = data.metricOrder || [];
  const maxValues = metricMaxValues(rows, metricOrder);

  const shell = document.createElement("div");
  shell.className = "matrix-shell";

  const mediaGrid = document.createElement("div");
  mediaGrid.className = "media-grid";
  mediaGrid.style.setProperty("--cols", String(rows.length));

  const topLeft = document.createElement("div");
  mediaGrid.appendChild(topLeft);

  for (const row of rows) {
    const card = document.createElement("div");
    card.className = "card";
    const checked = state.selectedAssetIds.has(row.assetId) ? "checked" : "";
    const mediaSrc = `/api/media?path=${encodeURIComponent(row.filePath)}`;
    card.innerHTML = `
      <div class="media-select">
        <label><input type="checkbox" data-asset-id="${escapeHtml(row.assetId)}" ${checked} /></label>
      </div>
      <div class="media-shell">
        ${row.mediaType === "video"
          ? `<video src="${mediaSrc}" controls preload="metadata"></video>`
          : `<img src="${mediaSrc}" alt="${escapeHtml(row.fileName)}" />`}
      </div>
      <div class="filename ${row.isBest ? "best" : ""}" title="${escapeHtml(row.fileName)}">${escapeHtml(row.fileName)}</div>
    `;
    card.querySelector("input[type=checkbox]").addEventListener("change", (event) => {
      if (event.target.checked) state.selectedAssetIds.add(row.assetId);
      else state.selectedAssetIds.delete(row.assetId);
      updateSelectionSummary();
    });
    mediaGrid.appendChild(card);
  }

  const metricsTableWrap = document.createElement("div");
  metricsTableWrap.className = "metrics-table-wrap";

  const metricsTable = document.createElement("table");
  metricsTable.className = "metrics-table";

  const colgroup = document.createElement("colgroup");
  colgroup.innerHTML = `
    <col class="metric-label-col" />
    ${rows.map(() => '<col class="metric-value-col" />').join("")}
  `;
  metricsTable.appendChild(colgroup);

  const tbody = document.createElement("tbody");
  tbody.innerHTML = metricOrder.map((metric) => {
    const help = escapeHtml(state.metricHelp[metric] || "");
    const cells = rows.map((row) => {
      const metricData = row.metrics?.[metric] || {};
      const isMax = Number(metricData.raw) === maxValues[metric] && maxValues[metric] !== null;
      const cls = isMax ? "metric-value max" : "metric-value";
      return `<td class="${cls}" title="${escapeHtml(metricData.help || "")}">${escapeHtml(metricData.value ?? "")}</td>`;
    }).join("");
    return `
      <tr class="metric-row">
        <th class="metric-label" title="${help}" scope="row">${escapeHtml(state.metricLabels[metric] || metric)}</th>
        ${cells}
      </tr>
    `;
  }).join("");
  metricsTable.appendChild(tbody);
  metricsTableWrap.appendChild(metricsTable);

  shell.appendChild(mediaGrid);
  shell.appendChild(metricsTableWrap);
  el.groupView.innerHTML = "";
  el.groupView.appendChild(shell);
  updateSelectionSummary();
}

async function deleteSelected() {
  if (!state.selectedAssetIds.size) {
    setStatus("Ничего не выбрано");
    return;
  }
  const payload = {
    media_mode: state.mediaMode,
    asset_ids: Array.from(state.selectedAssetIds),
  };
  const result = await fetchJson("/api/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  setStatus(`Удалено asset: ${result.deletedAssets}, primary: ${result.deletedFiles}, sidecar: ${result.deletedSidecars}`);
  state.selectedAssetIds.clear();
  updateSelectionSummary();
  await loadGroups();
}

function syncControlVisibility() {
  const isPhoto = state.mediaMode === "photo";
  el.photoFilters.classList.toggle("hidden", !isPhoto);
  el.videoFilters.classList.toggle("hidden", isPhoto);
  el.mergeSceneMode.parentElement.classList.toggle("hidden", !isPhoto);
  el.modeButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === state.mediaMode));
}

function bindControls() {
  el.mergeSceneMode.addEventListener("change", async (e) => {
    state.mergeSceneMode = e.target.checked;
    saveState();
    await loadGroups();
  });
  el.showAllMetrics.addEventListener("change", async (e) => {
    state.showAllMetrics = e.target.checked;
    saveState();
    if (state.currentGroupId != null) await loadGroup(state.currentGroupId);
  });
  el.filterPeople.addEventListener("change", async (e) => { state.filterPeople = e.target.checked; saveState(); await loadGroups(); });
  el.filterLandscapes.addEventListener("change", async (e) => { state.filterLandscapes = e.target.checked; saveState(); await loadGroups(); });
  el.filterDocuments.addEventListener("change", async (e) => { state.filterDocuments = e.target.checked; saveState(); await loadGroups(); });
  el.showSingleVideos.addEventListener("change", async (e) => { state.showSingleVideos = e.target.checked; saveState(); await loadGroups(); });
  el.filterLivePhotoVideos.addEventListener("change", async (e) => { state.filterLivePhotoVideos = e.target.checked; saveState(); await loadGroups(); });
  el.filterRegularVideos.addEventListener("change", async (e) => { state.filterRegularVideos = e.target.checked; saveState(); await loadGroups(); });
  el.prevGroup.addEventListener("click", async () => {
    const index = state.groups.findIndex((g) => g.id === state.currentGroupId);
    if (index > 0) {
      state.currentGroupId = state.groups[index - 1].id;
      state.selectedAssetIds.clear();
      saveState();
      renderGroupList();
      updateSelectionSummary();
      await loadGroup(state.currentGroupId);
    }
  });
  el.nextGroup.addEventListener("click", async () => {
    const index = state.groups.findIndex((g) => g.id === state.currentGroupId);
    if (index >= 0 && index < state.groups.length - 1) {
      state.currentGroupId = state.groups[index + 1].id;
      state.selectedAssetIds.clear();
      saveState();
      renderGroupList();
      updateSelectionSummary();
      await loadGroup(state.currentGroupId);
    }
  });
  el.deleteSelected.addEventListener("click", deleteSelected);
  el.modeButtons.forEach((btn) => btn.addEventListener("click", async () => {
    state.mediaMode = btn.dataset.mode;
    state.selectedAssetIds.clear();
    saveState();
    syncControlVisibility();
    updateSelectionSummary();
    await loadGroups();
  }));

  document.addEventListener("keydown", async (event) => {
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (event.key === "ArrowUp" || event.key === "k") {
      event.preventDefault();
      el.prevGroup.click();
    } else if (event.key === "ArrowDown" || event.key === "j") {
      event.preventDefault();
      el.nextGroup.click();
    }
  });
}

async function main() {
  try {
    restoreState();
    bindControls();
    syncControlVisibility();
    updateSelectionSummary();
    await loadMeta();
    await loadGroups();
  } catch (error) {
    console.error(error);
    setStatus(String(error), true);
  }
}

main();
