/*
  STEP 8 파이프라인 대시보드 동작

  플로우차트를 Mermaid.js가 아니라 인라인 SVG로 그린 이유
  1. 1초에 5번(200ms마다) 노드의 ms 숫자, 활성 테두리, 병목 표시가 바뀐다.
     Mermaid는 글(flowchart TD 문법)을 통째로 다시 그려야 해서 깜빡이고 느리다.
     인라인 SVG는 이미 그려 둔 노드의 글자와 클래스만 바꾸면 된다.
  2. 노드 클릭, 키보드 선택, 흐리게 표시 같은 상태를 CSS 클래스로 바로 붙일 수 있다.
     색도 CSS 변수를 그대로 따르므로 다크/라이트 모드가 저절로 맞는다.
  3. 외부 라이브러리(CDN)가 필요 없어 인터넷이 없는 실습실에서도 동작한다.
  단계가 6개뿐인 한 줄짜리 흐름이라, 자동 배치 기능(Mermaid의 장점)이 없어도 좌표를 직접 정하기 쉽다.
*/

"use strict";

// ultralytics model.predict() 한 번 안에서 일어나는 단계
const YOLO_STAGES = ["preprocess", "inference", "postprocess"];

// 로그 콘솔에 남길 최대 줄 수
const MAX_LINES = 500;

// 플로우차트 노드 배치 (SVG 좌표)
const LAYOUT = { width: 280, left: 28, nodeWidth: 228, nodeHeight: 40, gap: 34, top: 8 };

const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  meta: [],
  nodes: {},
  selected: null,
  paused: false,
  pending: [],
};

const $ = (id) => document.getElementById(id);


/* ============================================================
   1. 플로우차트 (인라인 SVG)
   ============================================================ */

/**
 * SVG 요소를 만들고 속성을 붙인다.
 * @param {string} tag 요소 이름 (rect, text ...)
 * @param {object} attrs 속성 이름 → 값
 * @param {Element} parent 붙일 부모 요소 (없으면 붙이지 않음)
 * @returns {Element} 만든 요소
 */
function svg(tag, attrs, parent) {
  const element = document.createElementNS(SVG_NS, tag);

  for (const [name, value] of Object.entries(attrs || {})) {
    element.setAttribute(name, value);
  }

  if (parent) {
    parent.appendChild(element);
  }

  return element;
}


/**
 * /api/pipeline의 단계 목록으로 위에서 아래로 흐르는(TD) 플로우차트를 그린다.
 * 노드 아래 화살표 옆에 평균 ms를 쓰고, YOLO 내부 세 단계는 점선 상자로 묶는다.
 * @param {Array} meta 단계 메타데이터 목록
 */
function buildChart(meta) {
  const { width, left, nodeWidth, nodeHeight, gap, top } = LAYOUT;
  const height = top + meta.length * (nodeHeight + gap);
  const centerX = left + nodeWidth / 2;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });

  const defs = svg("defs", {}, root);
  const marker = svg("marker", {
    id: "arrow", viewBox: "0 0 8 8", refX: "7", refY: "4",
    markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse",
  }, defs);
  svg("path", { d: "M0,0 L8,4 L0,8 z", class: "arrowhead" }, marker);

  const yOf = (index) => top + index * (nodeHeight + gap);

  // YOLO 내부 단계를 점선 상자로 묶는다: "이 셋은 model.predict() 한 번 안에 숨어 있다"
  const groupIndexes = meta.map((item, i) => (item.external ? i : -1)).filter((i) => i >= 0);

  if (groupIndexes.length > 0) {
    const first = yOf(groupIndexes[0]) - 8;
    const last = yOf(groupIndexes[groupIndexes.length - 1]) + nodeHeight + 8;
    svg("rect", { x: left - 10, y: first, width: nodeWidth + 20, height: last - first, class: "group-box" }, root);

    const labelY = (first + last) / 2;
    const label = svg("text", {
      x: left - 16, y: labelY, class: "group-label", "text-anchor": "middle",
      transform: `rotate(-90 ${left - 16} ${labelY})`,
    }, root);
    label.textContent = "model.predict() 1회";
  }

  meta.forEach((item, index) => {
    const y = yOf(index);
    const group = svg("g", { class: "node", tabindex: "0", role: "button", "aria-label": `${item.id} 단계 설명 보기` }, root);

    svg("rect", { x: left, y, width: nodeWidth, height: nodeHeight, rx: "2", class: "box" }, group);

    const title = svg("text", { x: left + 12, y: y + 25 }, group);
    const name = svg("tspan", { class: "name" }, title);
    name.textContent = item.id;
    const label = svg("tspan", { class: "label", dx: "8" }, title);
    label.textContent = item.label;

    const badge = svg("text", { x: left + nodeWidth - 10, y: y + 25, "text-anchor": "end", class: "badge" }, group);
    badge.textContent = "병목";

    if (index < meta.length - 1) {
      svg("line", {
        x1: centerX, y1: y + nodeHeight, x2: centerX, y2: y + nodeHeight + gap - 2,
        class: "edge", "marker-end": "url(#arrow)",
      }, root);
    }

    const ms = svg("text", { x: centerX + 10, y: y + nodeHeight + gap / 2 + 4, class: "ms" }, root);
    ms.textContent = "- ms";

    group.addEventListener("click", () => selectNode(item.id));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectNode(item.id);
      }
    });

    state.nodes[item.id] = { group, ms, item };
  });

  $("chart").replaceChildren(root);
}


/**
 * 노드를 선택하고 상세 패널에 docstring 설명을 보여 준다.
 * @param {string} stageId 단계 이름
 */
function selectNode(stageId) {
  state.selected = stageId;

  for (const [id, node] of Object.entries(state.nodes)) {
    node.group.classList.toggle("is-selected", id === stageId);
  }

  showDetail(state.nodes[stageId].item);
}


/**
 * 상세 패널을 채운다. 글자는 모두 textContent로 넣어 HTML로 해석되지 않게 한다.
 * @param {object} item 단계 메타데이터
 */
function showDetail(item) {
  const panel = $("detail");
  const parts = [];

  const heading = document.createElement("h3");
  heading.textContent = `${item.id} · ${item.label}`;
  parts.push(heading);

  const fn = document.createElement("p");
  fn.className = "fn";
  fn.textContent = item.function ? `${item.function}()` : "등록된 함수 없음";

  if (item.external) {
    fn.textContent += " · ultralytics 내부 단계, results[0].speed로 측정";
  }

  parts.push(fn);

  if (!item.parsed) {
    const warning = document.createElement("p");
    warning.className = "warning";
    warning.textContent = "docstring을 읽지 못해 이름만 표시합니다. 서버 콘솔의 경고를 확인하세요.";
    parts.push(warning);
  }

  const list = document.createElement("dl");
  const fields = [["summary", "무엇을 하나"], ["input", "입력"], ["output", "출력"], ["reason", "왜 이렇게 하나"]];

  for (const [key, title] of fields) {
    if (!item[key]) {
      continue;
    }

    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = title;
    dd.textContent = item[key];
    list.append(dt, dd);
  }

  if (item.params && item.params.length > 0) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    const ul = document.createElement("ul");
    dt.textContent = "매개변수";
    ul.className = "params";

    for (const param of item.params) {
      const li = document.createElement("li");
      const code = document.createElement("code");
      code.textContent = param.name;
      li.append(code, ` ${param.text}`);
      ul.appendChild(li);
    }

    dd.appendChild(ul);
    list.append(dt, dd);
  }

  parts.push(list);
  panel.replaceChildren(...parts);
}


/**
 * ms 숫자를 보기 좋게 줄인다. 작을수록 소수점을 더 보여 준다.
 * @param {number} ms 밀리초
 * @returns {string} 예: "0.42", "3.8", "38"
 */
function formatMs(ms) {
  if (ms < 1) {
    return ms.toFixed(2);
  }

  if (ms < 10) {
    return ms.toFixed(1);
  }

  return ms.toFixed(0);
}


/**
 * 200ms마다 받은 집계로 노드의 ms, 활성 테두리, 병목, 흐림을 바꾼다.
 * @param {object} snap 서버 집계
 */
function updateChart(snap) {
  for (const [id, node] of Object.entries(state.nodes)) {
    const ms = snap.stage_ms[id];
    const isBottleneck = snap.bottleneck === id;
    const isSkipped = snap.skipped && YOLO_STAGES.includes(id);

    if (ms === null || ms === undefined) {
      node.ms.textContent = "- ms";
    } else {
      const share = snap.share[id] || 0;
      node.ms.textContent = `${formatMs(ms)} ms · ${share.toFixed(0)}%`;
    }

    if (isSkipped) {
      node.ms.textContent += " · skip";
    }

    node.group.classList.toggle("is-active", snap.active === id);
    node.group.classList.toggle("is-bottleneck", isBottleneck);
    node.group.classList.toggle("is-skipped", isSkipped);
    node.ms.classList.toggle("is-bottleneck", isBottleneck);
  }

  if (snap.active) {
    $("active-text").textContent = `실행 중: ${snap.active}`;
  } else {
    $("active-text").textContent = "대기 중 (다음 카메라 프레임)";
  }
}


/**
 * "한 프레임 시간 구성" 막대를 그린다. 단계별 평균 ms의 비율만큼 폭을 나눈다.
 * GPU로 바꾸면 inference 칸이 얼마나 줄어드는지 한눈에 비교할 수 있다.
 * @param {object} snap 서버 집계
 */
function updateStack(snap) {
  const stack = $("stack");
  const legend = $("stack-legend");
  const segments = [];
  const legendItems = [];

  for (const item of state.meta) {
    const ms = snap.stage_ms[item.id];

    if (ms === null || ms === undefined) {
      continue;
    }

    const share = snap.share[item.id] || 0;
    const isBottleneck = snap.bottleneck === item.id;

    const segment = document.createElement("span");
    segment.style.width = `${share}%`;
    segment.title = `${item.id} ${formatMs(ms)} ms (${share.toFixed(1)}%)`;
    segment.classList.toggle("is-bottleneck", isBottleneck);
    segments.push(segment);

    const text = document.createElement("span");
    text.textContent = `${item.id} ${share.toFixed(0)}%`;
    text.classList.toggle("is-bottleneck", isBottleneck);
    legendItems.push(text);
  }

  stack.replaceChildren(...segments);
  legend.replaceChildren(...legendItems);
  $("total-ms").textContent = snap.total_ms > 0 ? `합계 ${formatMs(snap.total_ms)} ms` : "-";
}


/* ============================================================
   2. 상단 바
   ============================================================ */

/**
 * 상단 바의 FPS와 설정 값을 바꾼다.
 * @param {object} snap 서버 집계
 */
function updateTopbar(snap) {
  $("fps").textContent = snap.fps.toFixed(1);
  $("device").textContent = snap.device;
  $("imgsz").textContent = snap.imgsz;
  $("conf").textContent = snap.conf;
  $("skip").textContent = snap.skip;
  $("frame").textContent = `#${String(snap.frame).padStart(4, "0")}`;
}


/* ============================================================
   3. 로그 콘솔
   ============================================================ */

/**
 * 새 로그 줄을 받는다. 일시정지 중이면 쌓아 두고, 아니면 바로 콘솔에 붙인다.
 * @param {Array} lines [{seq, level, text}] 목록
 * @param {number} dropped 200ms 동안 너무 많아 서버가 생략한 줄 수
 */
function receiveLogs(lines, dropped) {
  const batch = lines.slice();

  if (dropped > 0) {
    batch.unshift({ level: "notice", text: `… ${dropped}줄 생략 (200ms 동안 로그가 너무 많음)` });
  }

  if (batch.length === 0) {
    return;
  }

  if (state.paused) {
    state.pending.push(...batch);

    if (state.pending.length > MAX_LINES) {
      state.pending.splice(0, state.pending.length - MAX_LINES);
    }

    $("log-count").textContent = `일시정지 중 · 새 ${state.pending.length}줄`;
    return;
  }

  appendLines(batch);
}


/**
 * 콘솔 아래쪽에 줄을 붙이고, 오래된 줄은 지운다.
 * 사용자가 위로 스크롤해 읽는 중이면 자동 스크롤을 멈춘다.
 * @param {Array} lines [{level, text}] 목록
 */
function appendLines(lines) {
  const consoleBox = $("console");
  const atBottom = consoleBox.scrollHeight - consoleBox.scrollTop - consoleBox.clientHeight < 24;
  const fragment = document.createDocumentFragment();

  for (const line of lines) {
    const row = document.createElement("span");
    row.className = `line ${line.level}`;
    row.textContent = line.text;
    fragment.appendChild(row);
  }

  consoleBox.appendChild(fragment);

  while (consoleBox.childElementCount > MAX_LINES) {
    consoleBox.firstElementChild.remove();
  }

  if (atBottom) {
    consoleBox.scrollTop = consoleBox.scrollHeight;
  }

  $("log-count").textContent = `${consoleBox.childElementCount}줄`;
}


/**
 * 일시정지 버튼과 레벨 필터를 연결한다.
 * 필터는 줄을 지우지 않고 콘솔에 hide-레벨 클래스를 붙여 숨기기만 한다.
 */
function setupLogControls() {
  const pause = $("pause-button");

  pause.addEventListener("click", () => {
    state.paused = !state.paused;
    pause.setAttribute("aria-pressed", String(state.paused));
    pause.textContent = state.paused ? "재개" : "일시정지";

    if (!state.paused && state.pending.length > 0) {
      const lines = state.pending;
      state.pending = [];
      appendLines(lines);
    } else if (state.paused) {
      $("log-count").textContent = "일시정지 중";
    }
  });

  for (const box of document.querySelectorAll(".log-toolbar input[data-level]")) {
    box.addEventListener("change", () => {
      $("console").classList.toggle(`hide-${box.dataset.level}`, !box.checked);
    });
  }
}


/* ============================================================
   4. 실시간 연결 (Server-Sent Events)
   ============================================================ */

/**
 * /events에 연결한다. EventSource는 연결이 끊기면 스스로 다시 붙는다.
 */
function connectEvents() {
  const conn = $("conn");
  const source = new EventSource("/events");
  let lastError = "";

  source.onopen = () => {
    conn.textContent = "실시간";
    conn.className = "conn live";
  };

  source.onmessage = (event) => {
    const snap = JSON.parse(event.data);

    updateTopbar(snap);

    if (!$("camera-toggle").disabled) {
      showCameraState(snap.camera_on);
    }

    updateChart(snap);
    updateStack(snap);
    receiveLogs(snap.logs, snap.dropped_logs);

    if (snap.error && snap.error !== lastError) {
      lastError = snap.error;
      receiveLogs([{ level: "WARN", text: `[서버] ${snap.error}` }], 0);
    }
  };

  source.onerror = () => {
    conn.textContent = "연결 끊김 · 다시 연결 중";
    conn.className = "conn lost";
  };
}


/* ============================================================
   5. 촬영과 저장 이력 (STEP 3 기능 유지)
   ============================================================ */

/**
 * 저장 이력 썸네일 하나를 맨 앞에 넣는다. 최근 HISTORY_LIMIT장만 남긴다.
 * @param {object} item {filename, url}
 */
function addThumb(item) {
  const list = $("thumbs");
  const li = document.createElement("li");
  const link = document.createElement("a");
  const img = document.createElement("img");

  link.href = item.url;
  link.target = "_blank";
  link.rel = "noopener";
  link.title = item.filename;
  img.src = item.url;
  img.alt = item.filename;
  img.loading = "lazy";

  link.appendChild(img);
  li.appendChild(link);
  list.prepend(li);

  while (list.childElementCount > HISTORY_LIMIT) {
    list.lastElementChild.remove();
  }

  $("history-empty").hidden = list.childElementCount > 0;
}


/**
 * 페이지를 열 때 서버에 저장된 이력을 불러온다.
 */
async function loadHistory() {
  try {
    const res = await fetch("/captures");
    const data = await res.json();

    for (const item of data.captures.slice().reverse()) {
      addThumb(item);
    }
  } catch (error) {
    $("capture-status").textContent = "저장 이력을 불러오지 못했습니다.";
  }
}


/**
 * POST /capture로 지금 탐지 결과 프레임을 저장한다.
 */
async function capture() {
  const button = $("capture-button");
  const status = $("capture-status");

  button.disabled = true;
  status.textContent = "저장 중...";

  try {
    const res = await fetch("/capture", { method: "POST" });
    const data = await res.json();

    if (!res.ok) {
      status.textContent = data.error || "저장에 실패했습니다.";
      return;
    }

    addThumb(data);
    status.textContent = `저장: captures/detected/${data.filename} (#frame ${String(data.frame).padStart(4, "0")})`;
  } catch (error) {
    status.textContent = "서버에 연결할 수 없습니다.";
  } finally {
    button.disabled = false;
  }
}


/**
 * 촬영 버튼과 스페이스 키를 연결한다.
 */
function setupCapture() {
  const button = $("capture-button");
  button.addEventListener("click", capture);

  document.addEventListener("keydown", (event) => {
    if (event.code !== "Space" || event.repeat || button.disabled) {
      return;
    }

    const tag = document.activeElement ? document.activeElement.tagName : "";

    if (tag === "BUTTON" || tag === "INPUT" || document.activeElement.classList.contains("node")) {
      return;
    }

    event.preventDefault();
    capture();
  });

  $("stream").addEventListener("error", () => {
    $("stream-error").hidden = false;
  });
}


/* ============================================================
   6. 카메라 켜기/끄기
   ============================================================ */

/**
 * 상단 바의 카메라 버튼 글자를 지금 상태에 맞춘다.
 * @param {boolean} isOn 카메라가 켜져 있는지
 */
function showCameraState(isOn) {
  const button = $("camera-toggle");
  button.dataset.on = String(isOn);
  button.textContent = isOn ? "카메라 끄기" : "카메라 켜기";
  button.setAttribute("aria-pressed", String(!isOn));
}


/**
 * 카메라 버튼을 연결한다. 끄면 서버가 카메라 장치를 해제해 웹캠 표시등도 꺼진다.
 * 상태는 200ms마다 오는 집계(camera_on)로도 맞춰지므로, /camera 페이지에서 바꿔도 이 버튼이 따라온다.
 */
function setupCameraToggle() {
  const button = $("camera-toggle");

  button.addEventListener("click", async () => {
    const turnOn = button.dataset.on === "false";
    button.disabled = true;
    button.textContent = turnOn ? "켜는 중..." : "끄는 중...";

    try {
      const res = await fetch(`/camera/${turnOn ? "on" : "off"}`, {
        method: "POST",
        headers: { Accept: "application/json" },
      });
      const data = await res.json();
      showCameraState(data.on);
      $("capture-status").textContent = data.message;
    } catch (error) {
      $("capture-status").textContent = "서버에 연결할 수 없습니다.";
    } finally {
      button.disabled = false;
    }
  });
}


/* ============================================================
   7. 테마 (다크/라이트)
   ============================================================ */

/**
 * 테마 버튼을 연결한다. 고른 테마는 이 브라우저에만 기억한다.
 * 저장소를 쓸 수 없는 환경(사생활 보호 창 등)에서도 오류 없이 동작하게 try로 감싼다.
 */
function setupTheme() {
  const root = document.documentElement;

  try {
    const saved = localStorage.getItem("step8-theme");

    if (saved === "light" || saved === "dark") {
      root.dataset.theme = saved;
    }
  } catch (error) {
    // 저장소를 못 쓰면 시스템 설정을 따른다.
  }

  $("theme-toggle").addEventListener("click", () => {
    const current = root.dataset.theme
      || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    root.dataset.theme = next;

    try {
      localStorage.setItem("step8-theme", next);
    } catch (error) {
      // 기억하지 못해도 이번 화면에는 적용된다.
    }
  });
}


/* ============================================================
   8. 시작
   ============================================================ */

/**
 * 페이지가 열리면 단계 목록을 받아 플로우차트를 그리고 실시간 연결을 시작한다.
 */
async function init() {
  setupTheme();
  setupLogControls();
  setupCapture();
  setupCameraToggle();
  loadHistory();

  try {
    const res = await fetch("/api/pipeline");
    state.meta = await res.json();
    buildChart(state.meta);
  } catch (error) {
    $("chart").textContent = "단계 목록(/api/pipeline)을 불러오지 못했습니다.";
  }

  connectEvents();
}

init();
