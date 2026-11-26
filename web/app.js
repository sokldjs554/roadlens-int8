"use strict";

const $ = (id) => document.getElementById(id);
const views = [
  ["input", "원본 영상"],
  ["truth", "정답 마스크"],
  ["fp32", "FP32 예측"],
  ["int8", "INT8 예측"],
];
const percentage = (fraction) => (fraction * 100).toFixed(2);
const signed = (value) => {
  const rounded = Number(value.toFixed(2));
  return (rounded > 0 ? "+" : "") + rounded.toFixed(2);
};

function appendCell(row, tag, content) {
  const cell = document.createElement(tag);
  if (tag === "th") cell.scope = "row";
  cell.textContent = content;
  row.append(cell);
  return cell;
}

function renderMetrics(data) {
  $("sample").textContent = "평가 " + data.n + "장";
  $("device").textContent = data.device + " · TFLite / XNNPACK CPU";
  $("repeat").textContent = "배치 " + data.batch_size + " · 스레드 " + data.threads + " · 워밍업 " + data.warmup
    + "회 · 영상당 " + data.repeats + "회 · invoke() 구간";
  const latencyChange = (data.models.int8.p50_ms / data.models.fp32.p50_ms - 1) * 100;
  $("latency-change").textContent = Math.abs(latencyChange).toFixed(1) + "% " + (latencyChange <= 0 ? "감소" : "증가");
  $("glance-device").textContent = data.device + " · XNNPACK CPU";
  for (const modelName of ["fp32", "int8"]) {
    const model = data.models[modelName];
    const row = document.createElement("tr");
    appendCell(row, "th", modelName.toUpperCase());
    appendCell(row, "td", percentage(model.iou) + "%");
    appendCell(row, "td", percentage(model.accuracy) + "%");
    appendCell(row, "td", model.p50_ms.toFixed(3));
    appendCell(row, "td", (model.bytes / 1024).toFixed(1));
    $("glance-values").append(row);
  }
  const metrics = [
    ["도로 IoU", "iou", percentage, "%"],
    ["픽셀 정확도", "accuracy", percentage, "%"],
    ["지연시간 p50", "p50_ms", (n) => n.toFixed(3), "ms"],
    ["지연시간 p95", "p95_ms", (n) => n.toFixed(3), "ms"],
    ["모델 크기", "bytes", (n) => (n / 1024).toFixed(1), "KiB"],
    ["프로세스 RSS", "rss_mib", (n) => n.toFixed(1), "MiB"],
  ];
  for (const [label, key, format, unit] of metrics) {
    const row = document.createElement("tr");
    appendCell(row, "th", label);
    for (const model of [data.models.fp32, data.models.int8]) {
      const cell = appendCell(row, "td", format(model[key]));
      const suffix = document.createElement("small");
      suffix.textContent = unit;
      cell.append(suffix);
    }
    $("metrics").append(row);
  }
  for (const group of data.groups) {
    const row = document.createElement("tr");
    appendCell(row, "th", group.name);
    appendCell(row, "td", group.n + "장");
    appendCell(row, "td", percentage(group.fp32) + "%");
    appendCell(row, "td", percentage(group.int8) + "%");
    appendCell(row, "td", signed((group.int8 - group.fp32) * 100));
    $("group-rows").append(row);
  }
}

function createExplorer(scenes) {
  let index = 0;
  let mode = "split";
  const menu = $("scene-menu");
  const gallery = $("gallery");
  const wipe = $("wipe");
  const frame = $("frame");
  const modeButtons = [];
  const tiles = [];
  for (const [key, label] of views) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "mode";
    button.setAttribute("aria-label", label + " 확대");
    const image = new Image();
    image.alt = "";
    const title = document.createElement("span");
    title.textContent = label;
    button.append(image, title);
    button.addEventListener("click", () => selectMode(key));
    $("modes").append(button);
    modeButtons.push({ key, image, button });
  }
  scenes.forEach((scene, i) => {
    const option = new Option(scene.id + " · " + scene.condition, String(i));
    menu.add(option);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tile";
    button.setAttribute("aria-label", scene.id + " " + scene.condition + " 선택");
    const image = new Image();
    image.alt = "";
    image.loading = "lazy";
    image.src = scene.input;
    const title = document.createElement("b");
    title.textContent = scene.id;
    const condition = document.createElement("small");
    condition.textContent = scene.condition;
    const caption = document.createElement("span");
    caption.className = "tile-text";
    caption.append(title, condition);
    button.append(image, caption);
    button.addEventListener("click", () => selectScene(i));
    gallery.append(button);
    tiles.push(button);
  });

  function selectMode(next) {
    mode = next;
    const scene = scenes[index];
    const split = mode === "split";
    const label = views.find(([key]) => key === mode)?.[1];
    frame.classList.toggle("single", !split);
    $("view-base").src = split ? scene.int8 : scene[mode];
    $("view-base").alt = scene.id + " " + (split ? "오른쪽 INT8 도로 예측" : label);
    $("view-top").src = scene.fp32;
    $("view-top").alt = split ? scene.id + " 왼쪽 FP32 도로 예측" : "";
    $("left-tag").textContent = split ? "FP32" : label;
    $("split-button").setAttribute("aria-pressed", String(split));
    for (const item of modeButtons) {
      item.button.setAttribute("aria-pressed", String(item.key === mode));
    }
  }

  function selectScene(next) {
    index = next;
    const scene = scenes[index];
    const currentTile = tiles[index];
    menu.value = String(index);
    tiles.forEach((tile, i) => tile.setAttribute("aria-pressed", String(i === index)));
    for (const item of modeButtons) item.image.src = scene[item.key];
    $("position").textContent = String(index + 1).padStart(2, "0") + " / " + String(scenes.length).padStart(2, "0");
    $("prev").disabled = index === 0;
    $("next").disabled = index === scenes.length - 1;
    $("scene-stats").textContent = scene.id + " · " + scene.condition + " · " + scene.selection + "　|　장면 도로 IoU　FP32 " + percentage(scene.iou_fp32) + "%　·　INT8 " + percentage(scene.iou_int8) + "%";
    const tileRect = currentTile.getBoundingClientRect();
    const galleryRect = gallery.getBoundingClientRect();
    if (tileRect.left < galleryRect.left || tileRect.right > galleryRect.right) {
      gallery.scrollLeft += tileRect.left - galleryRect.left - 3;
    }
    selectMode(mode);
  }

  function setWipe() {
    frame.style.setProperty("--pos", wipe.value + "%");
    wipe.setAttribute("aria-valuetext", "왼쪽 FP32 " + wipe.value + "%, 오른쪽 INT8 " + (100 - Number(wipe.value)) + "%");
  }

  menu.addEventListener("change", () => selectScene(Number(menu.value)));
  $("prev").addEventListener("click", () => selectScene(index - 1));
  $("next").addEventListener("click", () => selectScene(index + 1));
  $("split-button").addEventListener("click", () => selectMode("split"));
  wipe.addEventListener("input", setWipe);
  setWipe();
  selectScene(0);
}

async function start() {
  try {
    const response = await fetch("results.json");
    if (!response.ok) throw new Error("결과 파일 불러오기 실패");
    const data = await response.json();
    if (!data.scenes?.length) throw new Error("평가 장면 없음");
    renderMetrics(data);
    $("dashboard").hidden = false;
    createExplorer(data.scenes);
    $("load-state").hidden = true;
  } catch {
    $("dashboard").hidden = true;
    $("load-state").textContent = "결과를 불러오지 못했습니다. ";
    const link = document.createElement("a");
    link.href = "results.json";
    link.textContent = "평가 결과 파일 열기";
    $("load-state").append(link);
  }
}
start();
