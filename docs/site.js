const RESULTS_PATH = "mvs_pointcloud_results.csv";

const numericColumns = [
  "completeness_fraction",
  "l1_abs_m",
  "rmse_m",
  "chamfer_m",
  "f1_percent",
  "points_before_downsampling",
  "runtime_min",
];

const metricDirections = {
  completeness_fraction: "max",
  l1_abs_m: "min",
  rmse_m: "min",
  chamfer_m: "min",
  f1_percent: "max",
  runtime_min: "min",
};

const formatters = {
  completeness_fraction: (value) => value.toFixed(4),
  l1_abs_m: (value) => value.toFixed(4),
  rmse_m: (value) => value.toFixed(4),
  chamfer_m: (value) => value.toFixed(4),
  f1_percent: (value) => value.toFixed(2),
  points_before_downsampling: (value) => value.toLocaleString("en-US"),
  runtime_min: (value) => value.toLocaleString("en-US"),
};

let results = [];
let selectedDay = "Day 1";

function parseResults(csvText) {
  const [headerLine, ...dataLines] = csvText.trim().split(/\r?\n/);
  const headers = headerLine.split(",");

  return dataLines.filter(Boolean).map((line) => {
    const values = line.split(",");
    const row = Object.fromEntries(headers.map((header, index) => [header, values[index]]));

    for (const column of numericColumns) {
      row[column] = Number(row[column]);
    }

    return row;
  });
}

function bestMetricsByScene(rows) {
  const bestByScene = new Map();

  for (const row of rows) {
    if (!bestByScene.has(row.scene)) {
      bestByScene.set(row.scene, {});
    }

    const sceneBest = bestByScene.get(row.scene);
    for (const [metric, direction] of Object.entries(metricDirections)) {
      if (!(metric in sceneBest)) {
        sceneBest[metric] = row[metric];
        continue;
      }

      sceneBest[metric] = direction === "max"
        ? Math.max(sceneBest[metric], row[metric])
        : Math.min(sceneBest[metric], row[metric]);
    }
  }

  return bestByScene;
}

function metricCell(row, metric, bestByScene) {
  const cell = document.createElement("td");
  cell.textContent = formatters[metric](row[metric]);

  const sceneBest = bestByScene.get(row.scene);
  if (sceneBest && metric in metricDirections && row[metric] === sceneBest[metric]) {
    cell.classList.add("best");
  }

  return cell;
}

function renderResults() {
  const body = document.querySelector("#results-body");
  const visibleResults = selectedDay === "All"
    ? results
    : results.filter((row) => row.day === selectedDay);
  const bestByScene = bestMetricsByScene(results);
  const fragment = document.createDocumentFragment();
  let previousScene = null;

  for (const row of visibleResults) {
    const tableRow = document.createElement("tr");
    if (row.scene !== previousScene) {
      tableRow.classList.add("scene-start");
      previousScene = row.scene;
    }

    const sceneCell = document.createElement("td");
    sceneCell.textContent = row.scene;
    tableRow.append(sceneCell);

    const methodCell = document.createElement("td");
    methodCell.textContent = row.method;
    tableRow.append(methodCell);

    for (const metric of numericColumns) {
      tableRow.append(metricCell(row, metric, bestByScene));
    }

    fragment.append(tableRow);
  }

  body.replaceChildren(fragment);
}

async function loadResults() {
  const body = document.querySelector("#results-body");

  try {
    const response = await fetch(RESULTS_PATH);
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    results = parseResults(await response.text());
    renderResults();
  } catch (error) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 9;
    cell.className = "error-cell";
    cell.textContent = "Results could not be loaded. Download the CSV directly from the link above.";
    row.append(cell);
    body.replaceChildren(row);
    console.error(error);
  }
}

document.querySelectorAll("[data-day]").forEach((button) => {
  button.addEventListener("click", () => {
    selectedDay = button.dataset.day;
    document.querySelectorAll("[data-day]").forEach((candidate) => {
      candidate.setAttribute("aria-pressed", String(candidate === button));
    });
    renderResults();
  });
});

document.querySelector("#copy-citation").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const citation = document.querySelector("#citation code").textContent;

  try {
    await navigator.clipboard.writeText(citation);
    button.textContent = "Copied";
  } catch {
    button.textContent = "Select BibTeX below";
  }

  window.setTimeout(() => {
    button.textContent = "Copy BibTeX";
  }, 1800);
});

loadResults();