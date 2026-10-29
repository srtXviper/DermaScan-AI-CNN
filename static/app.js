const imageInput = document.getElementById("skin-image");
const dropZone = document.getElementById("drop-zone");
const preview = document.getElementById("image-preview");
const placeholder = document.getElementById("upload-placeholder");
const imageMeta = document.getElementById("image-meta");
const imageStatus = document.getElementById("image-status");
const modelResults = document.getElementById("model-results");
const checkImageBtn = document.getElementById("check-image-btn");
let selectedFile = null;

function setStatus(message, error = false) {
  imageStatus.textContent = message;
  imageStatus.classList.toggle("error", error);
}

function clearModelResults() {
  modelResults.replaceChildren();
  modelResults.hidden = true;
  modelResults.style.display = "none";
}

function addText(parent, tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = text;
  parent.appendChild(node);
  return node;
}

function selectImage(file) {
  if (!file) return;
  selectedFile = file;
  setStatus("");
  clearModelResults();
  preview.hidden = true;
  imageMeta.hidden = true;
  placeholder.hidden = false;

  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    selectedFile = null;
    setStatus("Unsupported file. Choose a JPG, PNG, or WEBP image.", true);
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    selectedFile = null;
    setStatus("This file is larger than 8 MB. Choose a smaller image.", true);
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.hidden = false;
    placeholder.hidden = true;
    imageMeta.hidden = false;
    imageMeta.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  };
  reader.onerror = () => setStatus("Could not preview this image. Please choose it again.", true);
  reader.readAsDataURL(file);
}

imageInput.addEventListener("change", event => selectImage(event.target.files[0]));
["dragenter", "dragover"].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault();
  dropZone.classList.add("dragover");
}));
["dragleave", "drop"].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
}));
dropZone.addEventListener("drop", event => selectImage(event.dataTransfer.files[0]));

function renderPredictions(data) {
  clearModelResults();
  modelResults.hidden = false;
  // Inline display makes the result visible even if an older stylesheet hides this component.
  modelResults.style.display = "block";

  addText(modelResults, "h4", "", "Experimental AI image results");
  addText(
    modelResults,
    "p",
    "model-note",
    `Model: ${data.model_id || "image classifier"}. Scores are model outputs, not calibrated probabilities and not diagnostic certainty.`
  );

  const predictions = Array.isArray(data.predictions) ? data.predictions : [];
  if (!predictions.length) {
    addText(
      modelResults,
      "p",
      "model-disclaimer",
      "The server accepted the image but did not return prediction labels. Check that the latest model-enabled app.py is running, then restart the server."
    );
    addText(modelResults, "p", "model-note", `Response mode: ${data.mode || "not supplied"}.`);
    return;
  }

  const list = document.createElement("div");
  list.className = "prediction-list";
  predictions.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "prediction-row";
    const label = document.createElement("strong");
    label.textContent = `${index + 1}. ${item.label || "Unlabelled class"}`;
    const score = document.createElement("span");
    const raw = Number(item.score);
    const percent = Number.isFinite(raw) ? (raw >= 0 && raw <= 1 ? raw * 100 : raw) : null;
    score.textContent = percent === null ? "Score unavailable" : `${percent.toFixed(1)}% model score`;
    row.append(label, score);
    list.appendChild(row);

    if (item.label) {
      const link = document.createElement("a");
      link.href = `https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(item.label + " skin condition")}`;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.className = "reference-link";
      link.textContent = `Find scientific articles about “${item.label}” on PubMed`;
      list.appendChild(link);
    }
  });
  modelResults.appendChild(list);

  addText(modelResults, "h4", "", "How to interpret this result");
  addText(
    modelResults,
    "p",
    "model-note",
    "These are image-classifier matches, not a confirmed condition. Skin conditions can look similar, and image quality, skin tone, lighting, and the body area may affect the output. Do not start, stop, or change medication based on these labels."
  );
  addText(
    modelResults,
    "p",
    "model-disclaimer",
    data.disclaimer || "Educational/research use only. This model is not clinically validated. A qualified healthcare professional can assess the concern in context."
  );
}

checkImageBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    setStatus("Choose an image first.", true);
    return;
  }
  const old = checkImageBtn.innerHTML;
  checkImageBtn.disabled = true;
  checkImageBtn.textContent = "Loading model / analyzing…";
  clearModelResults();
  try {
    const form = new FormData();
    form.append("image", selectedFile);
    const response = await fetch("/api/image-check", { method: "POST", body: form });
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error(`Server returned an unreadable response (HTTP ${response.status}). Check the PowerShell window.`);
    }
    if (!response.ok) throw new Error(data.detail || `Could not analyze image (HTTP ${response.status}).`);

    setStatus(`Image processed (${data.width} × ${data.height}). ${data.message || "Analysis response received."}`);
    renderPredictions(data);
    modelResults.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    setStatus(error.message || "Could not reach the server. Is the backend running?", true);
    modelResults.hidden = false;
    modelResults.style.display = "block";
    modelResults.replaceChildren();
    addText(modelResults, "h4", "Analysis could not be completed", "Analysis could not be completed");
    addText(modelResults, "p", "model-disclaimer", error.message || "Check that the backend is running and inspect the PowerShell error output.");
  } finally {
    checkImageBtn.disabled = false;
    checkImageBtn.innerHTML = old;
  }
});

document.getElementById("symptom-form").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = {
    duration: form.duration.value,
    itching: form.itching.checked,
    pain: form.pain.checked,
    changing: form.changing.checked,
    bleeding_or_open_sore: form.bleeding_or_open_sore.checked,
    fever_or_rapid_spread: form.fever_or_rapid_spread.checked,
    notes: form.notes.value
  };
  const button = form.querySelector('button[type="submit"]');
  const old = button.innerHTML;
  button.disabled = true;
  button.textContent = "Preparing summary…";
  try {
    const response = await fetch("/api/symptom-summary", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not generate summary.");
    document.getElementById("result-urgency").textContent = data.urgency_label;
    document.getElementById("result-summary").textContent = data.summary;
    document.getElementById("result-disclaimer").textContent = data.disclaimer;
    const result = document.getElementById("results");
    result.hidden = false;
    document.getElementById("step-number").textContent = "02";
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    alert(error.message || "Could not reach the server. Please try again.");
  } finally {
    button.disabled = false;
    button.innerHTML = old;
  }
});

document.getElementById("reset-btn").addEventListener("click", () => {
  document.getElementById("symptom-form").reset();
  document.getElementById("results").hidden = true;
  document.getElementById("step-number").textContent = "01";
  imageInput.value = "";
  selectedFile = null;
  clearModelResults();
  preview.hidden = true;
  preview.removeAttribute("src");
  placeholder.hidden = false;
  imageMeta.hidden = true;
  setStatus("");
  document.getElementById("screening").scrollIntoView({ behavior: "smooth" });
});
