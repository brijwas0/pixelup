(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const views = { upload: $("view-upload"), working: $("view-working"), result: $("view-result") };
  const ALLOWED = ["image/jpeg", "image/png", "image/webp"];
  const ALLOWED_EXT = /\.(jpe?g|png|webp)$/i;
  let maxMB = 25, speeds = {};

  const state = { file: null, beforeUrl: null, afterUrl: null, jobId: null, poll: null, abort: null, dims: null };

  function show(name) {
    for (const [k, el] of Object.entries(views)) el.hidden = k !== name;
  }
  function showError(msg) {
    const el = $("error");
    el.textContent = msg || "";
    el.hidden = !msg;
  }
  const fmtBytes = (n) => (n >= 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB");

  // ---- server limits (best effort) ----
  fetch("/api/health").then((r) => r.json()).then((h) => {
    if (h.max_upload_mb) { maxMB = h.max_upload_mb; $("maxmb").textContent = Math.round(maxMB); }
    speeds = h.sec_per_mp || {};
    updateEstimate();
  }).catch(() => {});

  // Rough time estimate from the server's measured speed on this computer.
  function updateEstimate() {
    const el = $("estimate");
    if (!state.dims || !speeds.fast) { el.hidden = true; return; }
    const mp = (state.dims[0] * state.dims[1]) / 1e6;
    const q = document.querySelector('input[name="quality"]:checked').value;
    const secs = mp * speeds.fast * 1.15 * (q === "best" ? 6 : 1);
    const txt = secs < 10 ? "a few seconds" : secs < 90 ? `about ${Math.round(secs / 5) * 5} seconds` : `about ${Math.round(secs / 60)} min`;
    el.textContent = `Estimated time: ${txt}`;
    el.hidden = false;
  }
  document.addEventListener("change", (e) => { if (e.target.name === "quality") updateEstimate(); });

  // ---- choosing a file ----
  const drop = $("drop"), input = $("file");
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => { if (input.files[0]) pick(input.files[0]); });
  $("change").addEventListener("click", () => input.click());
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) pick(f); });
  // allow dropping anywhere on the page
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f && !views.upload.hidden) pick(f); });

  function validate(f) {
    const heic = /heic|heif/i.test(f.type) || /\.hei[cf]$/i.test(f.name);
    if (heic) return "HEIC photos aren't supported yet. On iPhone, share as JPG or choose Settings → Camera → Formats → Most Compatible.";
    if (!ALLOWED.includes(f.type) && !ALLOWED_EXT.test(f.name)) return "Please choose a JPG, PNG or WEBP image.";
    if (f.size === 0) return "That file is empty.";
    if (f.size > maxMB * 1048576) return `That file is ${fmtBytes(f.size)}. The maximum is ${Math.round(maxMB)} MB.`;
    return null;
  }

  function pick(f) {
    showError("");
    const problem = validate(f);
    if (problem) { showError(problem); return; }
    if (state.beforeUrl) URL.revokeObjectURL(state.beforeUrl);
    state.file = f;
    state.beforeUrl = URL.createObjectURL(f);
    const probe = new Image();
    probe.onload = () => {
      state.dims = [probe.naturalWidth, probe.naturalHeight];
      $("fdetail").textContent = `${probe.naturalWidth} × ${probe.naturalHeight} · ${fmtBytes(f.size)}`;
      $("go").disabled = false;
      fetch("/api/health").then((r) => r.json()).then((h) => { speeds = h.sec_per_mp || {}; updateEstimate(); }).catch(updateEstimate);
    };
    probe.onerror = () => {
      showError("That image couldn't be read. It may be corrupted or in an unsupported format.");
      resetPick();
    };
    probe.src = state.beforeUrl;
    $("thumb").src = state.beforeUrl;
    $("fname").textContent = f.name;
    $("fdetail").textContent = fmtBytes(f.size);
    drop.hidden = true;
    $("picked").hidden = false;
    $("options").hidden = false;
  }

  function resetPick() {
    if (state.beforeUrl) URL.revokeObjectURL(state.beforeUrl);
    Object.assign(state, { file: null, beforeUrl: null, dims: null });
    input.value = "";
    drop.hidden = false;
    $("picked").hidden = true;
    $("options").hidden = true;
    $("go").disabled = true;
  }

  // ---- enhance ----
  $("go").addEventListener("click", start);

  const radio = (name) => document.querySelector(`input[name="${name}"]:checked`).value;

  async function start() {
    if (!state.file) return;
    showError("");
    const scale = radio("scale"), quality = radio("quality");
    const body = new FormData();
    body.append("file", state.file);
    body.append("scale", scale);
    body.append("quality", quality);
    body.append("grade", $("grade").checked);
    body.append("denoise", $("denoise").checked);

    setWorking("Uploading…", 0.02);
    show("working");
    state.abort = new AbortController();
    try {
      const res = await fetch("/api/enhance", { method: "POST", body, signal: state.abort.signal });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Upload failed. Please try again.");
      state.jobId = data.job_id;
      state.scale = Number(scale);
      pollJob();
    } catch (e) {
      if (e.name === "AbortError") return;
      fail(e instanceof TypeError ? "Couldn't reach the PixelUp server. Is it still running?" : e.message);
    }
  }

  function setWorking(msg, frac) {
    $("wmsg").textContent = msg;
    $("barfill").style.width = Math.round(frac * 100) + "%";
    document.querySelector(".bar").setAttribute("aria-valuenow", Math.round(frac * 100));
  }

  function pollJob() {
    let failures = 0;
    const tick = async () => {
      if (!state.jobId) return;
      try {
        const res = await fetch(`/api/jobs/${state.jobId}`, { cache: "no-store" });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || "Lost track of the job.");
        failures = 0;
        const j = await res.json();
        if (j.status === "error") return fail(j.error || "Enhancement failed.");
        if (j.status === "cancelled") return;
        if (j.status === "done") return finish(j);
        setWorking(j.message, j.progress);
      } catch (e) {
        if (++failures >= 6) return fail(e instanceof TypeError ? "Lost connection to the PixelUp server." : e.message);
      }
      state.poll = setTimeout(tick, 800);
    };
    tick();
  }

  async function finish(info) {
    setWorking("Downloading result…", 0.99);
    try {
      const res = await fetch(`/api/jobs/${state.jobId}/result`, { cache: "no-store" });
      if (!res.ok) throw new Error("Couldn't retrieve the enhanced photo. Please try again.");
      const blob = await res.blob();
      state.jobId = null;                       // the server forgets the photo once delivered
      if (state.afterUrl) URL.revokeObjectURL(state.afterUrl);
      state.afterUrl = URL.createObjectURL(blob);

      const ext = blob.type === "image/png" ? "png" : "jpg";
      const base = state.file.name.replace(/\.[^.]+$/, "") || "photo";
      const dl = $("download");
      dl.href = state.afterUrl;
      dl.download = `${base}-pixelup-${state.scale}x.${ext}`;
      dl.textContent = `Download · ${info.width} × ${info.height} · ${fmtBytes(blob.size)}`;
      $("rinfo").textContent = `${state.dims[0]} × ${state.dims[1]}  →  ${info.width} × ${info.height}`;
      const n = $("rnotice");
      n.textContent = info.notice || "";
      n.hidden = !info.notice;

      await new Promise((ok) => { const i = $("img-after"); i.onload = i.onerror = ok; i.src = state.afterUrl; });
      $("img-before").src = state.beforeUrl;
      $("slider").value = 50;
      $("compare").style.setProperty("--pos", "50%");
      show("result");
    } catch (e) {
      fail(e.message);
    }
  }

  function fail(msg) {
    clearTimeout(state.poll);
    state.jobId = null;
    show("upload");
    showError(msg);
  }

  $("cancel").addEventListener("click", () => {
    clearTimeout(state.poll);
    if (state.abort) state.abort.abort();
    if (state.jobId) fetch(`/api/jobs/${state.jobId}`, { method: "DELETE" }).catch(() => {});
    state.jobId = null;
    show("upload");
  });

  // ---- before / after slider ----
  $("slider").addEventListener("input", (e) => $("compare").style.setProperty("--pos", e.target.value + "%"));

  $("fullsize").addEventListener("click", () => { if (state.afterUrl) window.open(state.afterUrl, "_blank", "noopener"); });
  $("again").addEventListener("click", () => {
    if (state.afterUrl) URL.revokeObjectURL(state.afterUrl);
    state.afterUrl = null;
    $("img-after").removeAttribute("src");
    resetPick();
    showError("");
    show("upload");
  });
})();
