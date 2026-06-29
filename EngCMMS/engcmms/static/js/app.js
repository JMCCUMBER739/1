// EngCMMS front-end helpers (vanilla JS, no external dependencies).

// Shared Chart.js defaults for a consistent look.
if (window.Chart) {
  Chart.defaults.font.family =
    "'Segoe UI', system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif";
  Chart.defaults.color = "#6b7a90";
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
  Chart.defaults.maintainAspectRatio = false;
}

const PALETTE = [
  "#2563eb", "#16a34a", "#d97706", "#dc2626", "#0891b2",
  "#7c3aed", "#db2777", "#65a30d", "#0d9488", "#9333ea",
];

// Render every <canvas data-chart="..."> that carries a JSON spec.
document.querySelectorAll("canvas[data-chart]").forEach((el) => {
  let spec;
  try {
    spec = JSON.parse(el.getAttribute("data-chart"));
  } catch (e) {
    return;
  }
  const type = spec.type || "bar";
  const datasets = (spec.datasets || []).map((ds, i) => ({
    label: ds.label || "",
    data: ds.data || [],
    backgroundColor:
      type === "line"
        ? "rgba(37,99,235,.12)"
        : ds.colors || PALETTE[i % PALETTE.length],
    borderColor: ds.color || PALETTE[i % PALETTE.length],
    borderWidth: type === "line" ? 2 : 0,
    fill: type === "line" ? !!ds.fill : undefined,
    tension: 0.35,
    pointRadius: type === "line" ? 3 : undefined,
  }));

  if ((type === "doughnut" || type === "pie") && datasets[0]) {
    datasets[0].backgroundColor = PALETTE;
    datasets[0].borderColor = "#fff";
    datasets[0].borderWidth = 2;
  }

  new Chart(el, {
    type,
    data: { labels: spec.labels || [], datasets },
    options: {
      plugins: {
        legend: {
          display: spec.legend !== false,
          position: type === "doughnut" || type === "pie" ? "right" : "top",
        },
      },
      scales:
        type === "doughnut" || type === "pie"
          ? {}
          : {
              x: { stacked: !!spec.stacked, grid: { display: false } },
              y: {
                stacked: !!spec.stacked,
                beginAtZero: true,
                grid: { color: "#eef2f8" },
              },
            },
    },
  });
});

// Auto-suggest an email address from a contact/user name field.
document.querySelectorAll("[data-email-source]").forEach((nameInput) => {
  const targetSel = nameInput.getAttribute("data-email-source");
  const target = document.querySelector(targetSel);
  if (!target) return;
  nameInput.addEventListener("blur", () => {
    if (target.value.trim() !== "" || nameInput.value.trim() === "") return;
    fetch(`/contacts/suggest-email?name=${encodeURIComponent(nameInput.value)}`)
      .then((r) => r.json())
      .then((d) => {
        if (d.email) target.value = d.email;
      })
      .catch(() => {});
  });
});

// Load an email template's subject/body into the composer.
const tplSelect = document.getElementById("template-picker");
if (tplSelect) {
  tplSelect.addEventListener("change", () => {
    const id = tplSelect.value;
    if (!id) return;
    fetch(`/emails/template/${id}.json`)
      .then((r) => r.json())
      .then((d) => {
        const subj = document.querySelector("[name=subject]");
        const body = document.querySelector("[name=body]");
        const cat = document.querySelector("[name=category]");
        if (subj) subj.value = d.subject;
        if (body) body.value = d.body;
        if (cat && d.category) cat.value = d.category;
      })
      .catch(() => {});
  });
}

// Generic confirm guard for destructive actions.
document.querySelectorAll("form[data-confirm]").forEach((form) => {
  form.addEventListener("submit", (e) => {
    if (!window.confirm(form.getAttribute("data-confirm"))) e.preventDefault();
  });
});
