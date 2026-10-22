const bundled = window.EUROPA_REASONING_SITE_DATA || {};
const state = {
  catalog: bundled.catalog || null,
  results: bundled.results?.runs || [],
  detail: { runIndex: 0, language: "all", domain: "all", filter: "all", itemId: null },
};

const percent = value => value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
const shortModel = value => String(value || "").split("/").at(-1);
const LANGUAGE_FLAGS = Object.freeze({
  bul: "🇧🇬", ces: "🇨🇿", dan: "🇩🇰", deu: "🇩🇪", ell: "🇬🇷", eng: "🇬🇧",
  est: "🇪🇪", fin: "🇫🇮", fra: "🇫🇷", gle: "🇮🇪", hrv: "🇭🇷", hun: "🇭🇺",
  ita: "🇮🇹", lav: "🇱🇻", lit: "🇱🇹", mlt: "🇲🇹", nld: "🇳🇱", pol: "🇵🇱",
  por: "🇵🇹", ron: "🇷🇴", slk: "🇸🇰", slv: "🇸🇮", spa: "🇪🇸", swe: "🇸🇪",
  cat: "🇦🇩", eus: "🇪🇸", glg: "🇪🇸", bos: "🇧🇦", kat: "🇬🇪", mkd: "🇲🇰",
  sqi: "🇦🇱", srp: "🇷🇸", tur: "🇹🇷", ukr: "🇺🇦", isl: "🇮🇸", nor: "🇳🇴",
});
const languageFlag = code => LANGUAGE_FLAGS[code] || "🌐";

async function loadData() {
  if (state.catalog) {
    renderCatalog();
    renderResults();
    return;
  }
  try {
    const [catalogResponse, resultsResponse] = await Promise.all([
      fetch("data/catalog.json"),
      fetch("data/results.json"),
    ]);
    if (!catalogResponse.ok || !resultsResponse.ok) throw new Error("Project data unavailable");
    state.catalog = await catalogResponse.json();
    state.results = (await resultsResponse.json()).runs || [];
    renderCatalog();
    renderResults();
  } catch (error) {
    document.querySelector("#domain-grid").innerHTML = `<article class="domain-card"><span>!</span><h3>Reasoning data could not be loaded.</h3><p>${escapeHtml(error.message)}</p></article>`;
  }
}

function applyDeepLink() {
  const params = new URLSearchParams(window.location.search);
  const runId = params.get("run");
  const itemId = params.get("item");
  const language = params.get("language");
  const runIndex = state.results.findIndex(run => run.run_id === runId);
  if (runIndex >= 0) {
    state.detail.runIndex = runIndex;
    state.detail.itemId = itemId || null;
    state.detail.language = state.catalog?.suite?.languages?.some(candidate => candidate.code === language) ? language : "all";
    state.detail.domain = "all";
    state.detail.filter = "all";
  }
}

function syncDeepLink(runId, itemId) {
  const url = new URL(window.location.href);
  url.search = "";
  url.searchParams.set("run", runId);
  url.searchParams.set("item", itemId);
  if (state.detail.language !== "all") url.searchParams.set("language", state.detail.language);
  url.hash = "deep-dive";
  window.history.replaceState(null, "", url);
}

function renderCatalog() {
  const suite = state.catalog.suite;
  document.querySelector('[data-stat="items"]').textContent = suite.items;
  document.querySelector('[data-stat="languages"]').textContent = suite.languages.length;
  document.querySelector('[data-stat="templates"]').textContent = suite.templates.length;
  document.querySelector('[data-stat="domains"]').textContent = suite.domains.length;

  document.querySelector("#language-grid").innerHTML = suite.languages.map(language => `
    <article class="language-card">
      <header><span><i class="language-flag" aria-hidden="true">${languageFlag(language.code)}</i>${escapeHtml(language.code)}</span><span>${escapeHtml(language.scripts.join(" + "))}</span></header>
      <h3 lang="${escapeAttribute(language.bcp47 || language.code)}">${escapeHtml(language.autonym)}</h3>
      <p>${escapeHtml(language.name)}</p>
      <small>${language.item_count} items · ${escapeHtml(reviewLabel(language.review_states))}</small>
    </article>
  `).join("");

  document.querySelector("#domain-grid").innerHTML = suite.domains.map((domain, index) => `
    <article class="domain-card">
      <span>${String(index + 1).padStart(2, "0")}</span>
      <h3>${escapeHtml(domain.name)}</h3>
      <p>${escapeHtml(domain.description)}</p>
      <small>${domain.item_count} items · ${domain.capabilities.length} ${domain.capabilities.length === 1 ? "skill" : "skills"}</small>
    </article>
  `).join("");

  document.querySelector("#task-types").innerHTML = suite.task_types
    .map(task => `<span>${escapeHtml(task.name)}</span>`)
    .join("");

  document.querySelector("#integration-list").innerHTML = state.catalog.integrations.map((item, index) => `
    <a href="${escapeAttribute(item.url)}" target="_blank" rel="noopener noreferrer">
      <span>${String(index + 1).padStart(2, "0")}</span>
      <strong>${escapeHtml(item.name)}</strong>
      <p>${escapeHtml(item.role)}</p>
      <i aria-hidden="true">↗</i>
    </a>
  `).join("");
}

function renderResults() {
  if (!state.results.length) return;
  state.results.sort((left, right) =>
    Number(right.summary?.overall?.score || 0) - Number(left.summary?.overall?.score || 0));
  applyDeepLink();
  const view = document.querySelector("#results-view");
  view.className = "result-cards";
  view.innerHTML = state.results.map((run, index) => {
    const languageProfile = run.summary.language_profile || {};
    const malformed = Number(run.summary.counts?.malformed || 0);
    const diagnostics = run.summary.output_diagnostics || {};
    const criticalFailures = Number(run.summary.critical_gates?.failure_count || 0);
    const promptEchoes = Number(diagnostics.prompt_echo_count || 0);
    const repeatedSpans = Number(diagnostics.repeated_span_count || 0);
    const domains = (run.summary.domains || []).map(domain => `
      <div class="domain-bar">
        <span>${escapeHtml(label(domain.id))}</span>
        <i><span style="width:${Math.max(0, Math.min(100, Number(domain.score) * 100))}%"></span></i>
        <b>${percent(domain.score)}</b>
      </div>
    `).join("");
    const languageScores = new Map((run.summary.languages || []).map(language => [language.id, language]));
    const languages = (state.catalog?.suite?.languages || []).map(language => {
      const result = languageScores.get(language.code);
      return `<button class="language-score-row" type="button" data-run-language="${escapeAttribute(language.code)}" data-run-index="${index}" title="Inspect ${escapeAttribute(language.name)} answers">
        <span><i class="language-flag" aria-hidden="true">${languageFlag(language.code)}</i><b>${escapeHtml(language.autonym)}</b><small>${escapeHtml(language.code.toUpperCase())}</small></span>
        <strong>${percent(result?.score)}</strong>
      </button>`;
    }).join("");
    return `<article class="result-card">
      <header><h3>${escapeHtml(shortModel(run.model.id))}</h3><span>${escapeHtml(shortRevision(run.model.revision))}</span></header>
      <div class="protocol-line"><span>TARGET PROTOCOL</span><strong>${escapeHtml(reasoningLabel(run))}</strong><small>${escapeHtml(reasoningAllowance(run))}</small></div>
      <div class="judge-line"><span>JUDGE</span><strong>${escapeHtml(run.judge ? shortModel(run.judge.id) : "No LLM judge")}</strong><small>${escapeHtml(run.judge?.revision ? shortRevision(run.judge.revision) : "deterministic only")}</small></div>
      <div class="result-main">
        <div><strong>${percent(languageProfile.macro_language_score)}</strong><span>macro accuracy</span></div>
        <div><strong>${percent(languageProfile.minimum_language_score)}</strong><span>weakest language</span></div>
      </div>
      ${criticalFailures ? `<div class="format-alert critical-alert"><strong>${criticalFailures} critical gate failures</strong><span>correctness-critical dimensions fell below their declared minimum</span></div>` : ""}
      ${processBlock(run)}
      ${malformed ? `<div class="format-alert"><strong>${malformed} malformed</strong><span>responses had no \\boxed{} answer or violated a declared format</span></div>` : ""}
      ${promptEchoes || repeatedSpans ? `<div class="format-alert"><strong>${promptEchoes + repeatedSpans} output anomalies</strong><span>${promptEchoes} prompt echoes · ${repeatedSpans} repeated spans</span></div>` : ""}
      <div class="result-score-toggle" role="group" aria-label="Score breakdown for ${escapeAttribute(shortModel(run.model.id))}">
        <button type="button" data-score-view="domains" aria-pressed="true">Domains</button>
        <button type="button" data-score-view="languages" aria-pressed="false">Languages <span>${state.catalog?.suite?.languages?.length || 0}</span></button>
      </div>
      <div class="result-score-panel result-domains" data-score-panel="domains">${domains}</div>
      <div class="result-score-panel result-languages" data-score-panel="languages" hidden>${languages}</div>
      <button class="inspect-button" type="button" data-run-index="${index}">Inspect ${run.items?.length || 0} answers <span aria-hidden="true">→</span></button>
    </article>`;
  }).join("");
  setupDeepDive();
  setupResultCardToggles();
}

function setupResultCardToggles() {
  document.querySelectorAll(".result-card").forEach(card => {
    card.querySelectorAll("[data-score-view]").forEach(button => {
      button.addEventListener("click", () => {
        const view = button.dataset.scoreView;
        card.querySelectorAll("[data-score-view]").forEach(candidate => candidate.setAttribute("aria-pressed", String(candidate === button)));
        card.querySelectorAll("[data-score-panel]").forEach(panel => { panel.hidden = panel.dataset.scorePanel !== view; });
      });
    });
  });
  document.querySelectorAll("[data-run-language]").forEach(button => {
    button.addEventListener("click", () => {
      state.detail.runIndex = Number(button.dataset.runIndex);
      state.detail.language = button.dataset.runLanguage;
      state.detail.domain = "all";
      state.detail.filter = "all";
      state.detail.itemId = null;
      document.querySelector("#detail-model").value = String(state.detail.runIndex);
      document.querySelector("#detail-language").value = state.detail.language;
      document.querySelector("#detail-domain").value = "all";
      document.querySelector("#detail-filter").value = "all";
      renderDeepDive();
      document.querySelector("#deep-dive").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

function setupDeepDive() {
  const deepDive = document.querySelector("#deep-dive");
  const modelSelect = document.querySelector("#detail-model");
  const languageSelect = document.querySelector("#detail-language");
  const domainSelect = document.querySelector("#detail-domain");
  const filterSelect = document.querySelector("#detail-filter");
  deepDive.hidden = false;
  modelSelect.innerHTML = state.results.map((run, index) =>
    `<option value="${index}">${escapeHtml(runLabel(run))}</option>`).join("");
  domainSelect.innerHTML = `<option value="all">All domains</option>${state.catalog.suite.domains.map(domain =>
    `<option value="${escapeAttribute(domain.id)}">${escapeHtml(domain.name)}</option>`).join("")}`;
  languageSelect.innerHTML = `<option value="all">All languages</option>${state.catalog.suite.languages.map(language =>
    `<option value="${escapeAttribute(language.code)}">${languageFlag(language.code)} ${escapeHtml(language.autonym)} · ${escapeHtml(language.name)}</option>`).join("")}`;
  modelSelect.value = String(state.detail.runIndex);
  languageSelect.value = state.detail.language;
  domainSelect.value = state.detail.domain;
  filterSelect.value = state.detail.filter;
  modelSelect.onchange = event => {
    state.detail.runIndex = Number(event.target.value);
    state.detail.itemId = null;
    renderDeepDive();
  };
  languageSelect.onchange = event => {
    state.detail.language = event.target.value;
    state.detail.itemId = null;
    renderDeepDive();
  };
  domainSelect.onchange = event => {
    state.detail.domain = event.target.value;
    state.detail.itemId = null;
    renderDeepDive();
  };
  filterSelect.onchange = event => {
    state.detail.filter = event.target.value;
    state.detail.itemId = null;
    renderDeepDive();
  };
  document.querySelectorAll(".inspect-button[data-run-index]").forEach(button => {
    button.addEventListener("click", () => {
      state.detail.runIndex = Number(button.dataset.runIndex);
      state.detail.domain = "all";
      state.detail.language = "all";
      state.detail.filter = "all";
      state.detail.itemId = null;
      modelSelect.value = String(state.detail.runIndex);
      domainSelect.value = "all";
      languageSelect.value = "all";
      filterSelect.value = "all";
      renderDeepDive();
      deepDive.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
  renderDeepDive({ updateUrl: false });
  const initialAnchor = window.location.hash.slice(1);
  const anchorTarget = initialAnchor ? document.getElementById(initialAnchor) : null;
  if (anchorTarget) {
    window.requestAnimationFrame(() => anchorTarget.scrollIntoView({ behavior: "auto", block: "start" }));
  }
}

function renderDeepDive({ updateUrl = true } = {}) {
  const run = state.results[state.detail.runIndex];
  if (!run) return;
  const outputDiagnostics = run.summary?.output_diagnostics || {};
  const promptEchoIds = new Set(outputDiagnostics.prompt_echo_item_ids || []);
  const repeatedSpanIds = new Set(outputDiagnostics.repeated_span_item_ids || []);
  const judge = run.judge;
  const backendSettings = run.protocol?.backend_settings || {};
  const think = backendSettings.think;
  const protocolMode = think === false
    ? "Reasoning off"
    : think === true
      ? "Reasoning on"
      : backendSettings.resolved_device
        ? `Transformers · ${String(backendSettings.resolved_device).toUpperCase()}`
        : "Provider default";
  renderLanguageSummary(run);
  document.querySelector("#deep-protocol").innerHTML = `
    <div><span>TARGET MODEL</span><strong>${escapeHtml(run.model.id)}</strong><code title="${escapeAttribute(run.model.revision || "")}">${escapeHtml(shortRevision(run.model.revision))}</code></div>
    <div><span>LLM JUDGE</span><strong>${escapeHtml(judge?.id || "Not configured")}</strong><code title="${escapeAttribute(judge?.revision || "")}">${escapeHtml(judge?.revision ? shortRevision(judge.revision) : "—")}</code></div>
    <div><span>SUITE</span><strong>${escapeHtml(run.suite.id)} v${escapeHtml(run.suite.version)}</strong><code>${run.items?.length || 0} items</code></div>
    <div><span>TARGET PROTOCOL</span><strong>${escapeHtml(protocolMode)}</strong><code>${escapeHtml(reasoningAllowance(run))} · temp ${escapeHtml(run.protocol?.temperature ?? "—")}</code></div>`;
  const caveats = Array.isArray(run.limitations) ? run.limitations : [];
  const rescoring = run.rescoring_history?.at(-1);
  const judging = run.judging_history?.at(-1);
  const extension = run.extension_history?.at(-1);
  const rescoringHtml = rescoring ? `<div class="rescore-note"><span>SCORING REVISION</span><p><strong>v${escapeHtml(rescoring.source_suite_version)} → v${escapeHtml(rescoring.target_suite_version)}</strong> ${escapeHtml(rescoring.reason)}</p><small>Saved model responses and judge outputs were reused; no inference was repeated.</small></div>` : "";
  const judgingHtml = judging ? `<div class="rescore-note"><span>OFFLINE JUDGE PASS</span><p><strong>${escapeHtml(judging.judge?.id || judge?.id || "Named judge")}</strong> scored ${(judging.judged_item_ids || []).length} preserved open answers.</p><small>Target responses were reused unchanged; target generation was not repeated.</small></div>` : "";
  const extensionHtml = extension ? `<div class="rescore-note"><span>SUITE EXTENSION</span><p><strong>${(extension.preserved_item_ids || []).length} preserved + ${(extension.generated_item_ids || []).length} new</strong> responses were merged under v${escapeHtml(extension.current_suite_version)}.</p><small>Model, revision, backend, prompt and decoding settings passed the protocol-equivalence check; later scoring revisions are listed below.</small></div>` : "";
  const caveatHtml = caveats.length ? `<details><summary>Run limitations and comparability (${caveats.length})</summary><ul>${caveats.map(caveat => `<li>${escapeHtml(caveat)}</li>`).join("")}</ul></details>` : "";
  document.querySelector("#deep-caveats").innerHTML = extensionHtml + rescoringHtml + judgingHtml + caveatHtml;

  const entries = (run.items || []).filter(entry => {
    if (state.detail.language !== "all" && entry.item.language !== state.detail.language) return false;
    if (state.detail.domain !== "all" && entry.item.domain !== state.detail.domain) return false;
    if (state.detail.filter === "attention") return entry.sample.error || entry.sample.score == null || entry.sample.score < 1 || !entry.sample.passed || promptEchoIds.has(entry.item.id) || repeatedSpanIds.has(entry.item.id);
    if (state.detail.filter === "judged") return Boolean(entry.sample.judgment);
    if (state.detail.filter === "deterministic") return !entry.sample.judgment;
    return true;
  });
  if (!entries.some(entry => entry.item.id === state.detail.itemId)) {
    state.detail.itemId = entries[0]?.item.id || null;
  }
  const itemList = document.querySelector("#detail-items");
  if (!entries.length) {
    itemList.innerHTML = `<p class="no-items">No answers match these filters.</p>`;
    document.querySelector("#detail-answer").innerHTML = `<p class="no-items">Choose another filter to inspect evidence.</p>`;
    return;
  }
  itemList.innerHTML = entries.map((entry, index) => {
    const selected = entry.item.id === state.detail.itemId;
    const status = entry.sample.score == null ? "missing" : entry.sample.score >= 1 ? "perfect" : "partial";
    return `<button type="button" role="listitem" class="item-row ${selected ? "selected" : ""}" data-item-id="${escapeAttribute(entry.item.id)}" aria-pressed="${selected}">
      <span>${String(index + 1).padStart(2, "0")}</span>
      <div><strong>${escapeHtml(label(entry.item.capability))}</strong><small>${escapeHtml(languageName(entry.item.language))} · ${escapeHtml(label(entry.item.task_type))}</small></div>
      <b class="item-score ${status}">${percent(entry.sample.score)}</b>
    </button>`;
  }).join("");
  itemList.querySelectorAll("[data-item-id]").forEach(button => {
    button.addEventListener("click", () => {
      state.detail.itemId = button.dataset.itemId;
      renderDeepDive();
    });
  });
  const selected = entries.find(entry => entry.item.id === state.detail.itemId) || entries[0];
  if (updateUrl) syncDeepLink(run.run_id, selected.item.id);
  renderAnswer(run, selected);
}

function renderLanguageSummary(run) {
  const summary = document.querySelector("#deep-language-summary");
  const languageProfile = run.summary?.language_profile || {};
  if (state.detail.language === "all") {
    summary.innerHTML = `<div><span>MULTILINGUAL ACCURACY</span><strong>${percent(languageProfile.macro_language_score)}</strong><small>macro average across ${(run.summary?.languages || []).length} languages</small></div>
      <div><span>WEAKEST LANGUAGE</span><strong>${percent(languageProfile.minimum_language_score)}</strong><small>select a language to inspect its evidence</small></div>`;
    return;
  }
  const catalogLanguage = state.catalog?.suite?.languages?.find(candidate => candidate.code === state.detail.language);
  const languageResult = (run.summary?.languages || []).find(candidate => candidate.id === state.detail.language);
  const interval = Array.isArray(languageResult?.ci95) ? `${percent(languageResult.ci95[0])}–${percent(languageResult.ci95[1])}` : "not available";
  summary.innerHTML = `<div class="selected-language-name"><span>SELECTED LANGUAGE</span><strong><i class="language-flag" aria-hidden="true">${languageFlag(state.detail.language)}</i>${escapeHtml(catalogLanguage?.autonym || state.detail.language.toUpperCase())}</strong><small>${escapeHtml(catalogLanguage?.name || state.detail.language.toUpperCase())} · ${escapeHtml(state.detail.language.toUpperCase())}</small></div>
    <div><span>LANGUAGE ACCURACY</span><strong>${percent(languageResult?.score)}</strong><small>${languageResult?.n || 0} scored items · 95% CI ${escapeHtml(interval)}</small></div>
    <div><span>PASS RATE</span><strong>${percent(languageResult?.pass_rate)}</strong><small>items meeting their declared threshold</small></div>`;
}

function renderAnswer(run, entry) {
  const { item, sample } = entry;
  const outputDiagnostics = run.summary?.output_diagnostics || {};
  const promptEcho = (outputDiagnostics.prompt_echo_item_ids || []).includes(item.id);
  const repeatedSpan = (outputDiagnostics.repeated_span_item_ids || []).includes(item.id);
  const malformed = sample.score_details?.malformed === true;
  const partialFormatCredit = malformed && Number(sample.score) > 0;
  const reasoningEnabled = run.protocol?.backend_settings?.think === true;
  const reasoningChars = Number(sample.generation_metadata?.reasoning_char_count || 0);
  const criticalRequirements = item.rubric?.critical_dimensions || {};
  const criticalFailures = sample.score_details?.critical_failures || {};
  const dimensions = sample.judgment && sample.parsed && typeof sample.parsed === "object"
    ? Object.entries(sample.parsed).filter(([, value]) => Number.isFinite(Number(value)))
    : [];
  const dimensionHtml = dimensions.length ? `<div class="dimension-grid">${dimensions.map(([name, value]) => `
    <div class="${criticalRequirements[name] != null ? "critical-dimension" : ""}"><span>${escapeHtml(label(name))}${criticalRequirements[name] != null ? ` <em>critical ≥${escapeHtml(criticalRequirements[name])}</em>` : ""}</span><i><span style="width:${Math.max(0, Math.min(100, Number(value) * 25))}%"></span></i><b>${escapeHtml(value)}/4</b></div>`).join("")}</div>` : "";
  const conversation = item.turns?.length ? `<div class="conversation-history"><span>PRIOR CONVERSATION</span>${item.turns.map(turn => `
    <div class="conversation-turn ${escapeAttribute(turn.role)}"><b>${escapeHtml(turn.role)}</b><p>${escapeHtml(turn.content)}</p></div>`).join("")}</div>` : "";
  const context = item.context ? `<div class="question-context"><span>CONTEXT</span><p>${escapeHtml(item.context)}</p></div>` : "";
  const responseConstraintViolations = sample.score_details?.response_constraint_violations || [];
  const responseConstraintHtml = responseConstraintViolations.length
    ? `<p class="malformed-warning"><strong>Deterministic rubric constraint failed.</strong> ${escapeHtml(responseConstraintViolations.map(label).join(", "))}. The judge's ${percent(sample.score_details?.judge_score_before_constraints)} score was capped at ${percent(sample.score_details?.response_constraint_score_cap)} and the item did not pass.</p>`
    : "";
  const criticalGateHtml = Object.keys(criticalFailures).length
    ? `<p class="critical-gate-warning"><strong>Critical gate applied.</strong> ${Object.entries(criticalFailures).map(([name, evidence]) => `${escapeHtml(label(name))} scored ${escapeHtml(evidence.score)}/4; required ≥${escapeHtml(evidence.required)}/4`).join(" · ")}. The raw dimension average ${percent(sample.score_details?.judge_score_before_critical_gate)} was gated to <strong>${percent(sample.score)}</strong>.</p>`
    : "";
  const options = item.options?.length ? `<ul class="answer-options">${item.options.map(option => `<li>${escapeHtml(option)}</li>`).join("")}</ul>` : "";
  const rationale = sample.score_details?.reason;
  const rubric = item.rubric;
  const rubricHtml = rubric ? `<details class="rubric-details"><summary>Rubric and reference answer</summary><div>
    <p><strong>Pass threshold</strong> ${percent(rubric.pass_threshold)}</p>
    <p><strong>Critical gates</strong><br>${Object.keys(rubric.critical_dimensions || {}).length ? Object.entries(rubric.critical_dimensions).map(([name, minimum]) => `${escapeHtml(label(name))} ≥ ${escapeHtml(minimum)}/4`).join(" · ") : "None"}</p>
    <p><strong>Reference answer</strong><br>${escapeHtml(rubric.reference_answer || "No reference answer supplied.")}</p>
    <p><strong>Required points</strong></p><ul>${(rubric.required_points || []).map(point => `<li>${escapeHtml(point)}</li>`).join("")}</ul>
    <p><strong>Scoring dimensions</strong></p><dl>${Object.entries(rubric.dimensions || {}).map(([name, description]) => `<dt>${escapeHtml(label(name))}</dt><dd>${escapeHtml(description)}</dd>`).join("")}</dl>
  </div></details>` : "";
  const numericDetails = sample.scorer === "numeric" ? sample.score_details : null;
  const numericExplanation = numericDetails?.relative_error_percent != null
    ? `<p><strong>Numeric error.</strong> Expected ${escapeHtml(numericDetails.expected)}, parsed ${escapeHtml(sample.parsed)}: absolute error ${escapeHtml(formatMetric(numericDetails.absolute_error))}, relative error ${escapeHtml(formatMetric(numericDetails.relative_error_percent))}%. ${numericDetails.partial_credit_method === "relative_error" ? `The item score is 100% minus relative error, clamped to 0–100%: <strong>${percent(sample.score)}</strong>.` : "This item does not declare error-based partial credit."}</p>`
    : "";
  const deterministicExplanation = malformed
    ? partialFormatCredit
      ? `<p class="malformed-warning"><strong>Malformed format.</strong> The response violated the exact answer contract, but an unambiguous correct leading choice was detected, so ${percent(sample.score)} partial credit was awarded. It did not pass.</p>`
      : `<p class="malformed-warning"><strong>Malformed format.</strong> The response did not match the declared answer contract, so no credit was awarded${sample.parsed ? ` (detected choice: ${escapeHtml(sample.parsed)})` : ""}.</p>`
    : `${numericExplanation}<p>No LLM judge was used for this item. The answer was checked by the declared scorer.</p>`;
  const judgePanel = sample.judgment ? `
    <section class="judge-rationale">
      <div class="answer-subhead"><span>LLM JUDGE RATIONALE</span><strong>${escapeHtml(sample.judgment.model)}</strong></div>
      ${dimensionHtml}
      ${criticalGateHtml}
      ${responseConstraintHtml}
      <blockquote>${escapeHtml(rationale || "The judge returned scores without a written rationale.")}</blockquote>
      <p class="reasoning-note">Published score rationale supplied by the judge; not hidden chain-of-thought.</p>
      ${rubricHtml}
      <details><summary>Raw judge output</summary><pre>${escapeHtml(sample.judgment.response)}</pre></details>
    </section>` : `
    <section class="deterministic-score">
      <div class="answer-subhead"><span>SCORING DECISION</span><strong>Deterministic · ${escapeHtml(sample.scorer)}</strong></div>
      ${deterministicExplanation}
      <details><summary>Scorer details</summary><pre>${escapeHtml(JSON.stringify(sample.score_details || {}, null, 2))}</pre></details>
    </section>`;
  const outputWarning = promptEcho || repeatedSpan
    ? `<p class="malformed-warning"><strong>Output anomaly.</strong> ${promptEcho ? "The response appears to repeat benchmark prompt text." : ""} ${repeatedSpan ? "A 12-token span repeats non-contiguously." : ""} The raw response is preserved for review.</p>`
    : "";
  const process = sample.process || {};
  const traceChars = Number(process.trace_chars || reasoningChars || 0);
  const reasoningDisclosure = reasoningEnabled
    ? `<p class="reasoning-disclosure"><strong>Reasoning mode.</strong> Only the final answer below was scored; the trace is not published. ${
        process.trace_closed === false
          ? "<strong>The trace never closed, so there is no final answer.</strong> "
          : ""
      }Trace: ${process.trace_closed ? "closed" : "unclosed"}${process.trace_loop ? " · <strong>loop detected</strong>" : ""}${
        sample.reasoning_tokens != null ? ` · ${Number(sample.reasoning_tokens).toLocaleString("en-US")} reasoning tokens` : traceChars ? ` · ${traceChars.toLocaleString("en-US")} characters` : ""
      }${process.trace_script_share != null ? ` · ${percent(process.trace_script_share)} in ${escapeHtml(item.script)} script` : ""}${
        process.trace_language_detected ? ` · detected language ${escapeHtml(process.trace_language_detected)}` : ""
      }.</p>`
    : process.think_tag_in_answer ? `<p class="malformed-warning"><strong>Think tags in a default-mode answer.</strong></p>` : "";
  const tokenLabel = reasoningEnabled ? "generated tokens incl. reasoning" : "output tokens";
  const packProvenance = item.metadata?.translation_generator
    ? `${item.metadata.translation_generator}${item.metadata.translation_model_revision ? ` · ${shortRevision(item.metadata.translation_model_revision)}` : ""}${item.metadata.quality_corrections ? " · post-edited" : ""}`
    : "pack provenance unavailable";
  document.querySelector("#detail-answer").innerHTML = `
    <header class="answer-header">
      <div><span>${escapeHtml(languageName(item.language))} · ${escapeHtml(domainName(item.domain))}</span><h4>${escapeHtml(label(item.capability))}</h4><code>${escapeHtml(item.id)}</code></div>
      <div class="answer-score"><strong>${percent(sample.score)}</strong><span>${Object.keys(criticalFailures).length ? "critical gate failed" : partialFormatCredit ? "malformed · partial credit" : malformed ? "malformed format" : sample.passed ? "passed" : sample.passed === false ? "did not pass" : "unscored"}</span></div>
    </header>
    <section class="question-block">${conversation}${context}<span>${item.turns?.length ? "FINAL TASK" : "TASK"}</span><p>${escapeHtml(item.prompt)}</p>${options}</section>
    <section class="model-answer"><div class="answer-subhead"><span>MODEL ANSWER</span><strong>${escapeHtml(runLabel(run))}</strong></div>${reasoningDisclosure}${outputWarning}<pre>${escapeHtml(sample.response ?? "No response")}</pre></section>
    ${judgePanel}
    <div class="answer-meta"><span>${escapeHtml(item.language.toUpperCase())} · ${escapeHtml(item.script)}</span><span>${escapeHtml(label(item.review_status))}</span><span>pack: ${escapeHtml(packProvenance)}</span><span>${escapeHtml(label(item.task_type))}</span><span>${Math.round(Number(sample.latency_ms || 0))} ms</span><span>${sample.output_tokens ?? "—"} ${escapeHtml(tokenLabel)}</span><a href="${escapeAttribute(item.source.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.source.title)} ↗</a></div>`;
}

function processBlock(run) {
  const process = run.summary?.process?.overall || {};
  if (!process.reasoning_n) {
    const leak = Number(process.think_tag_in_answer_rate || 0);
    return leak ? `<div class="format-alert"><strong>${percent(leak)} think-tag leaks</strong><span>default-mode answers containing reasoning delimiters</span></div>` : "";
  }
  return `<div class="result-main process-main">
        <div><strong>${percent(process.trace_closure_rate)}</strong><span>traces closed</span></div>
        <div><strong>${percent(process.trace_loop_rate)}</strong><span>traces looping</span></div>
        <div><strong>${process.median_reasoning_tokens_closed == null ? "—" : Number(process.median_reasoning_tokens_closed).toLocaleString("en-US")}</strong><span>median reasoning tokens</span></div>
        <div><strong>${percent(process.mean_trace_script_share)}</strong><span>trace in prompt script</span></div>
      </div>`;
}

function reasoningLabel(run) {
  const think = run.protocol?.backend_settings?.think;
  if (think === true) return "Reasoning on";
  if (think === false) return "Reasoning off";
  return "Provider default";
}

function reasoningAllowance(run) {
  const settings = run.protocol?.backend_settings || {};
  if (settings.think !== true) return settings.think === false ? "answer budget only" : "not declared";
  const allowance = Number(settings.reasoning_token_budget);
  return Number.isFinite(allowance) ? `+${allowance.toLocaleString("en-US")} reasoning tokens` : "shared answer budget";
}

function runLabel(run) {
  return `${shortModel(run.model.id)} · ${reasoningLabel(run).toLowerCase()}`;
}

function setupTabs() {
  const buttons = document.querySelectorAll(".runner-tabs button");
  buttons.forEach(button => button.addEventListener("click", () => {
    buttons.forEach(candidate => candidate.setAttribute("aria-selected", String(candidate === button)));
    document.querySelectorAll(".command").forEach(command => command.classList.remove("active"));
    document.querySelector(`#command-${button.dataset.tab}`).classList.add("active");
  }));
  document.querySelector("#copy-command").addEventListener("click", async event => {
    const command = document.querySelector(".command.active").innerText;
    try {
      await navigator.clipboard.writeText(command);
      event.currentTarget.textContent = "Copied";
      window.setTimeout(() => { event.currentTarget.textContent = "Copy"; }, 1300);
    } catch {
      event.currentTarget.textContent = "Select text";
    }
  });
}

function label(value) {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());
}

function domainName(value) {
  return state.catalog?.suite?.domains?.find(domain => domain.id === value)?.name || label(value);
}

function languageName(value) {
  const language = state.catalog?.suite?.languages?.find(candidate => candidate.code === value);
  return language ? `${language.autonym} (${language.code})` : String(value || "").toUpperCase();
}

function reviewLabel(states) {
  const values = Array.isArray(states) ? states : [];
  if (values.includes("native_reviewed")) return "native reviewed";
  if (values.includes("source_native")) return "authored source";
  return "review requested";
}

function shortRevision(value) {
  if (!value) return "revision not supplied";
  const revision = String(value);
  return revision.startsWith("sha256:") ? revision.slice(7, 19) : revision.slice(0, 12);
}

function formatMetric(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return number.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function escapeAttribute(value) {
  return escapeHtml(value);
}

setupTabs();
loadData();
