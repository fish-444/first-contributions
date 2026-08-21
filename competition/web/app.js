/* 양돈 운영 콘솔 — 프론트.
 *
 * **계산을 하지 않는다.** 용량·상한·처방·일정은 전부 API 응답을 그리기만
 * 한다. 화면이 자기 산식을 갖는 순간 같은 농장에 대해 두 답이 생긴다 —
 * 이 프로젝트가 이미 겪은 사고다(등록 화면의 분만사 단위, 복당 이유두수
 * 12 vs 11). 유일한 예외는 preset() 인데, 그것도 서버가 내려준 상수만 쓴다.
 */
"use strict";

const $ = s => document.querySelector(s);
const el = (t, c) => { const e = document.createElement(t); if (c) e.className = c; return e; };
const n0 = x => Number(x).toLocaleString(undefined, { maximumFractionDigits: 0 });
const man = won => n0(won / 1e4) + "만원";

let CONST = null;              // /api/health 가 내려주는 상수
let STAGES = [];
let barns = {};                // {용도: {rooms, per}}
let lastCap = null;

async function api(path, opts) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...opts
  });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail ?? msg; } catch (e) { /* 본문 없음 */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.status === 204 ? null : r.json();
}

/* ── 입력 수집 ─────────────────────────────────── */
function num(id) { const v = parseFloat($(id).value); return isFinite(v) ? v : null; }

function setup() {
  const bs = STAGES.filter(s => barns[s] && barns[s].rooms > 0 && barns[s].per > 0)
    .map(s => ({
      name: s, stage: s, rooms: barns[s].rooms, per: barns[s].per,
      housing: s === "분만사" ? "crate" : (s === "교배사" ? "stall"
        : (["자돈사", "육성사", "비육사"].includes(s) ? "pen" : "group"))
    }));
  return {
    name: $("#f-name").value || null,
    n_sows: lastCap && lastCap.capacity.n_sows ? lastCap.capacity.n_sows : null,
    interval_days: +$("#f-iv").value,
    lactation_days: num("#f-lact") ?? 24,
    pre_farrow_days: num("#f-pre") ?? 7,
    washout_days: num("#f-wash") ?? 7,
    barns: bs,
    performance: {
      farrowing_rate: num("#p-fr"), weaned: num("#p-wl"), survival: num("#p-gs")
    }
  };
}

/* ── 기본 구성 ───────────────────────────────── */
// **서버가 짓는다.** 여기서 계산하면 서버의 방 수 보정(extra_rooms)을 모르는
// 채로 구성을 만들게 되고, 넣자마자 "막힘" 이 뜬다 — 실제로 그랬다.
async function preset(sows) {
  const q = new URLSearchParams({
    sows, interval_days: $("#f-iv").value,
    lactation: num("#f-lact") ?? 24, pre_farrow: num("#f-pre") ?? 7,
    washout: num("#f-wash") ?? 7, weaned: num("#p-wl") ?? 11
  });
  const r = await api("/api/capacity/preset?" + q);
  const out = {};
  for (const b of r.barns) out[b.stage] = { rooms: b.rooms, per: b.per };
  return out;
}

/* ── 돈사 표 ───────────────────────────────────── */
function drawBarns() {
  const box = $("#barns");
  box.innerHTML = STAGES.map(st => {
    const b = barns[st] || { rooms: 0, per: 0 };
    const tot = b.rooms * b.per;
    return `<div class="brow${tot ? "" : " off"}">
      <span class="use">${st}</span>
      <input data-st="${st}" data-k="rooms" type="number" inputmode="numeric"
        min="0" max="99" value="${b.rooms || ""}" placeholder="0" aria-label="${st} 방 수">
      <input data-st="${st}" data-k="per" type="number" inputmode="numeric"
        min="0" max="9999" value="${b.per || ""}" placeholder="0" aria-label="${st} 방당 자리">
      <span class="cap">${tot ? n0(tot) : "—"}</span></div>`;
  }).join("");
  const n = STAGES.filter(s => barns[s] && barns[s].rooms && barns[s].per).length;
  $("#h-barns").textContent = n
    ? `${n}개 용도 · 방 ${STAGES.reduce((a, s) => a + ((barns[s] || {}).rooms || 0), 0)}개`
    : "등록된 돈사가 없습니다.";
}

/* ── 그리기 (전부 API 응답) ────────────────────── */
const FLOW = ["교배사", "임신사", "분만사", "자돈사", "육성사", "비육사"];

function drawRail(cap) {
  const rail = $("#rail"), note = $("#railnote");
  const rows = FLOW.map(s => cap.rows.find(r => r.stage === s)).filter(Boolean);
  if (!rows.length) {
    rail.innerHTML = `<p class="empty">돈사를 등록하면 나옵니다.</p>`;
    note.textContent = ""; return;
  }
  const max = Math.max(...rows.map(r => r.sows || 0), 1);
  rail.innerHTML = rows.map(r => {
    const cls = r.why ? "blocked" : (r.stage === cap.binding ? "bind" : "");
    const tag = r.why ? `<span class="pill stop">막힘</span>`
      : (r.stage === cap.binding ? `<span class="pill">병목</span>` : "");
    const w = r.why ? 100 : Math.max(3, (r.sows / max) * 100);
    return `<div class="stage ${cls}"><div class="spine"></div>
      <span class="nm">${r.stage}${tag}</span>
      <div class="track"><div class="bar-fill" style="width:${w}%"></div></div>
      <span class="val">${r.why ? "막힘" : n0(r.sows) + "두"}</span></div>`;
  }).join("");

  if (!cap.flows) {
    note.className = "note bad";
    note.innerHTML = `<b>두수를 말하기 전에 막힌 곳이 있습니다 — ` +
      `${cap.blocked.map(r => `${r.stage}(${r.why})`).join(" · ")}.</b>
      자리가 남아도 회전이 안 되면 배치가 밀립니다.
      <b>두수를 줄여도 안 풀립니다</b> — 방을 늘리거나 간격을 넓혀야 합니다.`;
  } else if (cap.binding) {
    const b = rows.find(r => r.stage === cap.binding);
    const others = rows.filter(r => r.stage !== cap.binding && r.sows > 0);
    const slack = others.length ? Math.min(...others.map(r => r.sows)) - b.sows : 0;
    note.className = "note";
    note.innerHTML = `이 농장의 규모는 <b>${n0(cap.n_sows)}두</b>이고 붙잡고 있는 건
      <b>${cap.binding}</b>입니다.` +
      (slack > 0 ? ` 다음으로 좁은 칸이 ${n0(slack)}두 더 받으므로,
        <b>${cap.binding}를 넓히기 전까지는 다른 돈사를 키워도 두수가 안 늘어납니다.</b>` : "");
  } else { note.textContent = ""; }
}

function drawGauge(d) {
  const { capacity: cap, throughput: t, margin_per_pig: m, given } = d;
  const g = $("#gauge");
  if (!cap.flows || !cap.crates) {
    g.innerHTML = `<p class="empty">${cap.flows
      ? "분만사와 뒷단 돈사를 등록하면 출하 상한이 나옵니다."
      : "막힌 돈사를 먼저 푸세요 — 흐름이 안 돌면 상한을 말할 수 없습니다."}</p>`;
    $("#levers").innerHTML = ""; $("#caveat").innerHTML = ""; return;
  }
  const pct = Math.round(t.achieved * 100);
  g.innerHTML = `<div class="gauge">
    <div class="gtop"><span class="gbig">${n0(t.ceiling_year)}두</span>
      <span class="gsub">연간 출하 <b>상한</b> — 분만틀 ${t.crates} × 배치 ${t.batches_per_year}회/년</span></div>
    <div class="meter"><div class="fill" style="width:${Math.min(100, Math.max(0, pct))}%"></div>
      <span class="cap" style="color:${pct > 22 ? "#fff" : "var(--ink)"}">지금 ${n0(t.now_year)}두 · ${pct}%</span></div>
    <div class="gsub">${given
      ? `상한까지 <b>${n0(t.gap_year)}두</b> 남았습니다 — 연 <b>${man(t.gap_year * m.margin)}</b>.`
      : `<b>성적을 안 넣어서 ‘지금’은 우리 농장 값이 아닙니다</b> — 설계 기준으로 돌린 것이라 상한과 같게 나옵니다.`}
      ${t.weaned_room_bound ? `<br>복당 이유두수 상한이 목표 ${CONST.ceiling.weaned}두가 아니라
        <b>${t.top_weaned.toFixed(1)}두</b>입니다 — <b>방이 먼저 막습니다.</b>` : ""}</div>
    <div class="eqn">연간 출하 = <b>${t.crates}</b><span class="fx">분만틀</span>
      × <b>${(t.factors.fill * 100).toFixed(0)}%</b><span class="fx">채움률</span>
      × <b>${t.factors.weaned.toFixed(1)}</b><span class="fx">복당이유</span>
      × <b>${(t.factors.survival * 100).toFixed(0)}%</b><span class="fx">육성률</span>
      × <b>${t.batches_per_year}</b><span class="fx">배치/년</span>
      = <b>${n0(t.now_year)}두</b></div></div>`;

  const sorted = [...t.ways].sort((a, b) => b.gain - a.gain);
  $("#levers").innerHTML = sorted.map((w, i) => `
    <div class="lever${w.at_target ? " done" : (i === 0 ? " top" : "")}">
      <span class="nm">${w.name}</span>
      <span class="gain">${w.at_target ? "도달" : "+" + n0(w.gain) + "두"}</span>
      <span class="from">${w.now}${w.unit} → ${w.target}${w.unit}</span>
      <span class="won">${w.at_target ? "—" : man(w.gain * m.margin) + "/년"}</span>
      <span class="how">${w.how}</span></div>`).join("");

  $("#caveat").className = "note";
  $("#caveat").innerHTML = `<b>세 몫을 더하지 마세요.</b> 항이 곱해지므로 개별 합
    ${n0(t.sum_of_ways)}두 ≠ 총 격차 ${n0(t.gap_year)}두입니다.
    각 몫은 <b>그것 하나만</b> 설계 기준까지 올렸을 때의 값입니다.<br>
    상한을 더 올리려면 성적이 아니라 <b>돈사</b>를 늘려야 합니다 —
    지금 붙잡고 있는 건 <b>${cap.binding}</b>입니다.
    원/년은 <b>한계 이익 ${n0(m.margin)}원/두</b> 기준이라 사료·약품·수송만 뺐습니다.
    <b>증축 판단에는 쓸 수 없습니다.</b>`;
}

function drawVerdict(d) {
  const cap = d.capacity, t = d.throughput;
  const set = (id, v, cls) => {
    const box = document.getElementById(id).parentElement;
    box.className = "vitem" + (cls ? " " + cls : "");
    document.getElementById(id).textContent = v;
  };
  set("v-sows", cap.n_sows ? n0(cap.n_sows) + "두" : (cap.blocked.length ? "막힘" : "—"),
    cap.flows ? "" : "bad");
  set("v-bind", cap.binding || "—", cap.binding ? (cap.flows ? "hot" : "bad") : "");
  set("v-top", cap.crates ? n0(t.ceiling_year) + "두" : "—");
  const pct = cap.crates ? Math.round(t.achieved * 100) : null;
  set("v-pct", pct === null ? "—" : pct + "%",
    pct !== null && pct < 85 && d.given ? "hot" : "");
}

/* ── 갱신 ──────────────────────────────────────── */
let timer = null;
function refresh() {
  clearTimeout(timer);
  timer = setTimeout(async () => {
    const s = setup();
    if (!s.barns.length) {
      $("#rail").innerHTML = `<p class="empty">돈사를 등록하면 나옵니다.</p>`;
      $("#gauge").innerHTML = ""; $("#levers").innerHTML = "";
      $("#railnote").textContent = ""; $("#caveat").textContent = "";
      return;
    }
    try {
      const d = await api("/api/capacity", { method: "POST", body: JSON.stringify(s) });
      lastCap = d;
      drawVerdict(d); drawRail(d.capacity); drawGauge(d);
    } catch (e) {
      $("#railnote").className = "note bad";
      $("#railnote").textContent = "계산 실패: " + e.message;
    }
  }, 180);
}

/* ── 시뮬레이션 ────────────────────────────────── */
async function runWatch() {
  const btn = $("#b-watch"), out = $("#watch"), hint = $("#h-watch");
  btn.disabled = true; hint.className = "hint"; hint.textContent = "돌리는 중…";
  try {
    const r = await api("/api/capacity/watch?days=400",
      { method: "POST", body: JSON.stringify(setup()) });
    const bad = Object.entries(r.counts).filter(([k]) => k !== "유휴");
    const cls = r.verdict === "정상" ? "good" : (r.verdict === "흐름 실패" ? "stop" : "");
    out.innerHTML = `<div class="kpis">
        <div class="kpi"><span class="v">${r.verdict}</span><span class="k">판정</span>
          <span class="d">배치 ${r.batch_system} · 방당 분만틀 ${r.crate_count}</span></div>
        <div class="kpi"><span class="v">${n0(r.n_transitions)}</span><span class="k">전이</span>
          <span class="d">정상상태 ${n0(r.n_steady)}회</span></div>
        <div class="kpi"><span class="v">${(r.utilization * 100).toFixed(0)}%</span>
          <span class="k">방 가동률</span><span class="d">돈방 ${r.n_rooms}개</span></div>
      </div>
      <div class="chips" style="margin-top:12px">
        ${Object.entries(r.counts).map(([k, v]) =>
          `<span class="pill ${k === "유휴" ? "mute" : (v ? "stop" : "good")}">${k} ${v}회</span>`).join("")}
      </div>
      ${r.blocked && r.blocked.length ? `<p class="note bad" style="margin-top:10px">
        <b>돌려 보기 전에 이미 막혔다</b> — ${r.blocked.map(b => b.msg).join(" · ")}<br>
        위 ‘위반 0건’은 지켜져서가 아니라 <b>아무것도 움직이지 않아서</b>입니다.</p>` : ""}
      ${r.notes.length ? `<p class="note" style="margin-top:10px">
        ${r.notes.map(n => n.barn ? `· ${n.barn}(${n.stage}) — ${n.why}` : `· ${n.why}`).join("<br>")}</p>` : ""}`;
    hint.className = "hint " + (r.verdict === "정상" ? "ok" : "bad");
    hint.textContent = `판정 ${r.verdict} · 위반 ${bad.reduce((a, [, v]) => a + v, 0)}건`;
  } catch (e) {
    hint.className = "hint bad"; hint.textContent = e.message;
    out.innerHTML = "";
  } finally { btn.disabled = false; }
}

/* ── 번식 ──────────────────────────────────────── */
async function makeSchedule() {
  const hint = $("#h-sched");
  const d = $("#w-date").value;
  if (!d) { hint.className = "hint bad"; hint.textContent = "이유일을 넣으세요"; return; }
  hint.className = "hint"; hint.textContent = "생성 중…";
  try {
    const r = await api("/api/breeding/schedule", {
      method: "POST",
      body: JSON.stringify({
        weaning_date: d, parity: $("#w-parity").value,
        season_hot: $("#w-hot").value === "1"
      })
    });
    const s = r.summary;
    $("#summary").innerHTML = `<div class="kpis">
      <div class="kpi"><span class="v">${s.service_date}</span><span class="k">교배 예정</span>
        <span class="d">재귀발정 ${s.wei_days}일</span></div>
      <div class="kpi"><span class="v">${s.farrow_date}</span><span class="k">분만 예정</span>
        <span class="d">임신 115일</span></div>
      <div class="kpi"><span class="v">${s.cycle_days}일</span><span class="k">번식주기</span>
        <span class="d">회전 ${s.turnover_per_year}/년</span></div>
      <div class="kpi"><span class="v">${s.npd_days}일</span><span class="k">비생산일수</span>
        <span class="d">이 주기분</span></div></div>`;
    $("#tasks").innerHTML = `<div class="tblwrap" style="margin-top:14px"><table>
      <thead><tr><th>날짜</th><th>작업</th><th>내용</th></tr></thead><tbody>
      ${r.tasks.map(t => `<tr class="${t.estimated ? "est" : ""}">
        <td class="d">${t.date}</td><td>${t.task}</td><td>${t.detail}</td></tr>`).join("")}
      </tbody></table></div>
      <p class="note" style="margin-top:10px">회색 행은 <b>추정치</b>입니다(끝에 ~).
        실제 발정·교배가 확인되면 그 날짜로 바뀝니다.</p>`;
    hint.className = "hint ok"; hint.textContent = `${r.tasks.length}개 작업`;

    // 교배 적기는 일정과 같이 봐야 뜻이 산다.
    // **비교는 서버가 한다** — 각 주기마다 그 주기의 최적 프로토콜을 다시
    // 찾아야 공정하고, 그걸 화면에서 하면 또 갈린다.
    const det = await api("/api/breeding/detection");
    $("#timing").innerHTML = `<div class="kpis">${det.rows.map(r => `
      <div class="kpi"><span class="v">${r.conception.toFixed(3)}</span>
        <span class="k">${r.label}</span>
        <span class="d">${r.vs_continuous_pp === 0 ? "기준"
          : r.vs_continuous_pp.toFixed(1) + "%p"} · 발견 +${r.offsets.map(h => h.toFixed(0)).join("/+")}h</span>
      </div>`).join("")}</div>
      <p class="note" style="margin-top:10px">
        각 주기는 <b>그 주기에 최적화된 주입 시점</b>을 씁니다 — 시점을 고정한 채
        주기만 늘리면 CCTV가 부당하게 유리해집니다. 남는 차이가 곧
        <b>불확실성 자체의 비용</b>입니다.<br>
        ‘CCTV가 좋다’가 아니라 <b>‘하루 1회는
        ${Math.abs(det.rows[2].vs_continuous_pp).toFixed(1)}%p를 잃는다’</b>로 읽으세요.</p>`;
  } catch (e) {
    hint.className = "hint bad"; hint.textContent = e.message;
  }
}

/* ── 투자 순서 (병목 체인) ─────────────────────── */
async function runRelief() {
  const btn = $("#b-relief"), out = $("#relief"), hint = $("#h-relief");
  btn.disabled = true; hint.className = "hint"; hint.textContent = "재는 중…";
  try {
    const r = await api("/api/capacity/relief", {
      method: "POST", body: JSON.stringify(setup())
    });
    if (!r.groups.length) {
      out.innerHTML = `<p class="empty">돈사를 등록하면 나옵니다.</p>`;
      hint.textContent = ""; return;
    }
    out.innerHTML = `<div class="levers">${r.groups.map((g, i) => {
      const tie = g.stages.length > 1;
      return `<div class="lever${i === 0 ? " top" : ""}">
        <span class="nm">${i + 1}. ${g.stages.join(" + ")}
          ${tie ? `<span class="pill">동률 ${g.stages.length}곳</span>` : ""}</span>
        <span class="gain">${g.gain === null ? "—"
          : (g.gain > 0 ? "+" + n0(g.gain) + "두" : "±0")}</span>
        <span class="from">여기서 ${n0(g.n_sows)}두에 걸립니다</span>
        <span class="won">${g.gain === null ? "그 다음은 안 쟀습니다"
          : (g.gain > 0 ? "풀면 여기까지" : "")}</span>
        ${tie ? `<span class="how"><b>하나만 넓혀도 두수가 안 늘어납니다</b> —
          ${g.stages.join("·")} 가 같은 수준에서 나란히 걸려 있습니다.</span>` : ""}
      </div>`; }).join("")}</div>
      <p class="note" style="margin-top:10px">${r.note}
        <b>순서만 말하는 표입니다</b> — 넓히는 데 드는 비용과 실제 도달
        가능성은 여기서 다루지 않습니다.</p>`;
    hint.className = "hint ok";
    hint.textContent = `${r.groups.length}단계`;
  } catch (e) {
    hint.className = "hint bad"; hint.textContent = e.message; out.innerHTML = "";
  } finally { btn.disabled = false; }
}

/* ── 간격 what-if ──────────────────────────────── */
async function runInterval() {
  const btn = $("#b-iv"), out = $("#ivtab"), hint = $("#h-iv");
  btn.disabled = true; hint.className = "hint"; hint.textContent = "재는 중…";
  try {
    const r = await api("/api/capacity/interval", {
      method: "POST", body: JSON.stringify(setup())
    });
    const top = Math.max(...r.rows.map(x => x.n_sows), 1);
    out.innerHTML = `<div class="tblwrap"><table>
      <thead><tr><th>간격</th><th>받을 수 있는 모돈</th><th>병목</th>
        <th>연간 출하 상한</th><th>배치당 교배</th><th>한 날 집중도</th></tr></thead>
      <tbody>${r.rows.map(x => {
        const cur = x.current, blocked = x.n_sows === 0;
        return `<tr${cur ? ' class="cur"' : ""}>
          <td><b>${x.name}</b> <span class="d">${x.interval_days}일</span>
            ${cur ? `<span class="pill">지금</span>` : ""}
            ${!cur && x.interval_days === r.best
              ? `<span class="pill good">최대 규모</span>` : ""}</td>
          <td>${blocked ? `<span class="pill stop">막힘</span>`
            : `<span class="qbar"><i style="width:${x.n_sows / top * 100}%"></i></span>
               <span class="d">${n0(x.n_sows)}두</span>`}</td>
          <td class="d">${blocked
            ? `${x.blocked.join("·")} 방 부족` : x.binding}</td>
          <td class="d">${blocked ? "—" : n0(x.ceiling_year) + "두"}</td>
          <td class="d">${x.services_per_batch === null ? "—"
            : x.services_per_batch + "두"}</td>
          <td class="d">×${x.peak_ratio}</td></tr>`;
      }).join("")}</tbody></table></div>
      <p class="note">${r.note}</p>
      <p class="note"><b>막힌 간격은 두수를 줄여서 풀리지 않습니다</b> —
        회전이 안 되는 것이라 방을 늘리거나 간격을 넓혀야 합니다.
        ${r.given ? "" : `성적을 비웠으므로 출하는 <b>설계 상한</b>이고
        지금 나오는 값이 아닙니다.`}</p>`;
    hint.className = "hint ok";
    hint.textContent = r.best === r.current_interval
      ? "지금 간격이 규모가 가장 큽니다"
      : `규모 최대는 ${r.best}일`;
  } catch (e) {
    hint.className = "hint bad"; hint.textContent = e.message; out.innerHTML = "";
  } finally { btn.disabled = false; }
}

/* ── 내보내기 ──────────────────────────────────── */
// **파일을 브라우저가 만들지 않는다.** 서버가 CSV 를 내려주고 여기는 저장만
// 시킨다. 화면이 자기 CSV 를 만들면 등급 열과 각주 머리말이 빠지고, 그러면
// 격차 분해가 개입 효과처럼 읽힌다 — 그게 이 프로젝트가 막으려는 오독이다.
function exportBody(sheet) {
  if (sheet === "capacity" || sheet === "interval") return { setup: setup() };
  if (sheet === "season") {
    return { sows: Math.round(num("#s-sows") ?? lastCap?.capacity?.n_sows ?? 300) };
  }
  if (sheet === "diagnosis" || sheet === "priority") {
    return {
      sows: Math.round(num("#d-sows") ?? 300),
      performance: {
        weaned: num("#d-wl"), npd: num("#d-npd"),
        farrowing_rate: num("#d-fr"), wean_to_estrus: num("#d-we")
      }
    };
  }
  return {};
}

async function runExport(btn) {
  const sheet = btn.dataset.sheet;
  const hint = btn.parentElement.querySelector(".hint");
  const say = (cls, msg) => { if (hint) { hint.className = cls; hint.textContent = msg; } };
  btn.disabled = true; say("hint", "내보내는 중…");
  try {
    const r = await fetch(`/api/export/${sheet}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(exportBody(sheet))
    });
    if (!r.ok) {
      let m = r.statusText;
      try { m = (await r.json()).detail ?? m; } catch (e) { /* 본문 없음 */ }
      throw new Error(typeof m === "string" ? m : JSON.stringify(m));
    }
    // 파일 이름은 서버가 정한다 — **농장 이름을 넣지 않는다**(식별자다)
    const cd = r.headers.get("content-disposition") || "";
    const name = (cd.match(/filename="([^"]+)"/) || [])[1]
      || `yangdon_${sheet}.csv`;
    const url = URL.createObjectURL(await r.blob());
    const a = el("a"); a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    say("hint ok", name);
  } catch (e) {
    say("hint bad", e.message);
  } finally { btn.disabled = false; }
}

/* ── 오늘 할 일 ────────────────────────────────── */
async function runQueue() {
  const hint = $("#h-queue"), out = $("#queue");
  const first = $("#q-first").value, on = $("#q-on").value;
  if (!first) { hint.className = "hint bad"; hint.textContent = "첫 배치 이유일을 넣으세요"; return; }
  hint.className = "hint"; hint.textContent = "생성 중…";
  try {
    // 배치 날짜 생성도 서버가 한다 — 간격을 더하는 것도 계산이다
    const iv = +$("#f-iv").value;
    const b = await api(`/api/breeding/batches?first_weaning=${first}`
      + `&interval_days=${iv}&n=${Math.round(num("#q-n") ?? 7)}`);
    const q = new URLSearchParams();
    for (const d of b.weaning_dates) q.append("weaning", d);
    if (on) q.set("on", on);
    q.set("horizon", String(Math.round(num("#q-h") ?? 0)));
    const r = await api("/api/breeding/today?" + q);

    const rows = (list, late) => list.length ? `<div class="tblwrap"><table>
        <thead><tr><th>배치</th><th>날짜</th>${late ? "<th>지연</th>" : ""}
          <th>작업</th><th>내용</th></tr></thead>
        <tbody>${list.map(t => `<tr class="${t.estimated ? "est" : ""}">
          <td>${t.id ?? "—"}</td><td class="d">${t.date}</td>
          ${late ? `<td class="d"><span class="pill stop">+${t.late_days}일</span></td>` : ""}
          <td>${t.task}</td><td>${t.detail ?? ""}</td></tr>`).join("")}
        </tbody></table></div>`
      : `<p class="empty">없습니다.</p>`;

    out.innerHTML = `<div class="kpis" style="margin-top:4px">
        <div class="kpi"><span class="v">${r.due.length}</span><span class="k">할 일</span>
          <span class="d">${r.on} 기준 · 앞으로 ${r.horizon_days}일</span></div>
        <div class="kpi"><span class="v">${r.overdue.length}</span><span class="k">지난 것</span>
          <span class="d">기한이 지났습니다</span></div>
        <div class="kpi"><span class="v">${b.weaning_dates.length}</span><span class="k">배치</span>
          <span class="d">간격 ${iv}일</span></div></div>
      <h3 style="margin-top:16px;font-size:.95rem">오늘 할 일</h3>${rows(r.due, false)}
      <h3 style="margin-top:16px;font-size:.95rem">지난 것</h3>${rows(r.overdue, true)}
      <p class="note" style="margin-top:10px">배치 날짜는 <b>간격을 그대로 더한
        유도값</b>이고 실제 이력이 아닙니다 — ${b.grade}. 회색 행은 추정치입니다.</p>`;
    hint.className = "hint ok";
    hint.textContent = `할 일 ${r.due.length} · 지난 것 ${r.overdue.length}`;
  } catch (e) {
    hint.className = "hint bad"; hint.textContent = e.message; out.innerHTML = "";
  }
}

/* ── 성적 진단 · 처방 ──────────────────────────── */
// 등급을 문장에 박는다. 회수량 큰 순으로만 세우면 횡단면 비교가 농장 내
// 변화처럼 읽힌다 — 이 프로젝트가 실측/계산/가정을 구분해 온 것의 처방 버전.
const GRADE_CLS = { A: "good", B: "", C: "mute" };

async function runDiagnosis() {
  const hint = $("#h-diag");
  const perf = {
    weaned: num("#d-wl"), npd: num("#d-npd"),
    farrowing_rate: num("#d-fr"), wean_to_estrus: num("#d-we")
  };
  const sows = num("#d-sows") ?? 300;
  hint.className = "hint"; hint.textContent = "진단 중…";
  try {
    const d = await api(`/api/diagnosis?sows=${Math.round(sows)}`, {
      method: "POST", body: JSON.stringify(perf)
    });
    const g = d.diagnosis, p = d.priority;

    $("#diag-head").innerHTML = `<div class="kpis">
      <div class="kpi"><span class="v">${g.psy}</span><span class="k">내 PSY</span>
        <span class="d">항등식으로 낸 값</span></div>
      <div class="kpi"><span class="v">${g.psy_median_farm}</span><span class="k">중앙 농장</span>
        <span class="d">지표별 중앙값을 항등식에 넣은 합성값</span></div>
      <div class="kpi"><span class="v">${g.psy_gap > 0 ? "+" : ""}${g.psy_gap.toFixed(2)}두</span>
        <span class="k">격차</span>
        <span class="d">PSY 열 자체의 중앙은 ${g.psy_median_observed}</span></div></div>`;

    $("#diag-rows").innerHTML = `<div class="tblwrap" style="margin-top:14px"><table>
      <thead><tr><th>지표</th><th>내 값</th><th>중앙</th><th>거리</th>
        <th>되돌리면</th></tr></thead><tbody>
      ${g.rows.map(r => `<tr>
        <td>${r.name_ko}</td><td class="d">${r.value}</td><td class="d">${r.median}</td>
        <td class="d">IQR ${r.iqr_z > 0 ? "+" : ""}${r.iqr_z.toFixed(2)}
          <span class="pill ${r.band.includes("좋") ? "good"
            : (r.band.includes("나쁨") ? "stop" : "mute")}">${r.band}</span></td>
        <td class="d">${r.psy_recover != null
          ? (r.psy_recover > 0 ? "+" : "") + r.psy_recover.toFixed(2) + "두"
          : "간접 지표"}</td></tr>`).join("")}
      </tbody></table></div>
      <p class="note" style="margin-top:10px"><b>순위가 아니라 거리입니다.</b>
        IQR 단위로 중앙값에서 얼마나 떨어졌는지를 봅니다. 기준은
        국내 202농장 × 4년 = 466행 실측입니다.</p>`;

    // 처방 순서
    $("#prio-panel").style.display = "";
    $("#prio").innerHTML = p.rows.map((r, i) => `
      <div class="lever${i === 0 ? " top" : ""}">
        <span class="nm">${r.name}
          <span class="pill ${GRADE_CLS[r.grade] ?? ""}">${r.grade} · ${r.axis}</span></span>
        <span class="gain">${r.psy != null
          ? (r.psy > 0 ? "+" : "") + r.psy.toFixed(2) + "두" : "—"}</span>
        <span class="from">${r.target}</span>
        <span class="won">${r.won_year != null ? man(r.won_year) + "/년" : "—"}</span>
        ${r.note ? `<span class="how">${r.note}</span>` : ""}
      </div>`).join("");

    $("#prio-note").innerHTML =
      `<b>[축이 둘이다]</b> ${Object.entries(p.axes)
        .map(([k, v]) => `<br>· <b>${k}</b> — ${v}`).join("")}
       <br><br><b>[근거 등급]</b> ${Object.entries(p.grades)
        .map(([k, v]) => `<br>· <b>${k}</b> ${v[0]} — ${v[1]}`).join("")}`;
    $("#prio-foot").innerHTML =
      `<b>[합치지 않는다]</b> 개별 회수량 합 ${p.sum_of_parts}두 vs
       총 격차 ${Math.abs(p.psy_gap).toFixed(2)}두. ${p.sum_note}
       <br><br><b>${p.footer.replace(/\n/g, "<br>")}</b>`;

    hint.className = "hint ok";
    hint.textContent = `${d.given.length}개 지표로 진단 (비운 칸은 제외)`;
  } catch (e) {
    hint.className = "hint bad"; hint.textContent = e.message;
    $("#diag-head").innerHTML = ""; $("#diag-rows").innerHTML = "";
    $("#prio-panel").style.display = "none";
  }
}

/* ── 여름 손실 ─────────────────────────────────── */
// 분포 띠 — **중앙값 하나로 말하면 안 된다.** 농장마다 갈리는 게 요지라
// 분포를 그리고 내 위치를 찍는다.
function strip(q, me) {
  const lo = q.p10, hi = q.p90, span = Math.max(1e-9, hi - lo), pad = span * 0.2;
  const x0 = lo - pad, x1 = hi + pad;
  const px = x => ((x - x0) / (x1 - x0)) * 100;
  const cl = x => Math.max(0, Math.min(100, x));
  const col = me != null && me > q.median ? "var(--stop)" : "var(--good)";
  return `<div class="strip">
    <div class="band" style="left:${px(lo)}%;width:${px(hi) - px(lo)}%"></div>
    <div class="band2" style="left:${px(q.p25)}%;width:${px(q.p75) - px(q.p25)}%"></div>
    <div class="med" style="left:${px(q.median)}%"></div>
    ${me != null ? `<div class="me" style="left:calc(${cl(px(me))}% - 7px);
      background:${col}"></div>
      <div class="lb top" style="left:${cl(px(me))}%;color:${col}">
      내 농장 ${me > 0 ? "+" : ""}${me.toFixed(1)}</div>` : ""}
    <div class="lb" style="left:${px(lo)}%">하위10% ${lo}</div>
    <div class="lb" style="left:${px(q.median)}%">중앙 ${q.median}</div>
    <div class="lb" style="left:${px(hi)}%">상위10% ${hi}</div></div>`;
}

async function runSeason() {
  const hint = $("#h-season");
  const q = new URLSearchParams({ sows: Math.round(num("#s-sows") ?? 300) });
  for (const [k, id] of [["psy", "#s-psy"], ["summer", "#s-summer"],
                         ["winter", "#s-winter"]]) {
    const v = num(id); if (v !== null) q.set(k, v);
  }
  hint.className = "hint"; hint.textContent = "계산 중…";
  try {
    const d = await api("/api/season?" + q);
    const acc = d.accidents;

    if (d.given) {
      const m = d.mine;
      $("#season-head").innerHTML = `<div class="kpis">
        <div class="kpi"><span class="v">${m.loss_pp > 0 ? "+" : ""}${m.loss_pp.toFixed(1)}%p</span>
          <span class="k">우리 농장 여름 손실</span>
          <span class="d">겨울 ${m.winter} − 여름 ${m.summer}</span></div>
        <div class="kpi"><span class="v">${m.d_psy > 0 ? "+" : ""}${m.d_psy.toFixed(2)}두</span>
          <span class="k">연간 PSY 손실</span>
          <span class="d">PSY ${d.psy_used} · ${d.psy_source}</span></div>
        <div class="kpi"><span class="v">${man(m.won_year)}</span>
          <span class="k">연 손실 상한</span>
          <span class="d">${n0(d.n_sows)}두 기준</span></div></div>`;
      $("#season-dist").innerHTML = strip(d.loss, m.loss_pp)
        // percentile_hint 는 그 자체가 문장이다 — 뒤에 조사를 붙이면 깨진다
        + `<p class="note">국내 <b>${d.n_farms}농장</b> 분포 대비 —
           <b>${m.percentile_hint}</b></p>`;
    } else {
      const sc = d.scenario;
      $("#season-head").innerHTML = `<div class="kpis">
        <div class="kpi"><span class="v">${man(sc.median.won_year)}</span>
          <span class="k">중앙 농장이라면</span>
          <span class="d">여름 손실 ${sc.median.loss_pp > 0 ? "+" : ""}${sc.median.loss_pp}%p · 가정</span></div>
        <div class="kpi"><span class="v">${man(sc.p90.won_year)}</span>
          <span class="k">취약 상위10% 라면</span>
          <span class="d">여름 손실 +${sc.p90.loss_pp}%p · 가정</span></div>
        <div class="kpi"><span class="v">${n0(d.n_sows)}두</span>
          <span class="k">우리 규모로 환산</span>
          <span class="d">PSY ${d.psy_used} · ${d.psy_source}</span></div></div>`;
      $("#season-dist").innerHTML = strip(d.loss, null)
        + `<p class="note" style="color:var(--warn)"><b>두 칸을 비웠으므로
           위는 우리 농장 값이 아닙니다</b> — 국내 분포를 우리 규모로 환산한
           범위입니다. 어느 쪽인지 알려면 월별 분만율 12개월이 필요합니다.</p>
           <p class="note">패널 실측 기준(농장마다 <b>자기</b> PSY·<b>자기</b>
           겨울로 낸 금액)의 중앙은 <b>${man(d.panel_won_ref.median)}</b>으로
           위와 다릅니다 — 곱의 중앙값 ≠ 중앙값의 곱.</p>`;
    }

    // 축소 후 분포 — 표본 오차를 걷어내면 폭이 줄지만 **사라지지는 않는다**
    $("#season-dist").innerHTML += `<div class="kpis" style="margin-top:14px">
      <div class="kpi"><span class="v">${(d.spread.true_share * 100).toFixed(0)}%</span>
        <span class="k">진짜 농장 차이</span>
        <span class="d">관측 분산 중. 나머지는 표본 오차</span></div>
      <div class="kpi"><span class="v">${d.loss_shrunk.p10} ~ ${d.loss_shrunk.p90}</span>
        <span class="k">축소 후 분포 %p</span>
        <span class="d">관측 ${d.loss.p10} ~ ${d.loss.p90}</span></div>
      <div class="kpi"><span class="v">ρ ${d.join.PSY.rho}</span>
        <span class="k">PSY 와의 상관</span>
        <span class="d">연간 성적으로는 못 맞힙니다</span></div></div>`;

    // 무너지는 경로
    $("#season-why").style.display = "";
    const key = "임신사고(1차)";
    const top = Object.entries(acc.delta).sort((a, b) => b[1] - a[1]).slice(0, 4);
    $("#season-acc").innerHTML = `<div class="tblwrap"><table>
      <thead><tr><th>임신사고 구성</th><th>겨울</th><th>여름</th><th>차이</th></tr></thead>
      <tbody>${top.map(([k, v]) => `<tr${k === key ? ' style="font-weight:600"' : ""}>
        <td>${k.replace("임신사고", "").replace(/[()]/g, "")}</td>
        <td class="d">${(acc.winter[k] * 100).toFixed(1)}%</td>
        <td class="d">${(acc.summer[k] * 100).toFixed(1)}%</td>
        <td class="d"><span class="pill ${v > 0.02 ? "stop" : "mute"}">${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}%p</span></td>
      </tr>`).join("")}</tbody></table></div>
      <p class="note"><b>여름에 이유두수·재귀율은 거의 그대로인데 임신사고
        구성이 1차 재발 쪽으로 ${(acc.delta[key] * 100).toFixed(1)}%p 기웁니다.</b>
        무너지는 건 사양이 아니라 <b>착상</b>이라는 뜻이고, 겨냥할 시점은
        <b>교배 후 ${d.implantation_window[0]}~${d.implantation_window[1]}일
        착상기</b>입니다 — 이 구간 축사의 THI 를 낮추는 것이 처방입니다.</p>`;
    // 서버 문구가 **강조** 표기를 쓴다 — 그대로 넣으면 별표가 보인다
    $("#season-caveat").innerHTML = d.caveats
      .map(c => "· " + c.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")).join("<br>");

    hint.className = "hint ok";
    hint.textContent = d.given ? "우리 농장 값" : "분포 환산 (가정)";
  } catch (e) {
    hint.className = "hint bad"; hint.textContent = e.message;
    $("#season-head").innerHTML = ""; $("#season-dist").innerHTML = "";
    $("#season-why").style.display = "none";
  }
}

/* ── 농장 목록 ─────────────────────────────────── */
async function loadFarms() {
  const box = $("#farmlist");
  try {
    const fs = await api("/api/farms");
    if (!fs.length) { box.innerHTML = `<p class="empty">아직 등록한 농장이 없습니다.</p>`; return; }
    box.innerHTML = fs.map(f => `<div class="farmcard">
      <b>${f.name}</b>
      <span class="meta">돈사 ${(f.setup.barns || []).length}동 · ${f.updated_at.slice(0, 10)}</span>
      <span class="sp">
        <button class="act" data-load="${f.id}">불러오기</button>
        <button class="act" data-del="${f.id}">삭제</button></span></div>`).join("");
  } catch (e) { box.innerHTML = `<p class="hint bad">${e.message}</p>`; }
}

function applySetup(s) {
  barns = {};
  for (const b of s.barns || []) barns[b.stage] = { rooms: b.rooms, per: b.per };
  $("#f-iv").value = s.interval_days ?? 21;
  $("#f-lact").value = s.lactation_days ?? 24;
  $("#f-pre").value = s.pre_farrow_days ?? 7;
  $("#f-wash").value = s.washout_days ?? 7;
  const p = s.performance || {};
  $("#p-fr").value = p.farrowing_rate ?? "";
  $("#p-wl").value = p.weaned ?? "";
  $("#p-gs").value = p.survival ?? "";
  drawBarns(); refresh();
}

/* ── 조립 ──────────────────────────────────────── */
document.addEventListener("input", e => {
  const t = e.target;
  if (t.dataset && t.dataset.st) {
    const st = t.dataset.st, k = t.dataset.k;
    barns[st] = barns[st] || { rooms: 0, per: 0 };
    barns[st][k] = Math.max(0, Math.floor(+t.value || 0));
    const pos = t.selectionStart;
    drawBarns();
    const back = document.querySelector(`[data-st="${st}"][data-k="${k}"]`);
    if (back) { back.focus(); try { back.setSelectionRange(pos, pos); } catch (err) { /* number 입력은 미지원 */ } }
  }
  refresh();
});
document.addEventListener("change", refresh);

document.addEventListener("click", async e => {
  const t = e.target;
  if (t.classList.contains("tab")) {
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("on", x === t));
    document.querySelectorAll(".tabpane").forEach(p =>
      p.classList.toggle("hidden", p.id !== "tab-" + t.dataset.tab));
    if (t.dataset.tab === "farms") loadFarms();
    if (t.dataset.tab === "season") {
      if (lastCap && lastCap.capacity.n_sows) $("#s-sows").value = lastCap.capacity.n_sows;
      runSeason();
    }
    if (t.dataset.tab === "diagnosis") {
      // 돈사 탭에 넣은 성적을 끌어온다 — 같은 농장인데 두 번 넣게 하지 않는다
      const wl = num("#p-wl"), fr = num("#p-fr");
      if (wl !== null) $("#d-wl").value = wl;
      if (fr !== null) $("#d-fr").value = fr;
      if (lastCap && lastCap.capacity.n_sows) $("#d-sows").value = lastCap.capacity.n_sows;
    }
    return;
  }
  if (t.dataset.load) {
    const f = await api("/api/farms/" + t.dataset.load);
    $("#f-name").value = f.name;
    applySetup(f.setup);
    document.querySelector('.tab[data-tab="capacity"]').click();
    return;
  }
  if (t.dataset.del) {
    await api("/api/farms/" + t.dataset.del, { method: "DELETE" });
    loadFarms();
  }
});

$("#b-preset").onclick = async () => { barns = await preset(300); drawBarns(); refresh(); };
$("#b-clear").onclick = () => { barns = {}; drawBarns(); refresh(); };
$("#b-watch").onclick = runWatch;
$("#b-sched").onclick = makeSchedule;
$("#b-diag").onclick = runDiagnosis;
$("#b-season").onclick = runSeason;
$("#b-relief").onclick = runRelief;
$("#b-iv").onclick = runInterval;
document.querySelectorAll("button.ex").forEach(
  b => { b.onclick = () => runExport(b); });
$("#b-queue").onclick = runQueue;
$("#b-save").onclick = async () => {
  const hint = $("#h-save"), name = $("#f-name").value.trim();
  if (!name) { hint.className = "hint bad"; hint.textContent = "농장 이름을 넣으세요"; return; }
  try {
    const f = await api("/api/farms", {
      method: "POST", body: JSON.stringify({ name, setup: setup() })
    });
    hint.className = "hint ok"; hint.textContent = `저장됨 (#${f.id})`;
  } catch (e) { hint.className = "hint bad"; hint.textContent = e.message; }
};

(async function boot() {
  try {
    const h = await api("/api/health");
    CONST = h.constants; STAGES = h.stages;
    $("#conn").textContent = "연결됨"; $("#conn").className = "ok";
  } catch (e) {
    $("#conn").textContent = "서버 연결 실패"; $("#conn").className = "bad";
    return;
  }
  const today = new Date().toISOString().slice(0, 10);
  $("#w-date").value = today;
  $("#q-on").value = today;
  // 첫 배치는 기준일보다 앞이어야 '지난 것' 이 의미가 있다
  $("#q-first").value = new Date(Date.now() - 100 * 864e5).toISOString().slice(0, 10);
  barns = await preset(300);
  drawBarns();
  refresh();
})();
