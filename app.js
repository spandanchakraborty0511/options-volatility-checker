/**
 * SPY Options Volatility Checker — Client-Side Logic & Plotly Renderer
 * Progressive Async Loading & Sub-10ms Responsive UI
 */

document.addEventListener("DOMContentLoaded", () => {
    const dateSelect = document.getElementById("date-select");
    const expirySelect = document.getElementById("expiry-select");
    const volThresholdInput = document.getElementById("vol-threshold");
    const volThresholdVal = document.getElementById("vol-threshold-val");
    const btnRecalculate = document.getElementById("btn-recalculate");

    const valSpot = document.getElementById("val-spot");
    const valAtmIv = document.getElementById("val-atm-iv");
    const valRv21d = document.getElementById("val-rv-21d");
    const valIvRank = document.getElementById("val-iv-rank");
    const valVolPrem = document.getElementById("val-vol-prem");

    let currentMarketPoints = [];
    let currentSignalFilter = "ALL";
    let loadedSsviDate = null;
    let loadedTimelineDate = null;

    initApp();

    async function initApp() {
        setupEventListeners();
        await loadAvailableDates();
    }

    function setupEventListeners() {
        volThresholdInput.addEventListener("input", (e) => {
            volThresholdVal.textContent = `${parseFloat(e.target.value).toFixed(2)}%`;
        });

        dateSelect.addEventListener("change", async () => {
            const selectedDate = dateSelect.value;
            if (selectedDate) {
                loadedSsviDate = null;
                loadedTimelineDate = null;
                await loadExpiriesForDate(selectedDate);
                await refreshFastPrimaryData();
            }
        });

        expirySelect.addEventListener("change", async () => {
            await loadSviSmileData();
        });

        btnRecalculate.addEventListener("click", async () => {
            loadedSsviDate = null;
            loadedTimelineDate = null;
            await refreshFastPrimaryData();
        });

        document.querySelectorAll(".tab-btn").forEach(btn => {
            btn.addEventListener("click", async (e) => {
                document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
                document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

                e.target.classList.add("active");
                const targetTabId = e.target.getAttribute("data-tab");
                document.getElementById(targetTabId).classList.add("active");

                const selectedDate = dateSelect.value;

                if (targetTabId === "tab-ssvi" && loadedSsviDate !== selectedDate) {
                    await loadSsviSurfaceData(selectedDate);
                }

                if (targetTabId === "tab-timeline" && loadedTimelineDate !== selectedDate) {
                    await loadTimelineData(selectedDate);
                }

                window.dispatchEvent(new Event('resize'));
            });
        });

        document.querySelectorAll(".pill-btn").forEach(pill => {
            pill.addEventListener("click", (e) => {
                document.querySelectorAll(".pill-btn").forEach(p => p.classList.remove("active"));
                e.target.classList.add("active");
                currentSignalFilter = e.target.getAttribute("data-filter");
                renderSignalsTable();
            });
        });
    }

    async function loadAvailableDates() {
        try {
            const res = await fetch("/api/dates");
            const data = await res.json();
            if (data.success && data.dates.length > 0) {
                dateSelect.innerHTML = "";
                data.dates.forEach(d => {
                    const opt = document.createElement("option");
                    opt.value = d;
                    opt.textContent = d;
                    dateSelect.appendChild(opt);
                });
                dateSelect.value = data.dates[data.dates.length - 1];
                await loadExpiriesForDate(dateSelect.value);
                await refreshFastPrimaryData();
            }
        } catch (err) {
            console.error("Failed to load dates:", err);
        }
    }

    async function loadExpiriesForDate(dateStr) {
        try {
            const res = await fetch(`/api/expiries?date=${dateStr}`);
            const data = await res.json();
            if (data.success && data.expiries.length > 0) {
                expirySelect.innerHTML = "";
                data.expiries.forEach(exp => {
                    const opt = document.createElement("option");
                    opt.value = exp;
                    opt.textContent = exp;
                    expirySelect.appendChild(opt);
                });
                const defaultIdx = Math.min(3, data.expiries.length - 1);
                expirySelect.value = data.expiries[defaultIdx];
            }
        } catch (err) {
            console.error("Failed to load expiries:", err);
        }
    }

    async function refreshFastPrimaryData() {
        const selectedDate = dateSelect.value;
        if (!selectedDate) return;

        await Promise.all([
            loadSummaryMetrics(selectedDate),
            loadSviSmileData()
        ]);

        const activeTab = document.querySelector(".tab-btn.active").getAttribute("data-tab");
        if (activeTab === "tab-ssvi") {
            loadSsviSurfaceData(selectedDate);
        } else if (activeTab === "tab-timeline") {
            loadTimelineData(selectedDate);
        }
    }

    async function loadSummaryMetrics(dateStr) {
        try {
            const res = await fetch(`/api/summary?date=${dateStr}`);
            const data = await res.json();
            if (data.success && data.summary) {
                const s = data.summary;
                valSpot.textContent = s.underlying_price ? `$${s.underlying_price.toFixed(2)}` : "--";
                valAtmIv.textContent = s.current_atm_iv ? `${(s.current_atm_iv * 100).toFixed(2)}%` : "--";
                valRv21d.textContent = s.rv_21d ? `${(s.rv_21d * 100).toFixed(2)}%` : "--";
                valIvRank.textContent = s.iv_rank_52w !== null ? `${s.iv_rank_52w.toFixed(1)}%` : "--";
                
                if (s.volatility_premium !== null) {
                    const sign = s.volatility_premium >= 0 ? "+" : "";
                    valVolPrem.textContent = `${sign}${(s.volatility_premium * 100).toFixed(2)}%`;
                    valVolPrem.className = `metric-value ${s.volatility_premium >= 0 ? 'text-emerald' : 'text-rose'}`;
                } else {
                    valVolPrem.textContent = "--";
                }
            }
        } catch (err) {
            console.error("Failed to load summary metrics:", err);
        }
    }

    async function loadSviSmileData() {
        const dateStr = dateSelect.value;
        const expiryStr = expirySelect.value;
        const volThresh = parseFloat(volThresholdInput.value) / 100.0;

        if (!dateStr || !expiryStr) return;

        try {
            const res = await fetch(`/api/svi?date=${dateStr}&expiry=${expiryStr}&vol_thresh=${volThresh}`);
            const data = await res.json();

            if (data.success) {
                renderSviSmileChart(data);
                renderSviParams(data.params);
                currentMarketPoints = data.market_points || [];
                renderSignalsTable();
            }
        } catch (err) {
            console.error("Failed to load SVI smile:", err);
        }
    }

    function renderSviSmileChart(data) {
        const spot = data.spot;
        const smooth = data.smooth_curve;
        const points = data.market_points;

        const traceSviCurve = {
            x: smooth.strikes,
            y: smooth.iv,
            mode: 'lines',
            name: `Fitted SVI Model (RMSE: ${(data.params.rmse * 100).toFixed(2)}%)`,
            line: { color: '#6366F1', width: 3 }
        };

        const fairPoints = points.filter(p => p.signal === 'FAIR');
        const richPoints = points.filter(p => p.signal === 'RICH');
        const cheapPoints = points.filter(p => p.signal === 'CHEAP');

        const traceFair = {
            x: fairPoints.map(p => p.strike),
            y: fairPoints.map(p => p.iv * 100),
            mode: 'markers',
            name: 'FAIR Market IV',
            marker: { color: '#3B82F6', size: 8 }
        };

        const traceRich = {
            x: richPoints.map(p => p.strike),
            y: richPoints.map(p => p.iv * 100),
            mode: 'markers',
            name: 'RICH (Overpriced)',
            marker: { color: '#F43F5E', symbol: 'triangle-up', size: 12 }
        };

        const traceCheap = {
            x: cheapPoints.map(p => p.strike),
            y: cheapPoints.map(p => p.iv * 100),
            mode: 'markers',
            name: 'CHEAP (Underpriced)',
            marker: { color: '#10B981', symbol: 'triangle-down', size: 12 }
        };

        const layout = {
            title: { text: `SVI Volatility Smile (${data.date}, Expiry: ${data.expiry}, ${data.dte}d DTE)`, font: { color: '#F9FAFB', size: 16 } },
            paper_bgcolor: '#111827',
            plot_bgcolor: '#111827',
            xaxis: { title: 'Strike Price ($)', gridcolor: '#1F2937', color: '#9CA3AF' },
            yaxis: { title: 'Implied Volatility (%)', gridcolor: '#1F2937', color: '#9CA3AF' },
            shapes: [{
                type: 'line',
                x0: spot, y0: 0, x1: spot, y1: 1,
                yref: 'paper',
                line: { color: '#6B7280', width: 1.5, dash: 'dash' }
            }],
            legend: { font: { color: '#9CA3AF' } },
            margin: { t: 50, b: 50, l: 50, r: 30 }
        };

        Plotly.newPlot("chart-svi-smile", [traceSviCurve, traceFair, traceRich, traceCheap], layout, { responsive: true });
    }

    function renderSviParams(p) {
        document.getElementById("svi-p-a").textContent = p.a.toFixed(5);
        document.getElementById("svi-p-b").textContent = p.b.toFixed(5);
        document.getElementById("svi-p-rho").textContent = p.rho.toFixed(4);
        document.getElementById("svi-p-m").textContent = p.m.toFixed(4);
        document.getElementById("svi-p-sigma").textContent = p.sigma.toFixed(4);
        document.getElementById("svi-p-rmse").textContent = `${(p.rmse * 100).toFixed(2)}% IV`;
    }

    async function loadSsviSurfaceData(dateStr) {
        const arbBox = document.getElementById("arb-status-container");
        arbBox.innerHTML = `<span>⏳ Calibrating SSVI Multi-Expiry Volatility Surface for ${dateStr}...</span>`;
        
        try {
            const res = await fetch(`/api/ssvi?date=${dateStr}`);
            const data = await res.json();
            if (data.success) {
                renderSsviSurfaceChart(data);
                loadedSsviDate = dateStr;
                if (data.params.is_arbitrage_free) {
                    arbBox.className = "arb-status-bar";
                    arbBox.innerHTML = `✅ <b>Calendar Arbitrage Freedom Verified</b> — Multi-expiry SSVI Surface is completely arbitrage-free (0 violations across ${data.curves.length} expiries).`;
                } else {
                    arbBox.className = "arb-status-bar text-rose";
                    arbBox.innerHTML = `⚠️ <b>Calendar Arbitrage Detected</b> — Surface contains ${data.params.violations} violations.`;
                }
            }
        } catch (err) {
            console.error("Failed to load SSVI surface:", err);
            arbBox.innerHTML = `<span class="text-rose">Failed to load SSVI surface data.</span>`;
        }
    }

    function renderSsviSurfaceChart(data) {
        const traces = [];
        data.curves.forEach((c) => {
            traces.push({
                x: c.strikes,
                y: c.iv,
                mode: 'lines',
                name: `${c.expiry} (${c.dte}d)`
            });
        });

        const layout = {
            title: { text: `SSVI Multi-Expiry Surface Term Structure Curves (${data.date})`, font: { color: '#F9FAFB', size: 16 } },
            paper_bgcolor: '#111827',
            plot_bgcolor: '#111827',
            xaxis: { title: 'Strike Price ($)', gridcolor: '#1F2937', color: '#9CA3AF' },
            yaxis: { title: 'Implied Volatility (%)', gridcolor: '#1F2937', color: '#9CA3AF' },
            legend: { font: { color: '#9CA3AF' } },
            margin: { t: 50, b: 50, l: 50, r: 30 }
        };

        Plotly.newPlot("chart-ssvi-surface", traces, layout, { responsive: true });
    }

    async function loadTimelineData(dateStr) {
        try {
            const res = await fetch(`/api/timeline?date=${dateStr}`);
            const data = await res.json();
            if (data.success && data.timeline) {
                const t = data.timeline;
                
                const traceAtmIv = {
                    x: t.dates,
                    y: t.atm_iv,
                    mode: 'lines',
                    name: '30d ATM Implied Volatility (%)',
                    line: { color: '#6366F1', width: 2.2 }
                };
                
                const traceRv = {
                    x: t.dates,
                    y: t.rv_21d,
                    mode: 'lines',
                    name: '21d Realized Volatility (%)',
                    line: { color: '#F43F5E', width: 1.8, dash: 'dash' }
                };

                const layout = {
                    title: { text: `1-Year Volatility History: 30d ATM IV vs 21d Realized Vol (Up to ${dateStr})`, font: { color: '#F9FAFB', size: 16 } },
                    paper_bgcolor: '#111827',
                    plot_bgcolor: '#111827',
                    xaxis: { title: 'Date', color: '#9CA3AF', gridcolor: '#1F2937' },
                    yaxis: { title: 'Volatility (%)', color: '#9CA3AF', gridcolor: '#1F2937' },
                    legend: { font: { color: '#9CA3AF' }, orientation: 'h', y: 1.12 },
                    margin: { t: 60, b: 50, l: 50, r: 30 }
                };

                Plotly.newPlot("chart-timeline", [traceAtmIv, traceRv], layout, { responsive: true });
                loadedTimelineDate = dateStr;
            }
        } catch (err) {
            console.error("Failed to load timeline:", err);
        }
    }

    function renderSignalsTable() {
        const tbody = document.getElementById("signals-tbody");
        tbody.innerHTML = "";

        let filtered = currentMarketPoints;
        if (currentSignalFilter !== "ALL") {
            filtered = currentMarketPoints.filter(p => p.signal === currentSignalFilter);
        }

        if (filtered.length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" class="text-center">No option strikes match the selected filter.</td></tr>`;
            return;
        }

        filtered.forEach(p => {
            const tr = document.createElement("tr");
            const sigClass = p.signal === 'RICH' ? 'badge-signal-rich' : (p.signal === 'CHEAP' ? 'badge-signal-cheap' : 'badge-signal-fair');

            tr.innerHTML = `
                <td>$${p.strike.toFixed(2)}</td>
                <td><span class="badge">${p.option_type === 'C' ? 'CALL' : 'PUT'}</span></td>
                <td>${(p.iv * 100).toFixed(2)}%</td>
                <td>${(p.svi_iv * 100).toFixed(2)}%</td>
                <td class="${p.iv_diff_pct >= 0 ? 'text-rose' : 'text-emerald'}">${p.iv_diff_pct >= 0 ? '+' : ''}${p.iv_diff_pct.toFixed(2)}%</td>
                <td>$${p.bid.toFixed(2)} / $${p.ask.toFixed(2)}</td>
                <td>${p.volume ? p.volume : 0}</td>
                <td><span class="${sigClass}">${p.signal}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }
});
