// Enterprise Parallel RAG Pipeline Frontend Client

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initTelemetry();
    loadJobs();
});

// Tab Switching
function initTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const targetId = tab.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');

            if (targetId === 'tab-jobs') {
                loadJobs();
            } else if (targetId === 'tab-vectordb') {
                loadVectorDbData();
            }
        });
    });
}

// Live Hardware & JVM Telemetry
async function initTelemetry() {
    updateTelemetry();
    setInterval(updateTelemetry, 4000);
}

async function updateTelemetry() {
    try {
        const res = await fetch('/api/rag/system-metrics');
        if (!res.ok) return;
        const data = await res.json();

        const chip = document.getElementById('telemetry-chip');
        if (chip) {
            chip.innerHTML = `<span class="mem-dot"></span> ${data.availableProcessors} Cores &bull; Heap: ${data.usedMemoryMb} / ${data.maxMemoryMb} MB`;
        }
    } catch (e) {
        console.warn('Telemetry polling skipped:', e.message);
    }
}

// Ask RAG Question
async function submitQuestion() {
    const input = document.getElementById('qa-query');
    const query = input.value.trim();
    if (!query) return;

    const topK = document.getElementById('qa-topk').value;
    const window = document.getElementById('chat-window');
    const askBtn = document.getElementById('btn-ask');

    // Append user message
    appendMessage(query, 'user');
    input.value = '';

    // Append loading bot message
    const botMsgDiv = appendMessage('Searching vector chunks and synthesizing grounded answer...', 'bot');
    askBtn.disabled = true;

    try {
        const res = await fetch(`/api/rag/ask?query=${encodeURIComponent(query)}&topK=${topK}`, {
            method: 'POST'
        });
        const data = await res.json();

        let citationsHtml = '';
        if (data.citations && data.citations.length > 0) {
            citationsHtml = '<div class="citation-list">' + data.citations.map(c => `
                <span class="citation-tag" title="Page ${c.page} - ${escapeHtml(c.preview)}">
                    📚 ${escapeHtml(c.document)} &bull; ${Math.round(c.similarityScore * 100)}% Match
                </span>
            `).join('') + '</div>';
        }

        botMsgDiv.innerHTML = `
            <div style="font-weight:700; color:var(--primary); margin-bottom:6px;">RAG Assistant</div>
            <div style="white-space: pre-wrap;">${escapeHtml(data.answer)}</div>
            ${citationsHtml}
        `;
    } catch (e) {
        botMsgDiv.innerHTML = `<span style="color:#CF222E;">Error: ${escapeHtml(e.message)}</span>`;
    } finally {
        askBtn.disabled = false;
        window.scrollTop = window.scrollHeight;
    }
}

function appendMessage(text, sender) {
    const window = document.getElementById('chat-window');
    const div = document.createElement('div');
    div.className = `msg msg-${sender}`;
    div.innerHTML = sender === 'user' 
        ? `<div>${escapeHtml(text)}</div>`
        : `<div style="font-weight:700; color:var(--primary); margin-bottom:6px;">RAG Assistant</div><div>${escapeHtml(text)}</div>`;
    window.appendChild(div);
    window.scrollTop = window.scrollHeight;
    return div;
}

// Ingestion Pipeline Runner
async function runIngestion(endpoint) {
    const customPath = document.getElementById('ingest-path').value.trim();
    const statusText = document.getElementById('ingest-status');
    const outputPre = document.getElementById('ingest-output');

    statusText.innerHTML = '<span style="color:var(--secondary); font-weight:600;">⚡ Processing pipeline on NVMe SSD...</span>';
    outputPre.textContent = '// Running...';

    const url = customPath ? `${endpoint}?folderPath=${encodeURIComponent(customPath)}` : endpoint;

    try {
        const res = await fetch(url, { method: 'POST' });
        const data = await res.json();

        statusText.innerHTML = `<span style="color:#0A7A66; font-weight:700;">✔ Completed in ${data.totalExecutionTimeMs || data.executionTimeMs || 0} ms</span>`;
        outputPre.textContent = JSON.stringify(data, null, 2);

        if (data.parsing) {
            document.getElementById('stat-files').textContent = data.parsing.processed;
            document.getElementById('stat-pages').textContent = data.parsing.totalPages;
            document.getElementById('stat-speed').textContent = Math.round(data.parsing.pagesPerSecond);
        }
        if (data.chunking) {
            document.getElementById('stat-chunks').textContent = data.chunking.totalChunks;
        }

        loadJobs();
    } catch (e) {
        statusText.innerHTML = `<span style="color:#CF222E; font-weight:700;">❌ Ingestion failed: ${escapeHtml(e.message)}</span>`;
    }
}

// Load Document Jobs from H2/MySQL
async function loadJobs() {
    const tbody = document.getElementById('jobs-body');
    if (!tbody) return;

    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:16px;">Loading database records...</td></tr>';

    try {
        const res = await fetch('/api/rag/jobs');
        const jobs = await res.json();

        if (jobs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:16px; color:var(--text-muted);">No documents processed yet. Drop PDFs into data/raw_pdfs and click Ingest.</td></tr>';
            return;
        }

        tbody.innerHTML = jobs.map(j => `
            <tr>
                <td><strong>${escapeHtml(j.filename)}</strong></td>
                <td>
                    <span class="status-badge ${j.status === 'SUCCESS' ? 'status-success' : 'status-failed'}">
                        ${escapeHtml(j.status)}
                    </span>
                </td>
                <td>${j.totalPages}</td>
                <td>${j.executionTimeMs} ms</td>
                <td><code style="font-size:0.75rem; color:var(--primary);">${j.checksum ? j.checksum.substring(0, 16) + '...' : '-'}</code></td>
                <td>${j.processedAt ? j.processedAt.replace('T', ' ').substring(0, 19) : '-'}</td>
            </tr>
        `).join('');
    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="6" style="color:#CF222E; text-align:center;">Error loading database: ${escapeHtml(e.message)}</td></tr>`;
    }
}

// Direct Vector Search
async function submitSearch() {
    const input = document.getElementById('search-query');
    const query = input.value.trim();
    if (!query) return;

    const topK = document.getElementById('search-topk').value;
    const resultsContainer = document.getElementById('search-results');

    resultsContainer.innerHTML = '<p style="color:var(--secondary); font-weight:600;">Searching vector embeddings...</p>';

    try {
        const res = await fetch(`/api/rag/search?query=${encodeURIComponent(query)}&topK=${topK}`);
        const data = await res.json();

        if (data.length === 0) {
            resultsContainer.innerHTML = '<p style="color:var(--text-muted);">No matching vector chunks found.</p>';
            return;
        }

        resultsContainer.innerHTML = data.map((r, i) => `
            <div style="background:#F8FCFF; border:1px solid var(--border); border-radius:8px; padding:14px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <strong style="color:var(--primary);">#${i + 1} &bull; ${escapeHtml(r.document)} (Page ${r.page})</strong>
                    <span class="status-badge status-success">Score: ${Math.round(r.score * 100)}%</span>
                </div>
                <div style="font-size:0.8rem; color:var(--secondary); font-weight:600; margin-bottom:6px;">Section: ${escapeHtml(r.heading)}</div>
                <pre style="background:#FFFFFF; border:1px solid var(--border); padding:10px; border-radius:6px; font-size:0.82rem; white-space:pre-wrap; margin:0; color:var(--text-main);">${escapeHtml(r.text)}</pre>
            </div>
        `).join('');
    } catch (e) {
        resultsContainer.innerHTML = `<p style="color:#CF222E;">Search error: ${escapeHtml(e.message)}</p>`;
    }
}

async function loadVectorDbData() {
    const statusEl = document.getElementById('vdb-status');
    const collectionEl = document.getElementById('vdb-collection');
    const countEl = document.getElementById('vdb-count');
    const tbody = document.getElementById('vdb-body');

    try {
        const statusRes = await fetch('/api/rag/vector-db/status');
        if (statusRes.ok) {
            const status = await statusRes.json();
            if (status.chromaOnline) {
                statusEl.innerHTML = '<span style="color:#0A7A66; font-weight:700;">🟢 Online</span>';
            } else {
                statusEl.innerHTML = '<span style="color:var(--secondary); font-weight:700;" title="Running on local zero-copy vector store. To start ChromaDB: docker compose up -d">🟡 Embedded Local</span>';
            }
            collectionEl.textContent = status.collectionName || 'rag_documents';
            countEl.textContent = status.totalVectorsCount || 0;
        }
    } catch (e) {
        console.warn('Vector DB status error:', e);
    }

    if (tbody) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:16px;">Loading vector documents...</td></tr>';
        try {
            const docsRes = await fetch('/api/rag/vector-db/documents?limit=100');
            if (docsRes.ok) {
                const docs = await docsRes.json();
                if (docs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:16px; color:var(--text-muted);">No vectors indexed yet. Ingest documents from the Ingestion tab first.</td></tr>';
                    return;
                }
                tbody.innerHTML = docs.map(d => `
                    <tr>
                        <td><code style="font-size:0.75rem; color:var(--primary); font-weight:600;">${escapeHtml(d.id)}</code></td>
                        <td><strong>${escapeHtml(d.documentName)}</strong></td>
                        <td>${d.pageNumber}</td>
                        <td><span style="font-size:0.8rem; color:var(--secondary); font-weight:600;">${escapeHtml(d.heading)}</span></td>
                        <td><span class="status-badge status-success">128-dim</span></td>
                        <td><div style="max-width:350px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:0.8rem; color:var(--text-muted);">${escapeHtml(d.text)}</div></td>
                    </tr>
                `).join('');
            }
        } catch (e) {
            tbody.innerHTML = `<tr><td colspan="6" style="color:#CF222E; text-align:center;">Error loading vector documents: ${escapeHtml(e.message)}</td></tr>`;
        }
    }
}

function escapeHtml(text) {
    if (!text) return '';
    return text.toString()
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
