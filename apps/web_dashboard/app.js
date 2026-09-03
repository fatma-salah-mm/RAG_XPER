/**
 * XPER Enterprise Web Dashboard & Chat Application
 * Connects directly to RAG_XPER REST API with Client-Side & Server-Side Cache Memory.
 */

const API_BASE = window.location.origin;
let currentSessionId = "session_" + Date.now();
let selectedFile = null;

// Client Cache Memory (In-Browser Storage)
const clientCache = {
    get(key) {
        try {
            const raw = localStorage.getItem("rag_cache_" + key.trim().toLowerCase());
            if (!raw) return null;
            const item = JSON.parse(raw);
            if (Date.now() - item.time > 3600000) { // 1 Hour TTL
                localStorage.removeItem("rag_cache_" + key.trim().toLowerCase());
                return null;
            }
            return item.data;
        } catch (e) {
            return null;
        }
    },
    set(key, data) {
        try {
            localStorage.setItem("rag_cache_" + key.trim().toLowerCase(), JSON.stringify({
                time: Date.now(),
                data: data
            }));
        } catch (e) {}
    },
    clear() {
        Object.keys(localStorage).forEach(k => {
            if (k.startsWith("rag_cache_")) localStorage.removeItem(k);
        });
    }
};

// DOM Elements
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const messagesContainer = document.getElementById("messagesContainer");
const newChatBtn = document.getElementById("newChatBtn");
const langToggleBtn = document.getElementById("langToggleBtn");
const cacheBadge = document.getElementById("cacheBadge");

// View Elements
const tabChat = document.getElementById("tabChat");
const tabBooks = document.getElementById("tabBooks");
const tabStats = document.getElementById("tabStats");
const chatView = document.getElementById("chatView");
const booksView = document.getElementById("booksView");
const statsView = document.getElementById("statsView");

// Modal Elements
const uploadModal = document.getElementById("uploadModal");
const uploadModalBtn = document.getElementById("uploadModalBtn");
const quickAttachBtn = document.getElementById("quickAttachBtn");
const dropZone = document.getElementById("dropZone");
const fileUploadInput = document.getElementById("fileUploadInput");
const selectedFileName = document.getElementById("selectedFileName");
const submitUploadBtn = document.getElementById("submitUploadBtn");
const uploadProgressContainer = document.getElementById("uploadProgressContainer");
const uploadProgressBar = document.getElementById("uploadProgressBar");
const uploadPercent = document.getElementById("uploadPercent");
const uploadStatusText = document.getElementById("uploadStatusText");

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    setupEventListeners();
    fetchSystemStats();
});

function setupEventListeners() {
    // Input & Send
    sendBtn.addEventListener("click", handleSendMessage);
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // New Chat
    newChatBtn.addEventListener("click", resetChat);

    // Direction Toggle
    langToggleBtn.addEventListener("click", () => {
        const currentDir = document.documentElement.getAttribute("dir");
        if (currentDir === "rtl") {
            document.documentElement.setAttribute("dir", "ltr");
            langToggleBtn.innerText = "عربي";
        } else {
            document.documentElement.setAttribute("dir", "rtl");
            langToggleBtn.innerText = "EN";
        }
    });

    // Tab Switching
    tabChat.addEventListener("click", () => switchView("chat"));
    tabBooks.addEventListener("click", () => { switchView("books"); loadBooksList(); });
    tabStats.addEventListener("click", () => { switchView("stats"); fetchSystemStats(); });

    // Upload Modal
    uploadModalBtn.addEventListener("click", openUploadModal);
    quickAttachBtn.addEventListener("click", openUploadModal);
    dropZone.addEventListener("click", () => fileUploadInput.click());
    fileUploadInput.addEventListener("change", handleFileSelect);

    // Drag & Drop
    dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("border-xper-500", "bg-xper-50/40"); });
    dropZone.addEventListener("dragleave", () => { dropZone.classList.remove("border-xper-500", "bg-xper-50/40"); });
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("border-xper-500", "bg-xper-50/40");
        if (e.dataTransfer.files.length > 0) {
            selectedFile = e.dataTransfer.files[0];
            selectedFileName.innerText = `📄 ${selectedFile.name} (${Math.round(selectedFile.size / 1024)} KB)`;
        }
    });

    submitUploadBtn.addEventListener("click", handleUploadAndIngest);
}

function switchView(viewName) {
    [tabChat, tabBooks, tabStats].forEach(t => t.className = "flex-1 py-1.5 rounded-lg hover:text-xper-600 transition");
    [chatView, booksView, statsView].forEach(v => v.classList.add("hidden"));

    if (viewName === "chat") {
        tabChat.className = "flex-1 py-1.5 rounded-lg bg-white shadow-xs text-xper-700 transition font-bold";
        chatView.classList.remove("hidden");
    } else if (viewName === "books") {
        tabBooks.className = "flex-1 py-1.5 rounded-lg bg-white shadow-xs text-xper-700 transition font-bold";
        booksView.classList.remove("hidden");
    } else if (viewName === "stats") {
        tabStats.className = "flex-1 py-1.5 rounded-lg bg-white shadow-xs text-xper-700 transition font-bold";
        statsView.classList.remove("hidden");
    }
}

function sendQuickPrompt(promptText) {
    queryInput.value = promptText;
    handleSendMessage();
}

async function handleSendMessage() {
    const text = queryInput.value.trim();
    if (!text) return;

    queryInput.value = "";
    appendUserMessage(text);

    // Check Client Cache First
    const cachedData = clientCache.get(text);
    if (cachedData) {
        showCacheIndicator();
        renderAssistantMessage(cachedData, true);
        return;
    }

    // Append loading skeleton
    const loadingId = appendLoadingSkeleton();

    try {
        const startTime = Date.now();
        const res = await fetch(`${API_BASE}/v1/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: text, top_k: 6 })
        });

        const data = await res.json();
        removeLoadingSkeleton(loadingId);

        if (!res.ok) {
            appendErrorMessage(data.detail || "حدث خطأ أثناء معالجة السؤال.");
            return;
        }

        // Save to Client Cache
        clientCache.set(text, data);

        // Check if server indicated cache hit
        if (data.is_cached) showCacheIndicator();
        renderAssistantMessage(data, data.is_cached);

    } catch (err) {
        removeLoadingSkeleton(loadingId);
        appendErrorMessage("تعذر الاتصال بالخادم. تأكد من تشغيل الـ API.");
    }
}

function appendUserMessage(text) {
    const div = document.createElement("div");
    div.className = "flex justify-end animate-in";
    div.innerHTML = `
        <div class="bg-xper-600 text-white rounded-2xl rounded-tl-xs px-5 py-3 max-w-2xl text-sm leading-relaxed shadow-sm">
            ${escapeHtml(text)}
        </div>
    `;
    messagesContainer.appendChild(div);
    scrollToBottom();
}

function renderAssistantMessage(data, isCached = false) {
    const div = document.createElement("div");
    div.className = "flex items-start gap-3 max-w-3xl animate-in";

    let sourcesHtml = "";
    if (data.sources && data.sources.length > 0) {
        sourcesHtml = `
            <div class="mt-4 pt-3 border-t border-slate-100">
                <span class="text-[11px] font-bold text-slate-500 block mb-2">📑 المصادر والصفحات المستخرجة:</span>
                <div class="flex flex-wrap gap-2">
                    ${data.sources.map((s, idx) => `
                        <div class="bg-slate-50 hover:bg-slate-100 border border-slate-200/80 rounded-xl p-2.5 text-xs text-slate-700 transition flex-1 min-w-[200px]">
                            <div class="flex items-center justify-between font-bold text-xper-800 text-[11px] mb-1">
                                <span class="truncate max-w-[150px]">${s.source || "مستند"}</span>
                                <span class="bg-xper-100 text-xper-800 px-1.5 py-0.5 rounded text-[10px]">ص ${s.page || 1}</span>
                            </div>
                            <p class="text-[11px] text-slate-600 line-clamp-2">${escapeHtml(s.text)}</p>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    let reasoningHtml = "";
    if (data.reasoning) {
        reasoningHtml = `
            <details class="mb-3 bg-amber-50/60 border border-amber-200/60 rounded-xl p-3 text-xs text-amber-900 cursor-pointer">
                <summary class="font-bold flex items-center gap-1.5 select-none">
                    <span>💡 تحليل واستنتاج المساعد (Chain-of-Thought)</span>
                </summary>
                <div class="mt-2 pt-2 border-t border-amber-200/40 text-[11px] leading-relaxed whitespace-pre-wrap font-sans">
                    ${escapeHtml(data.reasoning)}
                </div>
            </details>
        `;
    }

    const cacheTag = isCached ? `<span class="inline-flex items-center gap-1 text-[10px] bg-emerald-100 text-emerald-800 font-bold px-2 py-0.5 rounded-full mb-2">⚡ تم الاسترجاع من الذاكرة المؤقتة (الكاش) في 3ms</span>` : '';

    const formattedAnswer = marked.parse(data.answer || "");

    div.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-xper-500 text-white flex items-center justify-center font-bold text-sm shrink-0 shadow-sm mt-1">
            ✦
        </div>
        <div class="bg-white border border-slate-200/80 rounded-2xl rounded-tr-xs p-5 shadow-xs text-slate-800 text-sm leading-relaxed flex-1">
            ${cacheTag}
            ${reasoningHtml}
            <div class="prose text-slate-800">${formattedAnswer}</div>
            ${sourcesHtml}
        </div>
    `;

    messagesContainer.appendChild(div);
    scrollToBottom();
}

function appendLoadingSkeleton() {
    const id = "loading_" + Date.now();
    const div = document.createElement("div");
    div.id = id;
    div.className = "flex items-start gap-3 max-w-md animate-pulse";
    div.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-xper-200 text-transparent flex items-center justify-center text-sm shrink-0">✦</div>
        <div class="bg-white border border-slate-200 rounded-2xl p-4 shadow-xs flex-1 space-y-2">
            <div class="h-3 bg-slate-200 rounded w-3/4"></div>
            <div class="h-3 bg-slate-100 rounded w-1/2"></div>
        </div>
    `;
    messagesContainer.appendChild(div);
    scrollToBottom();
    return id;
}

function removeLoadingSkeleton(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function appendErrorMessage(msg) {
    const div = document.createElement("div");
    div.className = "flex justify-center animate-in";
    div.innerHTML = `
        <div class="bg-rose-50 border border-rose-200 text-rose-700 text-xs px-4 py-2 rounded-xl">
            ⚠️ ${escapeHtml(msg)}
        </div>
    `;
    messagesContainer.appendChild(div);
    scrollToBottom();
}

function showCacheIndicator() {
    cacheBadge.classList.remove("hidden");
    setTimeout(() => cacheBadge.classList.add("hidden"), 3000);
}

function resetChat() {
    messagesContainer.innerHTML = `
        <div class="flex items-start gap-3 max-w-3xl">
            <div class="w-8 h-8 rounded-full bg-xper-500 text-white flex items-center justify-center font-bold text-sm shrink-0 shadow-sm mt-1">✦</div>
            <div class="bg-white border border-slate-200/80 rounded-2xl p-5 shadow-xs text-slate-700 leading-relaxed text-sm">
                <p class="font-bold text-slate-800 mb-1">مرحباً بك في محادثة جديدة! 📊</p>
                <p>تفضل بطرح سؤالك أو استفسارك وسأقوم بالإجابة بناءً على المستندات والبيانات المفهرسة.</p>
            </div>
        </div>
    `;
    currentSessionId = "session_" + Date.now();
}

// Upload Modal & Ingestion
function openUploadModal() { uploadModal.classList.remove("hidden"); }
function closeUploadModal() { uploadModal.classList.add("hidden"); resetUploadForm(); }

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        selectedFile = e.target.files[0];
        selectedFileName.innerText = `📄 ${selectedFile.name} (${Math.round(selectedFile.size / 1024)} KB)`;
        if (!document.getElementById("bookTitleInput").value) {
            document.getElementById("bookTitleInput").value = selectedFile.name.replace(/\.[^/.]+$/, "");
        }
    }
}

async function handleUploadAndIngest() {
    if (!selectedFile) {
        alert("يرجى اختيار ملف أولاً.");
        return;
    }

    const title = document.getElementById("bookTitleInput").value.trim() || selectedFile.name;
    const author = document.getElementById("bookAuthorInput").value.trim() || "";
    const category = document.getElementById("bookCategoryInput").value.trim() || "عام";
    const strategy = document.getElementById("chunkStrategySelect").value;

    uploadProgressContainer.classList.remove("hidden");
    uploadProgressBar.style.width = "20%";
    uploadPercent.innerText = "20%";
    uploadStatusText.innerText = "جارٍ رفع الملف إلى السيرفر...";
    submitUploadBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("strategy", strategy);
    formData.append("title", title);
    formData.append("author", author);
    formData.append("category", category);

    try {
        const res = await fetch(`${API_BASE}/v1/ingest/async`, {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "فشل رفع الملف.");

        const jobId = data.job_id;
        pollJobStatus(jobId, title, author, category);

    } catch (err) {
        alert("خطأ: " + err.message);
        submitUploadBtn.disabled = false;
        uploadProgressContainer.classList.add("hidden");
    }
}

function pollJobStatus(jobId, title, author, category) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/v1/jobs/${jobId}`);
            const data = await res.json();

            uploadProgressBar.style.width = `${data.progress}%`;
            uploadPercent.innerText = `${data.progress}%`;

            if (data.status === "completed") {
                clearInterval(interval);
                uploadStatusText.innerText = "✅ تمت الفهرسة بنجاح!";
                setTimeout(() => {
                    closeUploadModal();
                    loadBooksList();
                }, 1000);
            } else if (data.status === "failed") {
                clearInterval(interval);
                alert("فشلت المعالجة: " + (data.error || "خطأ غير معروف"));
                submitUploadBtn.disabled = false;
            }
        } catch (e) {
            clearInterval(interval);
            submitUploadBtn.disabled = false;
        }
    }, 1500);
}

function resetUploadForm() {
    selectedFile = null;
    fileUploadInput.value = "";
    selectedFileName.innerText = "اسحب الملف هنا أو انقر للاختيار";
    document.getElementById("bookTitleInput").value = "";
    document.getElementById("bookAuthorInput").value = "";
    document.getElementById("bookCategoryInput").value = "";
    uploadProgressContainer.classList.add("hidden");
    uploadProgressBar.style.width = "0%";
    submitUploadBtn.disabled = false;
}

// Load Books Catalog (MySQL)
async function loadBooksList() {
    const tbody = document.getElementById("booksTableBody");
    tbody.innerHTML = `<tr><td colspan="6" class="p-8 text-center text-slate-500">جارٍ جلب الكتب من قاعدة البيانات...</td></tr>`;

    try {
        const res = await fetch(`${API_BASE}/v1/books`);
        const books = await res.json();

        if (!books || books.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" class="p-8 text-center text-slate-500">لا توجد كتب مفهرسة حالياً. اضغط على "+ إضافة كتاب جديد" للبدء.</td></tr>`;
            return;
        }

        tbody.innerHTML = books.map(b => `
            <tr class="hover:bg-slate-50/80 transition">
                <td class="p-3.5 font-bold text-slate-800">
                    ${escapeHtml(b.title)}
                    <span class="block text-[10px] text-slate-400 font-normal">${escapeHtml(b.filename)}</span>
                </td>
                <td class="p-3.5">${escapeHtml(b.author || "—")} <span class="bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded text-[10px] mr-1">${escapeHtml(b.category || "عام")}</span></td>
                <td class="p-3.5 font-bold text-xper-700">${b.chunk_count || 0}</td>
                <td class="p-3.5"><span class="bg-slate-100 text-slate-700 px-2 py-0.5 rounded-full text-[10px] font-semibold">${b.strategy_used || "recursive"}</span></td>
                <td class="p-3.5"><span class="bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded-full text-[10px] font-bold">مفهرس</span></td>
                <td class="p-3.5 text-center">
                    <button onclick="deleteBookItem('${escapeHtml(b.filename)}')" class="text-rose-500 hover:text-rose-700 font-bold text-xs p-1">حذف</button>
                </td>
            </tr>
        `).join('');

    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="6" class="p-8 text-center text-rose-600">تعذر جلب قائمة الكتب من السيرفر.</td></tr>`;
    }
}

async function deleteBookItem(filename) {
    if (!confirm(`هل أنت متأكد من حذف المستند '${filename}'؟`)) return;
    try {
        const res = await fetch(`${API_BASE}/v1/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
        if (res.ok) {
            loadBooksList();
        }
    } catch (e) {
        alert("فشل الحذف.");
    }
}

// Telemetry & Stats
async function fetchSystemStats() {
    try {
        const res = await fetch(`${API_BASE}/metrics`);
        if (!res.ok) return;
        const stats = await res.json();
        document.getElementById("statQueries").innerText = stats.total_queries || 0;
        document.getElementById("statCacheHits").innerText = stats.cache_hits || 0;
        document.getElementById("statBM25").innerText = stats.bm25_indexed_chunks || 0;
    } catch (e) {}
}

function clearSystemCache() {
    clientCache.clear();
    alert("تم تفريغ الذاكرة المؤقتة (الكاش) بنجاح.");
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
