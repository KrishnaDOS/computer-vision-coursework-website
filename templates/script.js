const initialActiveTab = document.querySelector('.tab-content.active');
let activeTabId = initialActiveTab ? initialActiveTab.id : null;
const tabChangeListeners = [];
const globalResetHandlers = [];

function registerTabChangeListener(fn) {
    if (typeof fn === 'function') {
        tabChangeListeners.push(fn);
    }
}

function registerGlobalResetHandler(fn) {
    if (typeof fn === 'function') {
        globalResetHandlers.push(fn);
    }
}

async function runGlobalResetHandlers() {
    for (const handler of globalResetHandlers) {
        try {
            await handler();
        } catch (err) {
            console.warn('Global reset handler failed', err);
        }
    }
}

function notifyTabChangeListeners(prev, next) {
    tabChangeListeners.forEach((fn) => {
        try {
            fn(prev, next);
        } catch (err) {
            console.error('Tab change callback failed', err);
        }
    });
}

let module3LightboxOverlay = null;
let module3LightboxImage = null;

function ensureModule3Lightbox() {
    if (module3LightboxOverlay) return;
    module3LightboxOverlay = document.createElement('div');
    module3LightboxOverlay.className = 'module3-lightbox';
    module3LightboxOverlay.setAttribute('role', 'dialog');
    module3LightboxOverlay.setAttribute('aria-modal', 'true');

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'module3-lightbox__close';
    closeBtn.setAttribute('aria-label', 'Close preview');
    closeBtn.textContent = '✕';

    module3LightboxImage = document.createElement('img');
    module3LightboxImage.className = 'module3-lightbox__img';
    module3LightboxImage.alt = '';

    module3LightboxOverlay.appendChild(closeBtn);
    module3LightboxOverlay.appendChild(module3LightboxImage);
    document.body.appendChild(module3LightboxOverlay);

    closeBtn.addEventListener('click', closeModule3Lightbox);
    module3LightboxOverlay.addEventListener('click', (event) => {
        if (event.target === module3LightboxOverlay) {
            closeModule3Lightbox();
        }
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && module3LightboxOverlay.classList.contains('module3-lightbox--open')) {
            closeModule3Lightbox();
        }
    });
}

function closeModule3Lightbox() {
    if (!module3LightboxOverlay) return;
    module3LightboxOverlay.classList.remove('module3-lightbox--open');
    document.body.classList.remove('module3-lightbox-open');
}

function openModule3Lightbox(src, altText) {
    if (!src) return;
    ensureModule3Lightbox();
    if (!module3LightboxOverlay || !module3LightboxImage) return;
    module3LightboxImage.src = src;
    module3LightboxImage.alt = altText || 'Preview image';
    module3LightboxOverlay.classList.add('module3-lightbox--open');
    document.body.classList.add('module3-lightbox-open');
}

function markLightboxImageInteractive(img, groupName) {
    if (!img || !groupName) return;
    const alreadyConfigured = img.dataset.lightboxReady === 'true' && img.dataset.lightbox === groupName;
    if (!alreadyConfigured) {
        img.dataset.lightboxReady = 'true';
        img.setAttribute('data-lightbox', groupName);
        if (!img.hasAttribute('tabindex') || Number(img.getAttribute('tabindex')) < 0) {
            img.tabIndex = 0;
        }
        img.setAttribute('role', 'button');
        img.setAttribute('aria-haspopup', 'dialog');
    }
}

function markModule2ImageInteractive(img) {
    markLightboxImageInteractive(img, 'module2');
}

function markModule4ImageInteractive(img) {
    markLightboxImageInteractive(img, 'module4');
}

function initModule2LightboxBridge() {
    const module2Tab = document.getElementById('a2');
    if (!module2Tab) return;

    const activateLightbox = (event) => {
        const target = event.target;
        if (!target || typeof target.closest !== 'function') return;
        const img = target.closest('img[data-lightbox="module2"]');
        if (!img) return;
        if (event.type === 'keydown' && event.key !== 'Enter' && event.key !== ' ') {
            return;
        }
        if (event.type === 'keydown') {
            event.preventDefault();
        }
        openModule3Lightbox(img.src, img.alt || 'Module 2 preview image');
    };

    module2Tab.addEventListener('click', activateLightbox);
    module2Tab.addEventListener('keydown', activateLightbox);
}

document.addEventListener('DOMContentLoaded', initModule2LightboxBridge);

function initModule4LightboxBridge() {
    const module4Tab = document.getElementById('a4');
    if (!module4Tab) return;

    const handleLightbox = (event) => {
        const target = event.target;
        if (!target || typeof target.closest !== 'function') return;
        const img = target.closest('img[data-lightbox="module4"]');
        if (!img) return;
        if (event.type === 'keydown' && event.key !== 'Enter' && event.key !== ' ') {
            return;
        }
        if (event.type === 'keydown') {
            event.preventDefault();
        }
        openModule3Lightbox(img.src, img.alt || 'Module 4 panorama preview');
    };

    module4Tab.addEventListener('click', handleLightbox);
    module4Tab.addEventListener('keydown', handleLightbox);
}

document.addEventListener('DOMContentLoaded', initModule4LightboxBridge);

function switchTab(tabId) {
    const previousTab = activeTabId;
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    document.getElementById(tabId).classList.add('active');

    const buttons = Array.from(document.querySelectorAll('.tab-btn'));
    buttons.forEach(btn => {
        const isTarget = btn.getAttribute('onclick') && btn.getAttribute('onclick').includes(tabId);
        if (isTarget) {
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
            btn.setAttribute('tabindex', '0');
            try { btn.focus(); } catch (e) { }
        } else {
            btn.classList.remove('active');
            btn.setAttribute('aria-selected', 'false');
            btn.setAttribute('tabindex', '-1');
        }
    });

    activeTabId = tabId;
    if (previousTab !== tabId) {
        notifyTabChangeListeners(previousTab, tabId);
    }
}

(function attachTabKeyboard() {
    const tabButtons = Array.from(document.querySelectorAll('.tab-btn'));
    if (!tabButtons.length) return;

    function focusTab(index) {
        const btn = tabButtons[index];
        if (!btn) return;
        btn.click();
    }

    tabButtons.forEach((btn, idx) => {
        btn.addEventListener('keydown', (e) => {
            const key = e.key;
            if (key === 'ArrowRight') {
                e.preventDefault();
                focusTab((idx + 1) % tabButtons.length);
            } else if (key === 'ArrowLeft') {
                e.preventDefault();
                focusTab((idx - 1 + tabButtons.length) % tabButtons.length);
            } else if (key === 'Home') {
                e.preventDefault();
                focusTab(0);
            } else if (key === 'End') {
                e.preventDefault();
                focusTab(tabButtons.length - 1);
            } else if (key === 'Enter' || key === ' ') {
                e.preventDefault();
                btn.click();
            }
        });
    });
})();

/* ---------- Source explorer (collapsed panels) ---------- */
(function () {
    const sourcePanelsInitialized = new Set();

    function moduleIdFromPanel(panel) {
        if (!panel) return null;
        if (panel.dataset.module) return panel.dataset.module;
        if (panel.id && panel.id.startsWith('source-panel-')) {
            return panel.id.replace('source-panel-', '');
        }
        return null;
    }

    function setActiveSourceButton(moduleId, activeBtn) {
        if (!moduleId) return;
        const buttons = document.querySelectorAll(`.source-file[data-module="${moduleId}"]`);
        buttons.forEach((btn) => {
            const isActive = btn === activeBtn;
            btn.classList.toggle('active', isActive);
            btn.setAttribute('aria-pressed', String(isActive));
            if (isActive) {
                btn.setAttribute('aria-current', 'true');
            } else {
                btn.removeAttribute('aria-current');
            }
        });
    }

    function maybeLoadDefaultSource(panel) {
        if (!panel || !panel.classList.contains('source-panel')) return;
        const moduleId = moduleIdFromPanel(panel);
        if (!moduleId || sourcePanelsInitialized.has(moduleId)) return;
        const firstBtn = panel.querySelector('.source-file');
        if (!firstBtn) return;
        sourcePanelsInitialized.add(moduleId);
        handleFileSelection(firstBtn, { scrollIntoView: false });
    }

    function togglePanel(toggleBtn) {
        const panelId = toggleBtn.getAttribute('aria-controls');
        const panel = document.getElementById(panelId);
        const expanded = toggleBtn.getAttribute('aria-expanded') === 'true';

        toggleBtn.setAttribute('aria-expanded', String(!expanded));

        // Support both source and video toggle buttons. Choose the
        const isVideo = toggleBtn.classList.contains('video-toggle');
        const isInstructions = toggleBtn.classList.contains('instructions-toggle');
        // toggle open state using a CSS class so transitions (opacity/height)
        // can animate. Keep ARIA attributes in sync.
        if (expanded) {
            panel.classList.remove('open');
            panel.setAttribute('aria-hidden', 'true');
            toggleBtn.setAttribute('aria-expanded', 'false');
        } else {
            panel.classList.add('open');
            panel.setAttribute('aria-hidden', 'false');
            toggleBtn.setAttribute('aria-expanded', 'true');
            if (panel && panel.classList.contains('source-panel')) {
                maybeLoadDefaultSource(panel);
            }
        }

        let collapsedLabel = 'View CLI source code ▾';
        let expandedLabel = 'View CLI source code ▴';
        if (isVideo) {
            collapsedLabel = 'Watch the video ▾';
            expandedLabel = 'Watch the video ▴';
        } else if (isInstructions) {
            collapsedLabel = 'View module instructions ▾';
            expandedLabel = 'View module instructions ▴';
        }

        toggleBtn.innerText = expanded ? collapsedLabel : expandedLabel;
    }

    const LANGUAGE_MAP = {
        py: 'python',
        txt: 'plaintext',
        md: 'markdown'
    };

    function cleanupLanguageClasses(el) {
        el.classList.remove('hljs');
        Array.from(el.classList)
            .filter(cls => cls.startsWith('language-'))
            .forEach(cls => el.classList.remove(cls));
    }

    function handleFileSelection(btn, options = {}) {
        // Fetch the file from /source/<filename> and load it into the code pane.
        if (!btn) return;
        const moduleId = btn.dataset ? btn.dataset.module : null; // e.g. "a1"
        if (!moduleId) return;
        const codeEl = document.querySelector(`#source-code-${moduleId} code`);
        if (!codeEl) return;

        const { scrollIntoView = true } = options || {};

        const sourcePath = btn.dataset.sourcePath;
        if (!sourcePath) {
            codeEl.textContent = '[Error] No source path configured for this file.';
            setActiveSourceButton(moduleId, btn);
            return;
        }

        setActiveSourceButton(moduleId, btn);

        // Visual loading state
        cleanupLanguageClasses(codeEl);
        codeEl.textContent = 'Loading source…';
        if (scrollIntoView && codeEl.parentElement) {
            codeEl.parentElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        fetch(`/source/${encodeURIComponent(sourcePath)}`)
            .then(resp => {
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                return resp.text();
            })
            .then(text => {
                const extMatch = sourcePath.split('.').pop()?.toLowerCase() || '';
                const mappedLanguage = LANGUAGE_MAP[extMatch];

                cleanupLanguageClasses(codeEl);
                const highlightAvailable = typeof window !== 'undefined' && window.hljs;

                if (highlightAvailable && (mappedLanguage ? hljs.getLanguage(mappedLanguage) : true)) {
                    try {
                        let highlighted;
                        if (mappedLanguage && hljs.getLanguage(mappedLanguage)) {
                            highlighted = hljs.highlight(text, { language: mappedLanguage, ignoreIllegals: true }).value;
                            codeEl.innerHTML = highlighted;
                            codeEl.classList.add('hljs', `language-${mappedLanguage}`);
                        } else if (hljs.highlightAuto) {
                            highlighted = hljs.highlightAuto(text).value;
                            codeEl.innerHTML = highlighted;
                            codeEl.classList.add('hljs');
                        } else {
                            codeEl.textContent = text;
                        }
                    } catch (err) {
                        console.warn('Highlight.js failed, showing plain text.', err);
                        codeEl.textContent = text;
                    }
                } else {
                    codeEl.textContent = text;
                }

                // keep the loaded code visible and scroll to the top
                codeEl.parentElement.scrollTop = 0;

                // Populate the raw-view and download links in the same source panel
                try {
                    const panel = codeEl.closest('.source-panel');
                    if (panel) {
                        const viewBtn = panel.querySelector('.view-raw-btn');
                        const downloadBtn = panel.querySelector('.download-btn');
                        const rawUrl = `/source/${encodeURI(sourcePath)}`; // preserve path separators
                        const fileLabel = btn.dataset.filename || sourcePath.split('/').pop() || sourcePath;
                        if (viewBtn) {
                            viewBtn.href = rawUrl;
                            viewBtn.removeAttribute('aria-disabled');
                            viewBtn.removeAttribute('hidden');
                            const toolbar = panel.querySelector('.source-toolbar');
                            if (toolbar) {
                                toolbar.removeAttribute('hidden');
                                toolbar.setAttribute('aria-hidden', 'false');
                            }
                        }
                        if (downloadBtn) {
                            // Add a query param to force the server to send Content-Disposition
                            downloadBtn.href = rawUrl + '?download=1';
                            downloadBtn.setAttribute('download', fileLabel);
                            downloadBtn.removeAttribute('aria-disabled');
                            downloadBtn.removeAttribute('hidden');
                        }
                    }
                } catch (err) {
                    // non-fatal; toolbar actions are optional
                    console.warn('Failed to enable source toolbar', err);
                }
            })
            .catch(err => {
                codeEl.textContent = `[Error] Could not load source: ${err.message}`;
                try {
                    const panel = codeEl.closest('.source-panel');
                    if (panel) {
                        const toolbar = panel.querySelector('.source-toolbar');
                        if (toolbar) {
                            toolbar.setAttribute('hidden', '');
                            toolbar.setAttribute('aria-hidden', 'true');
                        }
                    }
                } catch (err) { /* ignore */ }
            });
    }

    document.addEventListener('click', (e) => {
        const t = e.target;
        if (t.classList && (t.classList.contains('source-toggle') || t.classList.contains('video-toggle') || t.classList.contains('instructions-toggle'))) {
            togglePanel(t);
            return;
        }

        if (t.classList && t.classList.contains('source-file')) {
            handleFileSelection(t);
            return;
        }
    });

    // keyboard support for toggles & files (Enter/Space)
    document.addEventListener('keydown', (e) => {
        const target = e.target;
        if (!target) return;

        if (target.classList && (target.classList.contains('source-toggle') || target.classList.contains('video-toggle') || target.classList.contains('instructions-toggle'))) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                togglePanel(target);
            }
        }

        if (target.classList && target.classList.contains('source-file')) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                handleFileSelection(target);
            }
        }
    });

})();

async function runModule(id) {
    const inputElem = document.getElementById(`input-${id}`);
    const outputElem = document.getElementById(`output-${id}`);
    const textData = inputElem.value;

    outputElem.innerText = "Processing on Azure VM...";

    try {
        const response = await fetch(`/api/${id}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: textData })
        });

        const data = await response.json();

        if (response.ok) {
            outputElem.innerText = data.result;
        } else {
            outputElem.innerText = "Error: " + (data.error || "Unknown server error");
        }
    } catch (error) {
        outputElem.innerText = "Network Error: Could not connect to Flask backend.\nIs app.py running?";
        console.error(error);
    }
}

document.addEventListener('DOMContentLoaded', initModule2Flow);
document.addEventListener('DOMContentLoaded', initModule2Part1Flow);
document.addEventListener('DOMContentLoaded', initModule2Part3Flow);
document.addEventListener('DOMContentLoaded', initModule3Flow);
document.addEventListener('DOMContentLoaded', initModule3Part5Showcase);
document.addEventListener('DOMContentLoaded', initModule4Flow);

(function () {
    const themeToggle = document.getElementById('theme-toggle');
    const iconEl = document.getElementById('theme-icon');

    const sunSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M12 4V2M12 22v-2M20 12h2M2 12h2M18.36 5.64l1.41-1.41M4.22 19.78l1.41-1.41M18.36 18.36l1.41 1.41M4.22 4.22l1.41 1.41M12 8a4 4 0 100 8 4 4 0 000-8z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    const moonSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';

    function updateIcon(theme) {
        if (theme === 'dark') {
            iconEl.innerHTML = sunSvg;
            themeToggle.setAttribute('aria-pressed', 'true');
        } else {
            iconEl.innerHTML = moonSvg;
            themeToggle.setAttribute('aria-pressed', 'false');
        }
    }

    function applyTheme(theme) {
        if (theme === 'dark') document.body.classList.add('dark');
        else document.body.classList.remove('dark');
        updateIcon(theme);
    }

    const stored = localStorage.getItem('site-theme');
    const initial = stored || 'dark';
    applyTheme(initial);

    themeToggle.addEventListener('click', () => {
        const isDark = document.body.classList.toggle('dark');
        const nextTheme = isDark ? 'dark' : 'light';
        localStorage.setItem('site-theme', nextTheme);
        updateIcon(nextTheme);
    });
})();

document.addEventListener('DOMContentLoaded', initModule1Flow);
document.addEventListener('DOMContentLoaded', initModule56Flow);
document.addEventListener('DOMContentLoaded', initModule56Part2Showcase);
document.addEventListener('DOMContentLoaded', initModule7Part1Flow);
document.addEventListener('DOMContentLoaded', initModule7Flow);
document.addEventListener('DOMContentLoaded', initGlobalReset);

function initModule56Flow() {
    const form = document.getElementById('module5-6-part2-form');
    if (!form) return;

    const fileInput = document.getElementById('module5-6-part2-video-input');
    const useSampleCheckbox = document.getElementById('module5-6-part2-use-sample');
    const runBtn = document.getElementById('module5-6-part2-run-btn');
    const resetBtn = document.getElementById('module5-6-part2-reset-btn');
    const statusLine = document.getElementById('module5-6-part2-status');
    const inputVideo = document.getElementById('module5-6-part2-input-video');
    const inputPlaceholder = document.getElementById('module5-6-part2-input-placeholder');
    const outputVideo = document.getElementById('module5-6-part2-output-video');
    const outputPlaceholder = document.getElementById('module5-6-part2-output-placeholder');
    const dictSelect = document.getElementById('module5-6-part2-dict');
    const paddingInput = document.getElementById('module5-6-part2-padding');

    if (!fileInput || !runBtn || !resetBtn || !statusLine || !inputVideo || !outputVideo) return;

    let hasResults = false;
    let createdBlobUrl = null;
    let samplePreviewToken = 0;

    const setStatus = (message, variant = 'info') => {
        statusLine.textContent = message;
        statusLine.dataset.variant = variant;
    };

    const clearVideoElement = (vidEl, placeholder) => {
        if (!vidEl) return;
        try { vidEl.pause(); } catch (e) { /* ignore invalid states */ }
        try { vidEl.removeAttribute('src'); } catch (e) { /* ignore */ }
        // Remove any dynamically injected <source> tags so the element fully resets
        try {
            Array.from(vidEl.querySelectorAll('source')).forEach(src => src.remove());
        } catch (e) { /* ignore */ }
        try { vidEl.load(); } catch (e) { /* ignore */ }
        vidEl.hidden = true;
        if (placeholder) placeholder.hidden = false;
    };

    const showVideo = (vidEl, placeholder, dataUrlOrBlobUrl) => {
        if (!dataUrlOrBlobUrl) {
            clearVideoElement(vidEl, placeholder);
            return;
        }
        // set a proper <source> child so the MIME type can be respected
        try {
            // Make sure we clear previous sources
            Array.from(vidEl.querySelectorAll('source')).forEach(s => s.remove());
            // clear previous inline src if present
            try { vidEl.removeAttribute('src'); } catch (e) { /* ignore */ }
            const srcEl = document.createElement('source');
            srcEl.src = dataUrlOrBlobUrl;
            // if this is a data URL, we can extract the mime
            if (typeof dataUrlOrBlobUrl === 'string' && dataUrlOrBlobUrl.startsWith('data:')) {
                const mime = dataUrlOrBlobUrl.split(';')[0].replace('data:', '') || 'video/mp4';
                srcEl.type = mime;
            }
            vidEl.appendChild(srcEl);
        } catch (err) {
            // fall back to setting src directly
            try { vidEl.src = dataUrlOrBlobUrl; } catch (_) { /* ignore */ }
        }
        vidEl.hidden = false;
        if (placeholder) placeholder.hidden = true;
        try { vidEl.load(); } catch (e) { /* ignore */ }
    };

    const hasSelected = () => Boolean((fileInput.files && fileInput.files.length) || (useSampleCheckbox && useSampleCheckbox.checked));

    fileInput.addEventListener('change', () => {
        const file = fileInput.files?.[0];
        if (!file) {
            runBtn.disabled = !hasSelected();
            return;
        }
        // create a local blob url for preview
        const url = URL.createObjectURL(file);
        showVideo(inputVideo, inputPlaceholder, url);
        runBtn.disabled = false;
        resetBtn.disabled = false;
        setStatus('Ready to run the tracker on the selected video.', 'info');
    });

    if (useSampleCheckbox) {
        useSampleCheckbox.addEventListener('change', async () => {
            const checked = useSampleCheckbox.checked;
            const requestToken = ++samplePreviewToken;

            if (checked) {
                fileInput.value = '';
                fileInput.disabled = true;
                runBtn.disabled = false;
                resetBtn.disabled = false;
                setStatus('Loading sample video preview…', 'info');
                try {
                    const resp = await fetch('/api/a56/part1/sample');
                    const data = await resp.json();
                    if (!resp.ok) throw new Error(data.error || 'Failed to load sample');
                    if (requestToken !== samplePreviewToken) return; // user toggled off mid-fetch
                    showVideo(inputVideo, inputPlaceholder, data.video);
                    setStatus(`Using sample ${data.filename}. Ready to run.`, 'info');
                } catch (err) {
                    if (requestToken !== samplePreviewToken) return;
                    fileInput.disabled = false;
                    useSampleCheckbox.checked = false;
                    setStatus(`Could not load sample: ${err.message}`, 'error');
                }
            } else {
                fileInput.disabled = false;
                clearVideoElement(inputVideo, inputPlaceholder);
                runBtn.disabled = !hasSelected();
                resetBtn.disabled = !hasResults;
                setStatus('Choose a video to get started.', 'info');
            }
        });
    }

        resetBtn.addEventListener('click', () => {
        form.reset();
        clearVideoElement(inputVideo, inputPlaceholder);
        clearVideoElement(outputVideo, outputPlaceholder);
        runBtn.disabled = true;
        resetBtn.disabled = true;
        hasResults = false;
            // hide download link and revoke any created blob url
            const dlArea = document.getElementById('module5-6-part2-download-area');
            const dlLink = document.getElementById('module5-6-part2-download-link');
            if (dlArea) dlArea.style.display = 'none';
            if (dlLink) dlLink.href = '#';
            if (createdBlobUrl) {
                try { URL.revokeObjectURL(createdBlobUrl); } catch (_) { /* ignore */ }
                createdBlobUrl = null;
            }
        setStatus('Choose a video to get started.', 'info');
    });

    form.addEventListener('submit', async (ev) => {
        ev.preventDefault();
        if (!hasSelected()) {
            setStatus('Please select a video or choose the example video.', 'error');
            return;
        }

        const formData = new FormData();
        if (useSampleCheckbox && useSampleCheckbox.checked) {
            formData.append('sample', 'aruco-marker.mp4');
        } else {
            formData.append('video', fileInput.files[0]);
        }
        formData.append('dict', dictSelect.value);
        formData.append('padding', paddingInput.value);

        runBtn.disabled = true;
        setStatus('Uploading video and running the ArUco tracker…', 'info');
        clearVideoElement(outputVideo, outputPlaceholder);
        // hide existing download link and revoke any previously created blob
        const dlArea = document.getElementById('module5-6-part2-download-area');
        const dlLink = document.getElementById('module5-6-part2-download-link');
        if (dlArea) dlArea.style.display = 'none';
        if (dlLink) dlLink.href = '#';
        if (createdBlobUrl) {
            try { URL.revokeObjectURL(createdBlobUrl); } catch (_) { /* ignore */ }
            createdBlobUrl = null;
        }

        try {
            const response = await fetch('/api/a56/part1', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Process failed');

            if (data.outputVideo) {
                // If we received a data URL, convert it to a blob URL for more compatibility
                try {
                    let videoUrl = data.outputVideo;
                        if (typeof videoUrl === 'string' && videoUrl.startsWith('data:')) {
                        const parts = videoUrl.split(',');
                        const meta = parts[0];
                        const base64 = parts[1];
                        const mimeMatch = meta.match(/data:([^;]+)/);
                        const mime = mimeMatch ? mimeMatch[1] : 'video/mp4';
                        const binary = atob(base64);
                        const len = binary.length;
                        const u8 = new Uint8Array(len);
                        for (let i = 0; i < len; i++) u8[i] = binary.charCodeAt(i);
                        const blob = new Blob([u8], { type: mime });
                        // revoke a previous blob url if present
                        if (createdBlobUrl) try { URL.revokeObjectURL(createdBlobUrl); } catch (_) { /* ignore */ }
                        videoUrl = URL.createObjectURL(blob);
                        createdBlobUrl = videoUrl;
                        // show the download link for the blob (create an object URL for download) — we'll reuse videoUrl
                        const dlArea = document.getElementById('module5-6-part2-download-area');
                        const dlLink = document.getElementById('module5-6-part2-download-link');
                        if (dlArea && dlLink) {
                            dlLink.href = videoUrl;
                            dlLink.download = data.outputFilename || 'processed.mp4';
                            dlArea.style.display = 'block';
                        }
                    }
                    showVideo(outputVideo, outputPlaceholder, videoUrl);
                } catch (err) {
                    console.warn('Failed to convert data URL to blob; using raw URL fallback', err);
                    showVideo(outputVideo, outputPlaceholder, data.outputVideo);
                }
                setStatus(`Process complete for ${data.originalFilename || 'uploaded video'}.`, 'success');
                hasResults = true;
            } else {
                setStatus('Processing completed but no output video was returned.', 'warning');
            }
            resetBtn.disabled = false;
        } catch (err) {
            setStatus(err.message || 'Unexpected error occurred.', 'error');
        } finally {
            runBtn.disabled = false;
        }
    });
}

function initModule56Part2Showcase() {
    const section = document.getElementById('module5-6-part3');
    const originalVideo = document.getElementById('module5-6-part3-original-video');
    const overlayVideo = document.getElementById('module5-6-part3-overlay-video');
    const overlayPlayer = document.getElementById('module5-6-part3-overlay-player');
    const overlayCanvas = document.getElementById('module5-6-part3-overlay-canvas');
    const statusLine = document.getElementById('module5-6-part3-status');
    const reloadBtn = document.getElementById('module5-6-part3-reload');
    const originalPlaceholder = document.getElementById('module5-6-part3-original-placeholder');
    const overlayHint = document.getElementById('module5-6-part3-overlay-hint');
    const frameLabel = document.getElementById('module5-6-part3-frame');
    const boxLabel = document.getElementById('module5-6-part3-box');
    const framesMetric = document.getElementById('module5-6-part3-metric-frames');
    const resolutionMetric = document.getElementById('module5-6-part3-metric-resolution');
    const fpsMetric = document.getElementById('module5-6-part3-metric-fps');
    const durationMetric = document.getElementById('module5-6-part3-metric-duration');
    const maskNote = document.getElementById('module5-6-part3-mask-note');

    if (!section || !originalVideo || !overlayVideo || !overlayCanvas || !statusLine || !overlayPlayer) {
        return;
    }

    const overlayCtx = overlayCanvas.getContext('2d');
    if (!overlayCtx) {
        return;
    }
    const originalFrame = originalVideo.closest('.result-image');
    let boxes = [];
    let fps = 30;
    let videoFrameCount = 0;
    let maskFrameCount = 0;
    let assetsLoaded = false;
    let isLoading = false;
    let overlayReady = false;
    let animationFrameId = null;

    const setStatus = (message, variant = 'info') => {
        statusLine.textContent = message;
        statusLine.dataset.variant = variant;
    };

    const clampFrameIndex = () => {
        if (!boxes.length || !overlayVideo) return 0;
        const idx = Math.round((overlayVideo.currentTime || 0) * fps);
        const maxIdx = boxes.length - 1;
        return Math.min(Math.max(idx, 0), maxIdx >= 0 ? maxIdx : 0);
    };

    const stopAnimation = () => {
        if (animationFrameId !== null) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
    };

    const drawFrame = () => {
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        if (!overlayReady || !boxes.length) {
            frameLabel.textContent = '--';
            boxLabel.textContent = '--';
            return;
        }

        const idx = clampFrameIndex();
        const entry = boxes[idx] || null;
        const frameTotal = videoFrameCount || boxes.length;
        frameLabel.textContent = frameTotal ? `${idx + 1} / ${frameTotal}` : `${idx + 1}`;

        if (!entry || !entry.box) {
            boxLabel.textContent = 'No mask';
            return;
        }

        const [x1, y1, x2, y2] = entry.box;
        const width = Math.max(0, x2 - x1);
        const height = Math.max(0, y2 - y1);
        const strokeWidth = Math.max(2, overlayCanvas.width * 0.0035);

        overlayCtx.lineWidth = strokeWidth;
        overlayCtx.strokeStyle = '#29d3d3';
        overlayCtx.fillStyle = 'rgba(41, 211, 211, 0.22)';
        overlayCtx.strokeRect(x1, y1, width, height);
        overlayCtx.fillRect(x1, y1, width, height);

        // subtle corner markers for easier spatial awareness
        overlayCtx.strokeStyle = '#0df0ff';
        overlayCtx.lineWidth = Math.max(1, strokeWidth * 0.6);
        const marker = Math.min(24, width * 0.2, height * 0.2);
        if (marker > 0) {
            overlayCtx.beginPath();
            overlayCtx.moveTo(x1, y1 + marker);
            overlayCtx.lineTo(x1, y1);
            overlayCtx.lineTo(x1 + marker, y1);
            overlayCtx.moveTo(x2 - marker, y1);
            overlayCtx.lineTo(x2, y1);
            overlayCtx.lineTo(x2, y1 + marker);
            overlayCtx.moveTo(x1, y2 - marker);
            overlayCtx.lineTo(x1, y2);
            overlayCtx.lineTo(x1 + marker, y2);
            overlayCtx.moveTo(x2 - marker, y2);
            overlayCtx.lineTo(x2, y2);
            overlayCtx.lineTo(x2, y2 - marker);
            overlayCtx.stroke();
        }

        boxLabel.textContent = `${width}px × ${height}px`;
    };

    const startAnimation = () => {
        stopAnimation();
        const tick = () => {
            drawFrame();
            animationFrameId = requestAnimationFrame(tick);
        };
        animationFrameId = requestAnimationFrame(tick);
    };

    const resetVideos = () => {
        stopAnimation();
        overlayReady = false;
        frameLabel.textContent = '--';
        boxLabel.textContent = '--';
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        overlayCanvas.width = overlayCanvas.height = 0;
        overlayVideo.removeAttribute('src');
        originalVideo.removeAttribute('src');
        try { overlayVideo.load(); } catch (e) { /* ignore */ }
        try { originalVideo.load(); } catch (e) { /* ignore */ }
        overlayPlayer.classList.remove('ready');
        if (overlayHint) {
            overlayHint.textContent = 'Loading overlay…';
            overlayHint.classList.remove('ready');
            overlayHint.hidden = false;
        }
        if (originalFrame) originalFrame.dataset.empty = 'true';
        originalVideo.hidden = true;
        if (originalPlaceholder) originalPlaceholder.hidden = false;
    };

    const populateMetrics = (data) => {
        framesMetric.textContent = `${data.maskFrameCount ?? '--'} / ${data.frameCount ?? '--'}`;
        resolutionMetric.textContent = data.frameWidth && data.frameHeight ? `${data.frameWidth} × ${data.frameHeight}` : '-- × --';
        fpsMetric.textContent = data.fps ? `${Number(data.fps).toFixed(2)} fps` : '-- fps';
        durationMetric.textContent = data.durationSeconds ? `${Number(data.durationSeconds).toFixed(2)} s` : '-- s';
        if (maskNote) {
            const maskName = data.maskFilename || 'iphone-moving-masks.npz';
            maskNote.innerHTML = `Masks: <code>${maskName}</code>`;
        }
    };

    const handleLoadedMetadata = () => {
        const width = overlayVideo.videoWidth || 0;
        const height = overlayVideo.videoHeight || 0;
        if (!width || !height) return;
        overlayCanvas.width = width;
        overlayCanvas.height = height;
        overlayReady = true;
        overlayPlayer.classList.add('ready');
        if (overlayHint) {
            overlayHint.textContent = 'Press play to visualize the SAM2 track.';
            overlayHint.classList.add('ready');
            overlayHint.hidden = false;
        }
        drawFrame();
    };

    overlayVideo.addEventListener('loadedmetadata', handleLoadedMetadata);
    overlayVideo.addEventListener('play', () => {
        if (!overlayReady) return;
        startAnimation();
    });
    overlayVideo.addEventListener('pause', stopAnimation);
    overlayVideo.addEventListener('ended', stopAnimation);
    overlayVideo.addEventListener('seeking', () => drawFrame());
    overlayVideo.addEventListener('timeupdate', () => {
        if (overlayVideo.paused) {
            drawFrame();
        }
    });

    if (originalVideo) {
        originalVideo.addEventListener('loadeddata', () => {
            originalVideo.hidden = false;
            if (originalPlaceholder) originalPlaceholder.hidden = true;
            if (originalFrame) originalFrame.dataset.empty = 'false';
        });
    }

    const loadAssets = async () => {
        if (isLoading) return;
        isLoading = true;
        reloadBtn && (reloadBtn.disabled = true);
        setStatus('Loading SAM2 assets…', 'info');
        resetVideos();

        try {
            const response = await fetch('/api/a56/part2/assets');
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to load assets');

            boxes = Array.isArray(data.boxes) ? data.boxes : [];
            fps = Number(data.fps) || 30;
            videoFrameCount = Number(data.frameCount) || boxes.length;
            maskFrameCount = Number(data.maskFrameCount) || boxes.length;
            populateMetrics(data);

            const videoUrl = data.videoUrl;
            if (!videoUrl) {
                throw new Error('Video URL missing in response');
            }

            overlayVideo.src = videoUrl;
            originalVideo.src = videoUrl;
            overlayVideo.load();
            originalVideo.load();
            assetsLoaded = true;
            setStatus('Assets ready. Scrub or press play to inspect the tracker.', 'success');
        } catch (err) {
            boxes = [];
            assetsLoaded = false;
            setStatus(err.message || 'Failed to load Module 5 & 6 Part 2 assets.', 'error');
        } finally {
            isLoading = false;
            if (reloadBtn) reloadBtn.disabled = false;
        }
    };

    reloadBtn && reloadBtn.addEventListener('click', loadAssets);

    registerTabChangeListener((prev, next) => {
        if (prev === 'a56' && overlayVideo) {
            try { overlayVideo.pause(); } catch (e) { /* ignore */ }
        }
        if (next === 'a56' && !assetsLoaded && !isLoading) {
            loadAssets();
        }
    });

    loadAssets();
}

function initModule7Part1Flow() {
    const refCanvas = document.getElementById('module7-ref-canvas');
    const leftCanvas = document.getElementById('module7-left-canvas');
    const rightCanvas = document.getElementById('module7-right-canvas');
    const measureCanvas = document.getElementById('module7-measure-canvas');
    if (!refCanvas || !leftCanvas || !rightCanvas || !measureCanvas) return;

    const ctxRef = refCanvas.getContext('2d');
    const ctxLeft = leftCanvas.getContext('2d');
    const ctxRight = rightCanvas.getContext('2d');
    const ctxMeasure = measureCanvas.getContext('2d');
    if (!ctxRef || !ctxLeft || !ctxRight || !ctxMeasure) return;

    const elements = {
        exampleBtn: document.getElementById('module7-example-btn'),
        ref: {
            canvas: refCanvas,
            ctx: ctxRef,
            wrapper: refCanvas.closest('.canvas-wrapper'),
            empty: document.getElementById('module7-ref-empty'),
            input: document.getElementById('module7-ref-input'),
            clear: document.getElementById('module7-ref-clear'),
            reset: document.getElementById('module7-ref-reset'),
            realWidth: document.getElementById('module7-ref-real-width'),
            distance: document.getElementById('module7-ref-distance'),
            pixelLabel: document.getElementById('module7-ref-px'),
            calcBtn: document.getElementById('module7-ref-calc'),
            status: document.getElementById('module7-ref-status'),
            card: document.getElementById('module7-calibration-card')
        },
        stereo: {
            card: document.getElementById('module7-stereo-card'),
            left: {
                canvas: leftCanvas,
                ctx: ctxLeft,
                wrapper: leftCanvas.closest('.canvas-wrapper'),
                empty: document.getElementById('module7-left-empty'),
                input: document.getElementById('module7-left-input'),
                clear: document.getElementById('module7-left-clear'),
                label: document.getElementById('module7-left-point')
            },
            right: {
                canvas: rightCanvas,
                ctx: ctxRight,
                wrapper: rightCanvas.closest('.canvas-wrapper'),
                empty: document.getElementById('module7-right-empty'),
                input: document.getElementById('module7-right-input'),
                clear: document.getElementById('module7-right-clear'),
                label: document.getElementById('module7-right-point')
            },
            baseline: document.getElementById('module7-baseline'),
            disparityLabel: document.getElementById('module7-disparity'),
            calcBtn: document.getElementById('module7-depth-btn'),
            status: document.getElementById('module7-stereo-status')
        },
        measure: {
            card: document.getElementById('module7-measure-card'),
            canvas: measureCanvas,
            ctx: ctxMeasure,
            wrapper: measureCanvas.closest('.canvas-wrapper'),
            empty: document.getElementById('module7-measure-empty'),
            resetBtn: document.getElementById('module7-measure-reset'),
            list: document.getElementById('module7-measurements-list'),
            status: document.getElementById('module7-measure-status'),
            summary: {
                focal: document.getElementById('module7-summary-focal'),
                depth: document.getElementById('module7-summary-depth'),
                disparity: document.getElementById('module7-summary-disparity'),
                baseline: document.getElementById('module7-summary-baseline')
            }
        }
    };

    if (!elements.ref.input || !elements.ref.calcBtn || !elements.stereo.calcBtn || !elements.measure.list) {
        return;
    }

    const MAX_WIDTH = 960;
    const MAX_HEIGHT = 720;
    const numberFormat = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

    const state = {
        focalLength: null,
        reference: {
            image: null,
            meta: null,
            points: [],
            pixelWidth: null,
            realWidth: null,
            distance: null
        },
        stereo: {
            left: { image: null, meta: null, point: null },
            right: { image: null, meta: null, point: null },
            disparity: null,
            baseline: null,
            depth: null
        },
        measurement: {
            pendingPoints: [],
            segments: [],
            isProcessing: false
        }
    };

    function calculateDisplaySize(img) {
        const widthRatio = MAX_WIDTH / img.width;
        const heightRatio = MAX_HEIGHT / img.height;
        const scale = Math.min(1, widthRatio, heightRatio);
        return {
            displayWidth: Math.round(img.width * scale),
            displayHeight: Math.round(img.height * scale)
        };
    }

    function formatNumber(value, digits = 2) {
        if (value === null || value === undefined) return null;
        const num = Number(value);
        if (!Number.isFinite(num)) return null;
        return num.toFixed(digits);
    }

    function setStatus(stage, message, variant = 'info') {
        const target = stage === 'ref' ? elements.ref.status : stage === 'stereo' ? elements.stereo.status : elements.measure.status;
        if (!target) return;
        target.textContent = message;
        target.dataset.variant = variant;
    }

    function updateSummary() {
        if (elements.measure.summary.focal) {
            elements.measure.summary.focal.textContent = state.focalLength ? numberFormat.format(state.focalLength) : '--';
        }
        if (elements.measure.summary.depth) {
            elements.measure.summary.depth.textContent = state.stereo.depth ? numberFormat.format(state.stereo.depth) : '--';
        }
        if (elements.measure.summary.disparity) {
            elements.measure.summary.disparity.textContent = state.stereo.disparity ? numberFormat.format(state.stereo.disparity) : '--';
        }
        if (elements.measure.summary.baseline) {
            elements.measure.summary.baseline.textContent = state.stereo.baseline ? numberFormat.format(state.stereo.baseline) : '--';
        }
    }

    function updateReferencePixelLabel() {
        const value = state.reference.pixelWidth;
        elements.ref.pixelLabel.textContent = value ? `${numberFormat.format(value)} px` : '--';
    }

    function updateStereoMetrics() {
        if (elements.stereo.left.label) {
            const pt = state.stereo.left.point;
            elements.stereo.left.label.textContent = pt ? `${numberFormat.format(pt.image.x)} , ${numberFormat.format(pt.image.y)}` : '--';
        }
        if (elements.stereo.right.label) {
            const pt = state.stereo.right.point;
            elements.stereo.right.label.textContent = pt ? `${numberFormat.format(pt.image.x)} , ${numberFormat.format(pt.image.y)}` : '--';
        }
        if (elements.stereo.disparityLabel) {
            elements.stereo.disparityLabel.textContent = state.stereo.disparity ? `${numberFormat.format(state.stereo.disparity)} px` : '--';
        }
        updateSummary();
    }

    function drawReferenceCanvas() {
        const bucket = state.reference;
        if (!bucket.image || !bucket.meta) return;
        const { ctx, canvas } = elements.ref;
        canvas.width = bucket.meta.displayWidth;
        canvas.height = bucket.meta.displayHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(bucket.image, 0, 0, bucket.meta.displayWidth, bucket.meta.displayHeight);

        bucket.points.forEach((point, idx) => {
            ctx.fillStyle = '#ff3b81';
            ctx.strokeStyle = '#0f1118';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(point.canvas.x, point.canvas.y, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = '#111827';
            ctx.font = '12px Inter, sans-serif';
            ctx.fillText(idx === 0 ? 'A' : 'B', point.canvas.x + 8, point.canvas.y - 8);
        });

        if (bucket.points.length === 2) {
            ctx.strokeStyle = '#31c48d';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(bucket.points[0].canvas.x, bucket.points[0].canvas.y);
            ctx.lineTo(bucket.points[1].canvas.x, bucket.points[1].canvas.y);
            ctx.stroke();
        }
    }

    function drawStereoCanvas(side) {
        const bucket = state.stereo[side];
        const el = elements.stereo[side];
        if (!bucket || !bucket.image || !bucket.meta || !el) return;
        el.canvas.width = bucket.meta.displayWidth;
        el.canvas.height = bucket.meta.displayHeight;
        el.ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
        el.ctx.drawImage(bucket.image, 0, 0, bucket.meta.displayWidth, bucket.meta.displayHeight);

        if (bucket.point) {
            el.ctx.fillStyle = '#f97316';
            el.ctx.strokeStyle = '#111827';
            el.ctx.lineWidth = 2;
            el.ctx.beginPath();
            el.ctx.arc(bucket.point.canvas.x, bucket.point.canvas.y, 7, 0, Math.PI * 2);
            el.ctx.fill();
            el.ctx.stroke();
        }
    }

    function drawMeasurementCanvas() {
        const meta = state.stereo.left.meta;
        const img = state.stereo.left.image;
        const { canvas, ctx, wrapper, empty } = elements.measure;
        if (!img || !meta || !state.stereo.depth) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            wrapper.dataset.state = 'empty';
            if (empty) empty.textContent = 'Calculate stereo depth to unlock the measurement canvas.';
            return;
        }

        if (canvas.width !== meta.displayWidth || canvas.height !== meta.displayHeight) {
            canvas.width = meta.displayWidth;
            canvas.height = meta.displayHeight;
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, meta.displayWidth, meta.displayHeight);
        wrapper.dataset.state = 'ready';
        if (empty) empty.textContent = 'Click two points (A → B) to log a measurement.';

        state.measurement.segments.forEach((segment, index) => {
            ctx.strokeStyle = '#06b6d4';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(segment.points[0].x, segment.points[0].y);
            ctx.lineTo(segment.points[1].x, segment.points[1].y);
            ctx.stroke();

            const midX = (segment.points[0].x + segment.points[1].x) / 2;
            const midY = (segment.points[0].y + segment.points[1].y) / 2;
            ctx.fillStyle = '#0f172a';
            ctx.font = '12px Inter, sans-serif';
            ctx.fillText(`#${index + 1} ${numberFormat.format(segment.realSize)} cm`, midX + 8, midY - 8);
        });

        state.measurement.pendingPoints.forEach((point, idx) => {
            ctx.fillStyle = idx === 0 ? '#d946ef' : '#facc15';
            ctx.strokeStyle = '#0f1118';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(point.canvas.x, point.canvas.y, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        });

        if (state.measurement.pendingPoints.length === 2) {
            const [a, b] = state.measurement.pendingPoints;
            ctx.setLineDash([6, 6]);
            ctx.strokeStyle = '#fbbf24';
            ctx.beginPath();
            ctx.moveTo(a.canvas.x, a.canvas.y);
            ctx.lineTo(b.canvas.x, b.canvas.y);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    function resetReferencePoints() {
        state.reference.points = [];
        state.reference.pixelWidth = null;
        elements.ref.reset.disabled = true;
        updateReferencePixelLabel();
        drawReferenceCanvas();
        setStatus('ref', 'Click the LEFT and RIGHT edges of the reference object.', 'info');
    }

    function resetReferenceImage() {
        state.reference.image = null;
        state.reference.meta = null;
        resetReferencePoints();
        elements.ref.canvas.width = 0;
        elements.ref.canvas.height = 0;
        if (elements.ref.wrapper) elements.ref.wrapper.dataset.state = 'empty';
        if (elements.ref.empty) elements.ref.empty.textContent = 'Upload the reference image to begin calibration.';
        elements.ref.input.value = '';
        elements.ref.clear.disabled = true;
        state.reference.realWidth = null;
        state.reference.distance = null;
        elements.ref.realWidth.value = '';
        elements.ref.distance.value = '';
    }

    function resetStereoSide(side, { clearInput = true } = {}) {
        const bucket = state.stereo[side];
        const el = elements.stereo[side];
        bucket.image = null;
        bucket.meta = null;
        bucket.point = null;
        if (clearInput) el.input.value = '';
        el.label.textContent = '--';
        el.canvas.width = 0;
        el.canvas.height = 0;
        if (el.wrapper) el.wrapper.dataset.state = 'empty';
        if (el.empty) el.empty.textContent = side === 'left' ? 'Load the LEFT stereo frame, then click a target point.' : 'Load the RIGHT frame to continue.';
    }

    function resetStereoWorkflow() {
        resetStereoSide('left');
        resetStereoSide('right');
        state.stereo.disparity = null;
        state.stereo.baseline = null;
        state.stereo.depth = null;
        elements.stereo.baseline.value = '';
        elements.stereo.calcBtn.disabled = true;
        elements.stereo.baseline.disabled = true;
        elements.stereo.left.input.disabled = true;
        elements.stereo.right.input.disabled = true;
        elements.stereo.left.clear.disabled = true;
        elements.stereo.right.clear.disabled = true;
        if (elements.stereo.card) {
            elements.stereo.card.classList.add('locked');
            elements.stereo.card.setAttribute('aria-disabled', 'true');
        }
        updateStereoMetrics();
        lockMeasurementStep();
        setStatus('stereo', 'Finish Step 1 to unlock the stereo depth tools.', 'info');
    }

    function lockMeasurementStep() {
        state.measurement.pendingPoints = [];
        state.measurement.segments = [];
        state.measurement.isProcessing = false;
        elements.measure.resetBtn.disabled = true;
        if (elements.measure.card) {
            elements.measure.card.classList.add('locked');
            elements.measure.card.setAttribute('aria-disabled', 'true');
        }
        renderMeasurements();
        drawMeasurementCanvas();
        setStatus('measure', 'Calculate stereo depth to unlock the measurement canvas.', 'info');
    }

    function unlockStereoStep() {
        if (elements.stereo.card) {
            elements.stereo.card.classList.remove('locked');
            elements.stereo.card.removeAttribute('aria-disabled');
        }
        elements.stereo.left.input.disabled = false;
        elements.stereo.right.input.disabled = false;
        elements.stereo.left.clear.disabled = false;
        elements.stereo.right.clear.disabled = false;
        elements.stereo.baseline.disabled = false;
        setStatus('stereo', 'Load the stereo pair and select the same feature in both images.', 'info');
    }

    function unlockMeasurementStep() {
        if (elements.measure.card) {
            elements.measure.card.classList.remove('locked');
            elements.measure.card.removeAttribute('aria-disabled');
        }
        drawMeasurementCanvas();
        setStatus('measure', 'Click two points (A then B) to capture a measurement.', 'info');
    }

    function renderMeasurements() {
        const list = elements.measure.list;
        if (!list) return;
        list.innerHTML = '';
        if (!state.measurement.segments.length) {
            const placeholder = document.createElement('li');
            placeholder.className = 'placeholder';
            placeholder.textContent = 'No measurements yet.';
            list.appendChild(placeholder);
            return;
        }
        state.measurement.segments.forEach((segment, index) => {
            const li = document.createElement('li');
            li.innerHTML = `Measurement #${index + 1}: ${numberFormat.format(segment.realSize)} cm <span class="meta">${numberFormat.format(segment.pixelDistance)} px</span>`;
            list.appendChild(li);
        });
        elements.measure.resetBtn.disabled = false;
    }

    function updateCalibrationButtonState() {
        const hasImage = Boolean(state.reference.image);
        const hasPoints = typeof state.reference.pixelWidth === 'number';
        const realWidth = parseFloat(elements.ref.realWidth.value);
        const distance = parseFloat(elements.ref.distance.value);
        const ready = hasImage && hasPoints && realWidth > 0 && distance > 0;
        elements.ref.calcBtn.disabled = !ready;
        return ready;
    }

    function updateStereoButtonState() {
        const ready = Boolean(state.focalLength && state.stereo.left.point && state.stereo.right.point && state.stereo.disparity && state.stereo.disparity > 0);
        const baselineVal = parseFloat(elements.stereo.baseline.value);
        const baselineReady = ready && baselineVal > 0;
        state.stereo.baseline = baselineVal > 0 ? baselineVal : null;
        elements.stereo.calcBtn.disabled = !baselineReady;
        updateSummary();
        return baselineReady;
    }

    function handleReferenceCanvasClick(event) {
        if (!state.reference.image || !state.reference.meta) return;
        if (state.reference.points.length === 2) return;

        const rect = elements.ref.canvas.getBoundingClientRect();
        const scaleX = state.reference.meta.naturalWidth / rect.width;
        const scaleY = state.reference.meta.naturalHeight / rect.height;
        const displayScaleX = state.reference.meta.displayWidth / rect.width;
        const displayScaleY = state.reference.meta.displayHeight / rect.height;
        const clickPoint = {
            canvas: {
                x: (event.clientX - rect.left) * displayScaleX,
                y: (event.clientY - rect.top) * displayScaleY
            },
            image: {
                x: (event.clientX - rect.left) * scaleX,
                y: (event.clientY - rect.top) * scaleY
            }
        };

        state.reference.points.push(clickPoint);
        if (state.reference.points.length === 2) {
            const [a, b] = state.reference.points;
            state.reference.pixelWidth = Math.hypot(b.image.x - a.image.x, b.image.y - a.image.y);
            elements.ref.reset.disabled = false;
            setStatus('ref', 'Enter the real width & distance, then store the focal length.', 'success');
        } else {
            setStatus('ref', 'Point A recorded. Click the opposite edge to complete the measurement.', 'info');
        }

        drawReferenceCanvas();
        updateReferencePixelLabel();
        updateCalibrationButtonState();
    }

    function handleStereoCanvasClick(side, event) {
        const bucket = state.stereo[side];
        const el = elements.stereo[side];
        if (!bucket.image || !bucket.meta) return;

        const rect = el.canvas.getBoundingClientRect();
        const scaleX = bucket.meta.naturalWidth / rect.width;
        const scaleY = bucket.meta.naturalHeight / rect.height;
        const displayScaleX = bucket.meta.displayWidth / rect.width;
        const displayScaleY = bucket.meta.displayHeight / rect.height;

        bucket.point = {
            canvas: {
                x: (event.clientX - rect.left) * displayScaleX,
                y: (event.clientY - rect.top) * displayScaleY
            },
            image: {
                x: (event.clientX - rect.left) * scaleX,
                y: (event.clientY - rect.top) * scaleY
            }
        };

        drawStereoCanvas(side);
        updateStereoMetrics();

        if (state.stereo.left.point && state.stereo.right.point) {
            state.stereo.disparity = Math.abs(state.stereo.left.point.image.x - state.stereo.right.point.image.x);
            if (state.stereo.disparity === 0) {
                setStatus('stereo', 'Disparity is zero. Choose distinct x positions in the left/right frames.', 'error');
            } else {
                setStatus('stereo', 'Baseline + disparity ready. Enter the baseline and calculate depth.', 'success');
            }
        }

        updateStereoMetrics();
        updateStereoButtonState();
    }

    function handleMeasurementCanvasClick(event) {
        if (!state.stereo.depth || !state.stereo.left.image || !state.stereo.left.meta || state.measurement.isProcessing) return;
        const rect = elements.measure.canvas.getBoundingClientRect();
        const meta = state.stereo.left.meta;
        const scaleX = meta.naturalWidth / rect.width;
        const scaleY = meta.naturalHeight / rect.height;
        const displayScaleX = meta.displayWidth / rect.width;
        const displayScaleY = meta.displayHeight / rect.height;

        const point = {
            canvas: {
                x: (event.clientX - rect.left) * displayScaleX,
                y: (event.clientY - rect.top) * displayScaleY
            },
            image: {
                x: (event.clientX - rect.left) * scaleX,
                y: (event.clientY - rect.top) * scaleY
            }
        };

        state.measurement.pendingPoints.push(point);
        if (state.measurement.pendingPoints.length > 2) {
            state.measurement.pendingPoints.shift();
        }

        drawMeasurementCanvas();

        if (state.measurement.pendingPoints.length === 2) {
            submitMeasurement();
        } else {
            setStatus('measure', 'Point A logged. Click point B to complete the measurement.', 'info');
        }
    }

    async function submitMeasurement() {
        if (state.measurement.pendingPoints.length !== 2 || state.measurement.isProcessing) return;
        if (!state.focalLength || !state.stereo.depth) {
            state.measurement.pendingPoints = [];
            drawMeasurementCanvas();
            setStatus('measure', 'Complete Steps 1 and 2 before measuring segments.', 'error');
            return;
        }
        const [a, b] = state.measurement.pendingPoints;
        const pixelDistance = Math.hypot(b.image.x - a.image.x, b.image.y - a.image.y);
        if (pixelDistance <= 0) {
            setStatus('measure', 'Select two different points to measure.', 'error');
            return;
        }

        state.measurement.isProcessing = true;
        setStatus('measure', 'Calculating measurement…', 'info');
        try {
            const response = await fetch('/api/a7/part1/measure', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    pixelDistance,
                    focalLength: state.focalLength,
                    depth: state.stereo.depth
                })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Measurement failed.');

            state.measurement.segments.push({
                points: [a.canvas, b.canvas],
                realSize: data.realSizeCm,
                pixelDistance
            });
            state.measurement.pendingPoints = [];
            renderMeasurements();
            drawMeasurementCanvas();
            setStatus('measure', 'Measurement stored. Click again to measure another segment.', 'success');
        } catch (err) {
            setStatus('measure', err.message || 'Unable to calculate measurement.', 'error');
        } finally {
            state.measurement.isProcessing = false;
        }
    }

    function loadReferenceImage(img) {
        const size = calculateDisplaySize(img);
        elements.ref.canvas.width = size.displayWidth;
        elements.ref.canvas.height = size.displayHeight;
        elements.ref.ctx.clearRect(0, 0, size.displayWidth, size.displayHeight);
        elements.ref.ctx.drawImage(img, 0, 0, size.displayWidth, size.displayHeight);
        state.reference.image = img;
        state.reference.meta = {
            displayWidth: size.displayWidth,
            displayHeight: size.displayHeight,
            naturalWidth: img.width,
            naturalHeight: img.height
        };
        resetReferencePoints();
        elements.ref.clear.disabled = false;
        if (elements.ref.wrapper) elements.ref.wrapper.dataset.state = 'ready';
        if (elements.ref.empty) elements.ref.empty.textContent = 'Click the LEFT and RIGHT edges of the known object.';
        setStatus('ref', 'Image loaded. Select the two calibration points.', 'info');
        updateCalibrationButtonState();
    }

    function loadStereoImage(side, img) {
        const target = elements.stereo[side];
        const size = calculateDisplaySize(img);
        target.canvas.width = size.displayWidth;
        target.canvas.height = size.displayHeight;
        target.ctx.clearRect(0, 0, size.displayWidth, size.displayHeight);
        target.ctx.drawImage(img, 0, 0, size.displayWidth, size.displayHeight);
        state.stereo[side].image = img;
        state.stereo[side].meta = {
            displayWidth: size.displayWidth,
            displayHeight: size.displayHeight,
            naturalWidth: img.width,
            naturalHeight: img.height
        };
        state.stereo[side].point = null;
        target.label.textContent = '--';
        if (target.wrapper) target.wrapper.dataset.state = 'ready';
        if (target.empty) {
            target.empty.textContent = side === 'left' ? 'Click a distinct feature (Point L).' : 'Click the matching feature inside the RIGHT image.';
        }
        drawStereoCanvas(side);
        updateStereoMetrics();
        updateStereoButtonState();
        if (side === 'left' && state.stereo.depth) {
            resetMeasurements();
        }
    }

    function loadImageFromFile(kind, file) {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                if (kind === 'reference') loadReferenceImage(img);
                else loadStereoImage(kind, img);
            };
            img.onerror = () => setStatus(kind === 'reference' ? 'ref' : 'stereo', 'Unable to load that file.', 'error');
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }

    function loadImageFromDataUrl(kind, dataUrl) {
        if (!dataUrl) return;
        const img = new Image();
        img.onload = () => {
            if (kind === 'reference') loadReferenceImage(img);
            else loadStereoImage(kind, img);
        };
        img.onerror = () => setStatus(kind === 'reference' ? 'ref' : 'stereo', 'Failed to load the sample image.', 'error');
        img.src = dataUrl;
    }

    function addAutofillHighlight(input, value) {
        if (!input || value === undefined || value === null) return;
        input.value = Number.isFinite(value) ? value : value;
        input.classList.add('autofilled', 'autofilled-pulse');
        const removeClass = () => {
            input.classList.remove('autofilled');
            input.classList.remove('autofilled-pulse');
            input.removeEventListener('input', removeClass);
        };
        input.addEventListener('input', removeClass);
        setTimeout(() => input.classList.remove('autofilled-pulse'), 4000);
    }

    async function handleSampleLoad() {
        if (!elements.exampleBtn) return;
        elements.exampleBtn.disabled = true;
        setStatus('ref', 'Loading example dataset…', 'info');
        try {
            const response = await fetch('/api/a7/part1/sample');
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to load example dataset.');

            if (data.reference?.image) {
                loadImageFromDataUrl('reference', data.reference.image);
            }
            if (data.left?.image) {
                loadImageFromDataUrl('left', data.left.image);
            }
            if (data.right?.image) {
                loadImageFromDataUrl('right', data.right.image);
            }

            const measurements = data.measurements || {};
            if (measurements.referenceWidthCm) {
                addAutofillHighlight(elements.ref.realWidth, measurements.referenceWidthCm);
            }
            if (measurements.referenceDistanceCm) {
                addAutofillHighlight(elements.ref.distance, measurements.referenceDistanceCm);
            }
            if (measurements.baselineCm) {
                addAutofillHighlight(elements.stereo.baseline, measurements.baselineCm);
                const wasDisabled = elements.stereo.baseline.disabled;
                if (wasDisabled) elements.stereo.baseline.disabled = false;
                elements.stereo.baseline.dispatchEvent(new Event('input', { bubbles: true }));
                if (wasDisabled) elements.stereo.baseline.disabled = true;
            }

            setStatus('ref', 'Example assets ready. Mark the calibration points, then enter the provided cm values.', 'success');
        } catch (err) {
            setStatus('ref', err.message || 'Unable to load example assets.', 'error');
        } finally {
            elements.exampleBtn.disabled = false;
        }
    }

    async function submitFocalLength() {
        if (!updateCalibrationButtonState()) return;
        elements.ref.calcBtn.disabled = true;
        setStatus('ref', 'Calculating focal length…', 'info');
        const payload = {
            pixelWidth: state.reference.pixelWidth,
            realWidth: parseFloat(elements.ref.realWidth.value),
            distance: parseFloat(elements.ref.distance.value)
        };
        try {
            const response = await fetch('/api/a7/part1/focal-length', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Unable to calculate focal length.');
            state.focalLength = data.focalLength;
            state.reference.realWidth = data.refRealWidth;
            state.reference.distance = data.refDistance;
            setStatus('ref', `Focal length stored: ${numberFormat.format(state.focalLength)} px`, 'success');
            unlockStereoStep();
            updateSummary();
        } catch (err) {
            setStatus('ref', err.message || 'Unable to calculate focal length.', 'error');
        } finally {
            updateCalibrationButtonState();
        }
    }

    async function submitDepth() {
        if (!updateStereoButtonState()) return;
        elements.stereo.calcBtn.disabled = true;
        setStatus('stereo', 'Calculating Z distance…', 'info');
        const payload = {
            focalLength: state.focalLength,
            baseline: parseFloat(elements.stereo.baseline.value),
            disparity: state.stereo.disparity
        };
        try {
            const response = await fetch('/api/a7/part1/depth', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Depth calculation failed.');
            state.stereo.depth = data.depthCm;
            state.stereo.baseline = data.baselineCm;
            setStatus('stereo', `Depth recovered: ${numberFormat.format(state.stereo.depth)} cm`, 'success');
            updateSummary();
            state.measurement.pendingPoints = [];
            state.measurement.segments = [];
            renderMeasurements();
            unlockMeasurementStep();
        } catch (err) {
            setStatus('stereo', err.message || 'Unable to compute depth.', 'error');
        } finally {
            elements.stereo.calcBtn.disabled = !updateStereoButtonState();
            drawMeasurementCanvas();
        }
    }

    function resetMeasurements() {
        state.measurement.pendingPoints = [];
        state.measurement.segments = [];
        state.measurement.isProcessing = false;
        elements.measure.resetBtn.disabled = true;
        renderMeasurements();
        drawMeasurementCanvas();
        setStatus('measure', 'Measurements cleared. Click in the image to capture new ones.', 'info');
    }

    function resetModule() {
        resetReferenceImage();
        resetStereoWorkflow();
        lockMeasurementStep();
        state.focalLength = null;
        [elements.ref.realWidth, elements.ref.distance, elements.stereo.baseline].forEach((input) => {
            if (!input) return;
            input.classList.remove('autofilled');
            input.classList.remove('autofilled-pulse');
        });
        updateCalibrationButtonState();
        updateSummary();
        setStatus('ref', 'Upload the reference image to begin calibration.', 'info');
    }

    elements.ref.input.addEventListener('change', (event) => {
        const file = event.target.files?.[0];
        loadImageFromFile('reference', file);
    });
    elements.ref.clear.addEventListener('click', resetModule);
    elements.ref.reset.addEventListener('click', resetReferencePoints);
    elements.ref.realWidth.addEventListener('input', updateCalibrationButtonState);
    elements.ref.distance.addEventListener('input', updateCalibrationButtonState);
    elements.ref.calcBtn.addEventListener('click', submitFocalLength);
    elements.ref.canvas.addEventListener('click', handleReferenceCanvasClick);

    elements.stereo.left.input.addEventListener('change', (event) => {
        const file = event.target.files?.[0];
        loadImageFromFile('left', file);
    });
    elements.stereo.right.input.addEventListener('change', (event) => {
        const file = event.target.files?.[0];
        loadImageFromFile('right', file);
    });
    function handleStereoClear(side) {
        resetStereoSide(side);
        state.stereo.disparity = null;
        state.stereo.depth = null;
        state.stereo.baseline = null;
        elements.stereo.baseline.value = '';
        updateStereoMetrics();
        updateStereoButtonState();
        lockMeasurementStep();
        drawMeasurementCanvas();
        setStatus('stereo', 'Stereo inputs reset. Reload the pair and reselect the matching points.', 'info');
    }

    elements.stereo.left.clear.addEventListener('click', () => handleStereoClear('left'));
    elements.stereo.right.clear.addEventListener('click', () => handleStereoClear('right'));
    elements.stereo.left.canvas.addEventListener('click', (event) => handleStereoCanvasClick('left', event));
    elements.stereo.right.canvas.addEventListener('click', (event) => handleStereoCanvasClick('right', event));
    elements.stereo.baseline.addEventListener('input', () => {
        const value = parseFloat(elements.stereo.baseline.value);
        state.stereo.baseline = Number.isFinite(value) && value > 0 ? value : null;
        updateSummary();
        updateStereoButtonState();
    });
    elements.stereo.calcBtn.addEventListener('click', submitDepth);

    elements.measure.canvas.addEventListener('click', handleMeasurementCanvasClick);
    elements.measure.resetBtn.addEventListener('click', resetMeasurements);

    if (elements.exampleBtn) {
        elements.exampleBtn.addEventListener('click', handleSampleLoad);
    }

    resetModule();
    window.__module7Part1Reset = resetModule;
    registerGlobalResetHandler(async () => {
        try {
            resetModule();
        } catch (err) {
            console.warn('Module 7 Part 1 global reset failed', err);
        }
    });
}

function initModule7Flow() {
    const form = document.getElementById('module7-part3-form');
    if (!form) return;

    const fileInput = document.getElementById('module7-video-input');
    const sampleCheckbox = document.getElementById('module7-use-sample');
    const runBtn = document.getElementById('module7-run-btn');
    const resetBtn = document.getElementById('module7-reset-btn');
    const statusLine = document.getElementById('module7-part3-status');
    const originalVideo = document.getElementById('module7-original-video');
    const annotatedVideo = document.getElementById('module7-annotated-video');
    const originalPlaceholder = document.getElementById('module7-original-placeholder');
    const annotatedPlaceholder = document.getElementById('module7-annotated-placeholder');
    const downloadArea = document.getElementById('module7-download-area');
    const downloadLink = document.getElementById('module7-annotated-download');
    const csvDownload = document.getElementById('module7-csv-download');
    const csvWrapper = document.getElementById('module7-csv-wrapper');
    const csvHead = document.getElementById('module7-csv-head');
    const csvBody = document.getElementById('module7-csv-body');
    const csvPlaceholder = document.getElementById('module7-csv-placeholder');
    const framesMetric = document.getElementById('module7-metric-frames');
    const durationMetric = document.getElementById('module7-metric-duration');
    const fpsMetric = document.getElementById('module7-metric-fps');
    const sizeMetric = document.getElementById('module7-metric-size');
    const csvMetric = document.getElementById('module7-metric-csv');

    if (!fileInput || !runBtn || !resetBtn || !statusLine || !originalVideo || !annotatedVideo) {
        return;
    }

    const SAMPLE_FILENAME = 'karate.mp4';
    const state = {
        sampleToken: 0,
        localPreviewUrl: null,
        hasResults: false
    };

    const hasSelection = () => Boolean((fileInput.files && fileInput.files.length) || (sampleCheckbox && sampleCheckbox.checked));

    const setStatus = (message, variant = 'info') => {
        statusLine.textContent = message;
        statusLine.dataset.variant = variant;
    };

    const revokeLocalPreview = () => {
        if (!state.localPreviewUrl) return;
        try { URL.revokeObjectURL(state.localPreviewUrl); } catch (err) { /* ignore */ }
        state.localPreviewUrl = null;
    };

    const clearVideoElement = (videoEl, placeholder) => {
        if (!videoEl) return;
        try { videoEl.pause(); } catch (err) { /* ignore */ }
        Array.from(videoEl.querySelectorAll('source')).forEach((src) => src.remove());
        try { videoEl.removeAttribute('src'); } catch (err) { /* ignore */ }
        videoEl.hidden = true;
        if (placeholder) placeholder.hidden = false;
        const wrapper = videoEl.closest('.result-image');
        if (wrapper) wrapper.dataset.empty = 'true';
        try { videoEl.load(); } catch (err) { /* ignore */ }
    };

    const showVideo = (videoEl, placeholder, source) => {
        if (!videoEl) return;
        Array.from(videoEl.querySelectorAll('source')).forEach((src) => src.remove());
        try { videoEl.removeAttribute('src'); } catch (err) { /* ignore */ }
        if (!source) {
            clearVideoElement(videoEl, placeholder);
            return;
        }
        const sourceEl = document.createElement('source');
        sourceEl.src = source;
        videoEl.appendChild(sourceEl);
        videoEl.hidden = false;
        if (placeholder) placeholder.hidden = true;
        const wrapper = videoEl.closest('.result-image');
        if (wrapper) wrapper.dataset.empty = 'false';
        try { videoEl.load(); } catch (err) { /* ignore */ }
    };

    const resetMetrics = () => {
        if (framesMetric) framesMetric.textContent = '--';
        if (durationMetric) durationMetric.textContent = '--';
        if (fpsMetric) fpsMetric.textContent = '--';
        if (sizeMetric) sizeMetric.textContent = '-- × --';
        if (csvMetric) csvMetric.textContent = '--';
    };

    const formatNumber = (value, digits = 2) => {
        if (value === null || value === undefined) return null;
        const num = Number(value);
        if (!Number.isFinite(num)) return null;
        return num.toFixed(digits);
    };

    const updateMetrics = (summary = {}) => {
        if (framesMetric) framesMetric.textContent = summary.frameCount ?? '--';
        if (durationMetric) {
            const formatted = formatNumber(summary.durationSeconds, 2);
            durationMetric.textContent = formatted ?? '--';
        }
        if (fpsMetric) {
            const formatted = formatNumber(summary.fps, 2);
            fpsMetric.textContent = formatted ?? '--';
        }
        if (sizeMetric) {
            if (summary.frameWidth && summary.frameHeight) {
                sizeMetric.textContent = `${summary.frameWidth} × ${summary.frameHeight}`;
            } else {
                sizeMetric.textContent = '-- × --';
            }
        }
        if (csvMetric) {
            const csvCount = summary.csvRowCount ?? summary.recordCount;
            csvMetric.textContent = csvCount ?? '--';
        }
    };

    const resetCsv = () => {
        if (csvHead) csvHead.innerHTML = '';
        if (csvBody) csvBody.innerHTML = '';
        if (csvWrapper) csvWrapper.dataset.empty = 'true';
        if (csvPlaceholder) csvPlaceholder.hidden = false;
        if (csvDownload) {
            csvDownload.hidden = true;
            csvDownload.href = '#';
            csvDownload.removeAttribute('download');
        }
    };

    const renderCsv = (preview) => {
        if (!csvHead || !csvBody || !csvWrapper) return;
        csvHead.innerHTML = '';
        csvBody.innerHTML = '';
        if (!preview || !Array.isArray(preview.columns) || !preview.columns.length || !Array.isArray(preview.rows) || !preview.rows.length) {
            csvWrapper.dataset.empty = 'true';
            if (csvPlaceholder) csvPlaceholder.hidden = false;
            return;
        }

        csvWrapper.dataset.empty = 'false';
        if (csvPlaceholder) csvPlaceholder.hidden = true;

        const headRow = document.createElement('tr');
        preview.columns.forEach((col) => {
            const th = document.createElement('th');
            th.textContent = col;
            headRow.appendChild(th);
        });
        csvHead.appendChild(headRow);

        preview.rows.forEach((row) => {
            const tr = document.createElement('tr');
            preview.columns.forEach((col) => {
                const td = document.createElement('td');
                td.textContent = row && row[col] != null ? row[col] : '';
                tr.appendChild(td);
            });
            csvBody.appendChild(tr);
        });
    };

    resetMetrics();
    resetCsv();
    clearVideoElement(originalVideo, originalPlaceholder);
    clearVideoElement(annotatedVideo, annotatedPlaceholder);
    setStatus('Choose a video or enable the sample clip to get started.', 'info');
    runBtn.disabled = true;
    resetBtn.disabled = true;

    fileInput.addEventListener('change', () => {
        revokeLocalPreview();
        if (fileInput.files && fileInput.files.length) {
            const file = fileInput.files[0];
            const url = URL.createObjectURL(file);
            state.localPreviewUrl = url;
            showVideo(originalVideo, originalPlaceholder, url);
            runBtn.disabled = false;
            resetBtn.disabled = false;
            setStatus('Ready to run pose tracking.', 'info');
        } else {
            clearVideoElement(originalVideo, originalPlaceholder);
            if (!state.hasResults) resetBtn.disabled = true;
            runBtn.disabled = !hasSelection();
            setStatus('Choose a video or enable the sample clip to get started.', 'info');
        }
    });

    if (sampleCheckbox) {
        sampleCheckbox.addEventListener('change', async () => {
            const token = ++state.sampleToken;
            if (sampleCheckbox.checked) {
                fileInput.value = '';
                fileInput.disabled = true;
                revokeLocalPreview();
                clearVideoElement(originalVideo, originalPlaceholder);
                runBtn.disabled = false;
                resetBtn.disabled = false;
                setStatus('Loading sample video preview…', 'info');
                try {
                    const resp = await fetch('/api/a7/part2/sample');
                    const data = await resp.json();
                    if (token !== state.sampleToken) return;
                    if (!resp.ok) throw new Error(data.error || 'Failed to load sample');
                    showVideo(originalVideo, originalPlaceholder, data.video);
                    setStatus(`Using sample ${data.filename}. Ready to run.`, 'info');
                } catch (err) {
                    if (token !== state.sampleToken) return;
                    setStatus(err.message || 'Failed to load sample.', 'error');
                    sampleCheckbox.checked = false;
                    fileInput.disabled = false;
                    runBtn.disabled = !hasSelection();
                    if (!state.hasResults) resetBtn.disabled = !hasSelection();
                }
            } else {
                if (token !== state.sampleToken) return;
                fileInput.disabled = false;
                if (!hasSelection()) {
                    clearVideoElement(originalVideo, originalPlaceholder);
                    runBtn.disabled = true;
                    if (!state.hasResults) resetBtn.disabled = true;
                    setStatus('Choose a video or enable the sample clip to get started.', 'info');
                }
            }
        });
    }

    const performFullReset = () => {
        form.reset();
        revokeLocalPreview();
        state.sampleToken += 1;
        state.hasResults = false;
        if (sampleCheckbox) {
            sampleCheckbox.checked = false;
            sampleCheckbox.disabled = false;
        }
        fileInput.disabled = false;
        fileInput.value = '';
        clearVideoElement(originalVideo, originalPlaceholder);
        clearVideoElement(annotatedVideo, annotatedPlaceholder);
        resetCsv();
        resetMetrics();
        if (downloadArea && downloadLink) {
            downloadArea.hidden = true;
            downloadLink.href = '#';
            downloadLink.removeAttribute('download');
        }
        setStatus('Choose a video or enable the sample clip to get started.', 'info');
        runBtn.disabled = true;
        resetBtn.disabled = true;
    };

    resetBtn.addEventListener('click', performFullReset);

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!hasSelection()) {
            setStatus('Please upload a video or enable the sample clip before running.', 'error');
            return;
        }

        const formData = new FormData();
        if (sampleCheckbox && sampleCheckbox.checked) {
            formData.append('sample', SAMPLE_FILENAME);
        } else if (fileInput.files && fileInput.files[0]) {
            formData.append('video', fileInput.files[0]);
        } else {
            setStatus('Please choose a video to continue.', 'error');
            return;
        }

        setStatus('Uploading video and running pose tracking…', 'info');
        runBtn.disabled = true;
        resetBtn.disabled = true;
        fileInput.disabled = true;
        if (sampleCheckbox) sampleCheckbox.disabled = true;

        try {
            const response = await fetch('/api/a7/part2', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Process failed');

            revokeLocalPreview();
            showVideo(originalVideo, originalPlaceholder, data?.original?.video || null);
            showVideo(annotatedVideo, annotatedPlaceholder, data?.annotated?.video || null);

            if (downloadArea && downloadLink) {
                if (data?.annotated?.video) {
                    downloadLink.href = data.annotated.video;
                    downloadLink.download = data.annotated.filename || 'annotated.mp4';
                    downloadArea.hidden = false;
                } else {
                    downloadLink.href = '#';
                    downloadLink.removeAttribute('download');
                    downloadArea.hidden = true;
                }
            }

            if (csvDownload && data?.csv?.dataUrl) {
                csvDownload.href = data.csv.dataUrl;
                csvDownload.download = data.csv.filename || 'pose-landmarks.csv';
                csvDownload.hidden = false;
            } else if (csvDownload) {
                csvDownload.hidden = true;
                csvDownload.href = '#';
                csvDownload.removeAttribute('download');
            }

            renderCsv(data?.csv?.preview || null);
            updateMetrics(data?.summary || {});
            if (csvMetric && data?.csv?.rowCount != null) {
                csvMetric.textContent = data.csv.rowCount;
            }

            state.hasResults = true;
            resetBtn.disabled = false;
            setStatus(data.message || 'Pose estimation complete.', 'success');
        } catch (err) {
            setStatus(err.message || 'Unexpected error occurred.', 'error');
        } finally {
            runBtn.disabled = !hasSelection();
            if (sampleCheckbox) {
                sampleCheckbox.disabled = false;
                fileInput.disabled = sampleCheckbox.checked;
            } else {
                fileInput.disabled = false;
            }
            if (!state.hasResults && !hasSelection()) {
                resetBtn.disabled = true;
            }
        }
    });

    form.__module7Reset = performFullReset;
    registerGlobalResetHandler(async () => {
        try {
            performFullReset();
        } catch (err) {
            console.warn('Module 7 Part 2 global reset failed', err);
        }
    });
}

/*
 * Global reset button - clears state across modules and UI.
 * Click the existing reset/clear buttons (if available) so module-specific
 * handlers run, then fall back to explicit form resets and cleanup.
 */
function initGlobalReset() {
    const globalResetBtn = document.getElementById('global-reset-btn');
    if (!globalResetBtn) return;
    const defaultLabel = globalResetBtn.textContent.trim() || 'Reset All';

    const setBusyState = (isBusy) => {
        if (isBusy) {
            globalResetBtn.disabled = true;
            globalResetBtn.dataset.state = 'resetting';
            globalResetBtn.textContent = 'Resetting…';
        } else {
            globalResetBtn.disabled = false;
            globalResetBtn.textContent = defaultLabel;
            delete globalResetBtn.dataset.state;
        }
    };

    const triggerButton = (id) => {
        if (!id) return false;
        const btn = document.getElementById(id);
        if (!btn) return false;
        const wasDisabled = btn.disabled;
        if (wasDisabled) btn.disabled = false;
        try { btn.dispatchEvent(new MouseEvent('click', { bubbles: true })); } catch (err) { console.warn(`Global reset could not trigger ${id}`, err); }
        if (wasDisabled) btn.disabled = true;
        return true;
    };

    const clearModule56Media = () => {
        const m56File = document.getElementById('module5-6-video-input');
        const m56InVideo = document.getElementById('module5-6-input-video');
        const m56OutVideo = document.getElementById('module5-6-output-video');
        const m56InPlaceholder = document.getElementById('module5-6-input-placeholder');
        const m56OutPlaceholder = document.getElementById('module5-6-output-placeholder');
        if (m56File) m56File.value = '';
        if (m56InVideo) {
            try { m56InVideo.pause(); m56InVideo.removeAttribute('src'); m56InVideo.load(); } catch (err) { /* ignore */ }
            m56InVideo.hidden = true;
            if (m56InPlaceholder) m56InPlaceholder.hidden = false;
        }
        if (m56OutVideo) {
            try { m56OutVideo.pause(); m56OutVideo.removeAttribute('src'); m56OutVideo.load(); } catch (err) { /* ignore */ }
            m56OutVideo.hidden = true;
            if (m56OutPlaceholder) m56OutPlaceholder.hidden = false;
        }
        const dlArea = document.getElementById('module5-6-download-area');
        if (dlArea) dlArea.style.display = 'none';
        const dlLink = document.getElementById('module5-6-download-link');
        if (dlLink) dlLink.removeAttribute('href');
        const m56UseSample = document.getElementById('module5-6-use-sample');
        if (m56UseSample) {
            m56UseSample.checked = false;
            m56UseSample.disabled = false;
        }
    };

    const collapsePanels = () => {
        const toggleSelectors = ['.instructions-toggle', '.video-toggle', '.source-toggle'];
        document.querySelectorAll(toggleSelectors.join(',')).forEach((btn) => {
            if (btn.getAttribute('aria-expanded') !== 'true') return;
            const panelId = btn.getAttribute('aria-controls');
            const panel = panelId ? document.getElementById(panelId) : null;
            if (panel) {
                panel.classList.remove('open');
                panel.setAttribute('aria-hidden', 'true');
            }
            btn.setAttribute('aria-expanded', 'false');
            if (btn.classList.contains('video-toggle')) {
                btn.textContent = 'Watch the video ▾';
            } else if (btn.classList.contains('instructions-toggle')) {
                btn.textContent = 'View module instructions ▾';
            } else {
                btn.textContent = 'View CLI source code ▾';
            }
        });
    };

    const resetSimpleIoBlocks = () => {
        ['a4', 'a56', 'a7'].forEach((id) => {
            const input = document.getElementById(`input-${id}`);
            const output = document.getElementById(`output-${id}`);
            if (input && 'value' in input) input.value = '';
            if (output && 'textContent' in output) output.textContent = '// Output will appear here...';
        });
    };

    globalResetBtn.addEventListener('click', async () => {
        const confirmed = confirm('Reset all modules and clear results? This will clear images, points, and outputs across every module.');
        if (!confirmed) return;

        setBusyState(true);
        try {
            triggerButton('module1-reset');

            await runGlobalResetHandlers();

            triggerButton('module5-6-reset-btn');
            clearModule56Media();

            resetSimpleIoBlocks();
            collapsePanels();

            // Ensure Module 1 file inputs are cleared regardless of UI state
            const refFile = document.getElementById('ref-image-input');
            const testFile = document.getElementById('test-image-input');
            if (refFile) refFile.value = '';
            if (testFile) testFile.value = '';

            document.querySelectorAll('.autofilled').forEach((el) => el.classList.remove('autofilled'));
            document.querySelectorAll('.autofilled-pulse').forEach((el) => el.classList.remove('autofilled-pulse'));
            const expectedEl = document.getElementById('test-expected-value');
            if (expectedEl) expectedEl.textContent = '--';

            document.dispatchEvent(new CustomEvent('globalreset', { detail: { triggeredAt: Date.now() } }));

            try {
                document.getElementById('ref-status').textContent = 'All modules reset.';
                document.getElementById('test-status').textContent = 'All modules reset.';
            } catch (err) { /* ignore */ }
        } finally {
            setBusyState(false);
        }
    });
}

function initModule2Flow() {
    const form = document.getElementById('module2-part2-form');
    const fileInput = document.getElementById('module2-image-input');
    const useSampleCheckbox = document.getElementById('module2-use-sample');
    const runBtn = document.getElementById('module2-run-btn');
    const resetBtn = document.getElementById('module2-reset-btn');
    const statusLine = document.getElementById('module2-part2-status');
    const originalImg = document.getElementById('module2-original-preview');
    const originalPlaceholder = document.getElementById('module2-original-placeholder');
    const blurredImg = document.getElementById('module2-blurred-preview');
    const restoredImg = document.getElementById('module2-restored-preview');
    const blurredPlaceholder = document.getElementById('module2-blurred-placeholder');
    const restoredPlaceholder = document.getElementById('module2-restored-placeholder');

    if (!form || !fileInput || !runBtn || !resetBtn || !statusLine || !blurredImg || !restoredImg || !originalImg) {
        return;
    }

    const resultFrames = [
        { img: originalImg, placeholder: originalPlaceholder, isOriginal: true },
        { img: blurredImg, placeholder: blurredPlaceholder, isOriginal: false },
        { img: restoredImg, placeholder: restoredPlaceholder, isOriginal: false }
    ];
    const defaultStatus = (statusLine.textContent || 'Choose an image to get started.').trim();

    const hasFileSelected = () => Boolean((fileInput.files && fileInput.files.length) || (useSampleCheckbox && useSampleCheckbox.checked));
    const hasResults = () => resultFrames.some(({ img, isOriginal }) => !isOriginal && img.dataset.loaded === 'true');

    const setStatus = (message, variant = 'info') => {
        statusLine.textContent = message;
        statusLine.dataset.variant = variant;
    };

    const showImage = (imgEl, placeholderEl, dataUrl) => {
        if (!imgEl) return;
        imgEl.src = dataUrl;
        imgEl.hidden = false;
        imgEl.dataset.loaded = 'true';
        const wrapper = imgEl.closest('.result-image');
        if (wrapper) wrapper.dataset.empty = 'false';
        if (placeholderEl) placeholderEl.hidden = true;
        markModule2ImageInteractive(imgEl);
    };

    const clearResults = (opts = { keepOriginal: false }) => {
        resultFrames.forEach(({ img, placeholder, isOriginal }) => {
            if (isOriginal && opts.keepOriginal) {
                return;
            }
            if (!img) return;
            img.removeAttribute('src');
            img.hidden = true;
            img.dataset.loaded = 'false';
            const wrapper = img.closest('.result-image');
            if (wrapper) wrapper.dataset.empty = 'true';
            if (placeholder) {
                placeholder.hidden = false;
                placeholder.textContent = isOriginal ? 'No image selected.' : 'No output yet.';
            }
        });
    };

    const showOriginal = (file) => {
        if (!file) {
            clearResults({ keepOriginal: false });
            return;
        }
        const reader = new FileReader();
        reader.onload = (evt) => {
            showImage(originalImg, originalPlaceholder, evt.target?.result || '');
        };
        reader.onerror = () => {
            clearResults({ keepOriginal: false });
            setStatus('Could not preview that image locally.', 'error');
        };
        reader.readAsDataURL(file);
    };

    const showOriginalFromUrl = (dataUrl) => {
        if (!dataUrl) {
            clearResults();
            return;
        }
        showImage(originalImg, originalPlaceholder, dataUrl);
    };

    clearResults();
    setStatus(defaultStatus, 'info');

    const resetModule = () => {
        form.reset();
        clearResults();
        runBtn.disabled = true;
        resetBtn.disabled = true;
        if (useSampleCheckbox) {
            useSampleCheckbox.checked = false;
            useSampleCheckbox.disabled = false;
        }
        try { fileInput.value = ''; } catch (err) { /* ignore */ }
        fileInput.disabled = false;
        setStatus(defaultStatus, 'info');
    };

    fileInput.addEventListener('change', () => {
        const hasFile = hasFileSelected();
        runBtn.disabled = !hasFile;
        resetBtn.disabled = !hasFile && !hasResults();
        if (hasFile) {
            // If the source is a file, prefer that preview
            if (fileInput.files && fileInput.files.length) {
                showOriginal(fileInput.files[0]);
            }
            setStatus('Ready to run the process.', 'info');
        } else if (!hasResults()) {
            clearResults();
            setStatus('Choose an image to get started.', 'info');
        }
    });

    if (useSampleCheckbox) {
        useSampleCheckbox.addEventListener('change', async () => {
            const checked = useSampleCheckbox.checked;
            if (checked) {
                // disable file input
                fileInput.value = '';
                fileInput.disabled = true;
                resetBtn.disabled = false;
                // fetch the sample preview
                setStatus('Loading example image preview…', 'info');
                try {
                    const resp = await fetch('/api/a2/part2/sample');
                    const data = await resp.json();
                    if (!resp.ok) throw new Error(data.error || 'Failed to load sample');
                    showOriginalFromUrl(data.image);
                    setStatus(`Using sample ${data.filename}. Ready to run.`, 'info');
                    runBtn.disabled = false;
                } catch (err) {
                    fileInput.disabled = false;
                    useSampleCheckbox.checked = false;
                    setStatus(`Could not load sample: ${err.message}`, 'error');
                }
            } else {
                fileInput.disabled = false;
                clearResults();
                setStatus(defaultStatus, 'info');
            }
        });
    }

    resetBtn.addEventListener('click', resetModule);

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!hasFileSelected()) {
            setStatus('Please select an image before running the process.', 'error');
            return;
        }

        const formData = new FormData();
        if (useSampleCheckbox && useSampleCheckbox.checked) {
            // signal server to use the pre-configured sample
            formData.append('sample', 'tree.jpg');
        } else {
            formData.append('image', fileInput.files[0]);
        }

        runBtn.disabled = true;
        resetBtn.disabled = true;
        setStatus('Uploading image and running the Gaussian blur + inverse filtering process...', 'info');

        try {
            const response = await fetch('/api/a2/part2', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Process failed');
            }

            showImage(blurredImg, blurredPlaceholder, data.blurredImage);
            showImage(restoredImg, restoredPlaceholder, data.restoredImage);
            resetBtn.disabled = false;
            const extraBits = [];
            if (Number.isFinite(data.kernelSize)) {
                extraBits.push(`kernel ${data.kernelSize}`);
            }
            if (Number.isFinite(data.sigma)) {
                extraBits.push(`sigma ${Number(data.sigma).toFixed(2)}`);
            }
            const meta = extraBits.length ? ` (${extraBits.join(', ')})` : '';
            setStatus(`Process complete for ${data.originalFilename || 'uploaded image'}${meta}.`, 'success');
        } catch (err) {
            clearResults({ keepOriginal: true });
            setStatus(err.message || 'Unexpected error occurred.', 'error');
        } finally {
            if (hasFileSelected()) {
                runBtn.disabled = false;
            }
            if (hasFileSelected() || hasResults()) {
                resetBtn.disabled = false;
            }
        }
    });
    registerGlobalResetHandler(resetModule);
}

function initModule2Part1Flow() {
    const form = document.getElementById('module2-part1-form');
    if (!form) return;

    const thresholdInput = document.getElementById('module2-part1-threshold');
    const runBtn = document.getElementById('module2-part1-run-btn');
    const clearBtn = document.getElementById('module2-part1-clear-btn');
    const statusLine = document.getElementById('module2-part1-status');
    const sceneImg = document.getElementById('module2-part1-scene');
    const scenePlaceholder = document.getElementById('module2-part1-scene-placeholder');
    const gallery = document.getElementById('module2-part1-gallery');
    const galleryPlaceholder = document.getElementById('module2-part1-gallery-placeholder');
    const matchCountEl = document.getElementById('module2-part1-match-count');
    const totalCountEl = document.getElementById('module2-part1-total');

    if (!thresholdInput || !runBtn || !clearBtn || !statusLine || !gallery || !matchCountEl || !totalCountEl) {
        return;
    }

    let sceneLoaded = false;
    let hasResults = false;
    const defaultStatus = (statusLine.textContent || 'Load the Module 2 tab to view the reference scene, then run the detector.').trim();

    const setStatus = (message, variant = 'info') => {
        statusLine.textContent = message;
        statusLine.dataset.variant = variant;
    };

    const setSceneImage = (dataUrl, alt = 'Reference scene for template matching') => {
        if (!sceneImg) return;
        if (dataUrl) {
            sceneImg.src = dataUrl;
            sceneImg.alt = alt;
            sceneImg.hidden = false;
            sceneImg.removeAttribute('aria-hidden');
            scenePlaceholder && (scenePlaceholder.hidden = true);
            const wrapper = sceneImg.closest('.result-image');
            if (wrapper) wrapper.dataset.empty = 'false';
            sceneLoaded = true;
            markModule2ImageInteractive(sceneImg);
        } else {
            sceneImg.hidden = true;
            if (scenePlaceholder) scenePlaceholder.hidden = false;
            const wrapper = sceneImg.closest('.result-image');
            if (wrapper) wrapper.dataset.empty = 'true';
        }
    };

    const templateLabel = (name) => {
        if (!name) return 'Unknown template';
        return name
            .replace(/^template_/i, '')
            .replace(/_/g, ' ')
            .replace(/\.png$/i, '')
            .trim() || name;
    };

    const renderMatches = (matches = []) => {
        gallery.innerHTML = '';
        if (!matches.length) {
            gallery.dataset.empty = 'true';
            if (galleryPlaceholder) galleryPlaceholder.hidden = false;
            return;
        }

        gallery.dataset.empty = 'false';
        if (galleryPlaceholder) galleryPlaceholder.hidden = true;
        matches.forEach((match) => {
            const card = document.createElement('article');
            card.className = 'match-card';
            card.dataset.result = match.matched ? 'hit' : 'miss';

            const header = document.createElement('div');
            header.className = 'match-card__header';

            const labelSpan = document.createElement('span');
            labelSpan.className = 'match-chip';
            labelSpan.textContent = match.matched ? 'Match' : 'No match';

            const title = document.createElement('strong');
            title.textContent = templateLabel(match.template);

            header.appendChild(labelSpan);
            header.appendChild(title);
            card.appendChild(header);

            if (match.matched && match.outputImage) {
                const img = document.createElement('img');
                img.src = match.outputImage;
                img.alt = `Detected ${templateLabel(match.template)}`;
                img.loading = 'lazy';
                img.className = 'match-card__image';
                markModule2ImageInteractive(img);
                card.appendChild(img);
            }

            const message = document.createElement('p');
            message.className = 'match-card__message';
            if (match.error) {
                message.textContent = `Error: ${match.error}`;
            } else if (match.matched) {
                message.textContent = 'Correlation above threshold – see highlighted regions.';
            } else {
                message.textContent = 'No detections cleared the chosen threshold.';
            }
            card.appendChild(message);

            gallery.appendChild(card);
        });
    };

    const updateSummary = (matched = 0, total = 0) => {
        matchCountEl.textContent = matched;
        totalCountEl.textContent = total;
    };

    const clearUI = (opts = { resetSummary: true }) => {
        renderMatches([]);
        hasResults = false;
        clearBtn.disabled = true;
        if (opts.resetSummary) {
            updateSummary(0, 0);
        }
    };

    const clearServerResults = async (options = { silent: false, keepalive: false }) => {
        try {
            const usePost = Boolean(options.keepalive);
            const fetchConfig = {
                method: usePost ? 'POST' : 'DELETE',
                keepalive: Boolean(options.keepalive),
            };
            if (usePost) {
                fetchConfig.headers = { 'Content-Type': 'application/json' };
                fetchConfig.body = JSON.stringify({ cleanup: true });
            }
            await fetch('/api/a2/part1/results', fetchConfig);
        } catch (err) {
            if (!options.silent) {
                console.warn('Failed to clear Module 2 Part 1 outputs on server.', err);
            }
        }
    };

    const resetModule = async ({ resetSummary = true, statusMessage } = {}) => {
        clearUI({ resetSummary });
        setStatus(statusMessage || defaultStatus, 'info');
        await clearServerResults({ silent: true });
    };

    const loadScenePreview = async () => {
        if (sceneLoaded) return;
        try {
            const response = await fetch('/api/a2/part1/scene');
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Unable to load scene.');
            setSceneImage(data.image, `Reference scene (${data.filename || 'scene.jpg'})`);
            if (scenePlaceholder) scenePlaceholder.textContent = 'Scene preview ready.';
        } catch (err) {
            setSceneImage(null);
            setStatus(err.message, 'error');
        }
    };

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const thresholdValue = parseFloat(thresholdInput.value);
        if (Number.isNaN(thresholdValue) || thresholdValue < 0 || thresholdValue > 1) {
            setStatus('Please enter a threshold between 0.0 and 1.0.', 'error');
            return;
        }

        setStatus('Running correlation across all templates…', 'info');
        runBtn.disabled = true;
        thresholdInput.disabled = true;

        try {
            const response = await fetch('/api/a2/part1', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ threshold: thresholdValue }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Correlation run failed.');

            if (data.scene?.image) {
                setSceneImage(data.scene.image, `Reference scene (${data.scene.filename || 'scene'})`);
            }
            renderMatches(data.matches || []);
            const matched = data.summary?.matched ?? 0;
            const total = data.summary?.total ?? (data.matches ? data.matches.length : 0);
            updateSummary(matched, total);
            hasResults = true;
            clearBtn.disabled = false;

            if (matched > 0) {
                setStatus(`Detected ${matched} object(s) using threshold ${thresholdValue.toFixed(2)}.`, 'success');
            } else {
                setStatus(`No templates passed the threshold ${thresholdValue.toFixed(2)}.`, 'warning');
            }
        } catch (err) {
            clearUI({ resetSummary: false });
            setStatus(err.message || 'Unexpected error while running correlation.', 'error');
        } finally {
            runBtn.disabled = false;
            thresholdInput.disabled = false;
        }
    });

    clearBtn.addEventListener('click', async () => {
        await resetModule({ statusMessage: 'Cleared generated match files.' });
    });

    registerTabChangeListener((prev, next) => {
        if (prev === 'a2' && next !== 'a2') {
            clearUI({ resetSummary: true });
            clearServerResults({ silent: true });
        } else if (next === 'a2') {
            loadScenePreview();
        }
    });

    window.addEventListener('beforeunload', () => {
        clearServerResults({ silent: true, keepalive: true });
    });

    registerGlobalResetHandler(() => resetModule({ resetSummary: true }));

    loadScenePreview();
}

function initModule2Part3Flow() {
    const form = document.getElementById('module2-part3-form');
    if (!form) return;

    const thresholdInput = document.getElementById('module2-part3-threshold');
    const blurInput = document.getElementById('module2-part3-blur');
    const runBtn = document.getElementById('module2-part3-run-btn');
    const resetBtn = document.getElementById('module2-part3-reset-btn');
    const statusLine = document.getElementById('module2-part3-status');
    const detectionCountEl = document.getElementById('module2-part3-detection-count');
    const templateCountEl = document.getElementById('module2-part3-template-count');
    const detectionLog = document.getElementById('module2-part3-detection-log');

    const sceneImg = document.getElementById('module2-part3-scene');
    const scenePlaceholder = document.getElementById('module2-part3-scene-placeholder');
    const detectionsImg = document.getElementById('module2-part3-detections');
    const detectionsPlaceholder = document.getElementById('module2-part3-detections-placeholder');
    const blurredImg = document.getElementById('module2-part3-blurred');
    const blurredPlaceholder = document.getElementById('module2-part3-blurred-placeholder');

    const templatesGrid = document.getElementById('module2-part3-templates-grid');
    const templatesPlaceholder = document.getElementById('module2-part3-templates-placeholder');

    if (!thresholdInput || !blurInput || !runBtn || !resetBtn || !statusLine || !detectionCountEl || !templateCountEl || !detectionLog || !sceneImg || !detectionsImg || !blurredImg || !templatesGrid) {
        return;
    }

    let referencesLoaded = false;
    let hasResults = false;
    const defaultStatus = (statusLine.textContent || 'Reference assets will load automatically — adjust the parameters and run whenever you are ready.').trim();

    const setStatus = (message, variant = 'info') => {
        statusLine.textContent = message;
        statusLine.dataset.variant = variant;
    };

    const setImageState = (imgEl, placeholderEl, dataUrl) => {
        if (!imgEl) return;
        const wrapper = imgEl.closest('.result-image');
        if (dataUrl) {
            imgEl.src = dataUrl;
            imgEl.hidden = false;
            imgEl.dataset.loaded = 'true';
            if (wrapper) wrapper.dataset.empty = 'false';
            if (placeholderEl) placeholderEl.hidden = true;
            markModule2ImageInteractive(imgEl);
        } else {
            imgEl.removeAttribute('src');
            imgEl.hidden = true;
            imgEl.dataset.loaded = 'false';
            if (wrapper) wrapper.dataset.empty = 'true';
            if (placeholderEl) placeholderEl.hidden = false;
        }
    };

    const formatLabel = (label) => {
        if (!label) return 'Unknown template';
        return label.replace(/^template_/i, '').replace(/_/g, ' ').trim() || label;
    };

    const renderDetections = (detections = []) => {
        detectionLog.dataset.empty = detections.length ? 'false' : 'true';
        detectionLog.innerHTML = '';
        if (!detections.length) {
            detectionLog.textContent = 'Run the detector to list every match, including correlation scores.';
            return;
        }

        detections.forEach((det, index) => {
            const entry = document.createElement('p');
            entry.className = 'module2-part3-log__item';
            const prettyLabel = formatLabel(det.label);
            const score = Number(det.score).toFixed(2);
            entry.innerHTML = `<strong>${index + 1}. ${prettyLabel}</strong> – score ${score}`;
            detectionLog.appendChild(entry);
        });
    };

    const renderTemplates = (templates = []) => {
        templateCountEl.textContent = templates.length;
        templatesGrid.innerHTML = '';
        if (!templates.length) {
            templatesGrid.dataset.empty = 'true';
            if (templatesPlaceholder) templatesPlaceholder.hidden = false;
            return;
        }
        templatesGrid.dataset.empty = 'false';
        if (templatesPlaceholder) templatesPlaceholder.hidden = true;

        templates.forEach((tpl) => {
            const card = document.createElement('figure');
            card.className = 'template-card';

            const img = document.createElement('img');
            img.src = tpl.image;
            img.alt = tpl.filename || 'Template image';
            img.loading = 'lazy';
            markModule2ImageInteractive(img);
            card.appendChild(img);

            const caption = document.createElement('figcaption');
            caption.textContent = formatLabel(tpl.label || tpl.filename);
            card.appendChild(caption);

            templatesGrid.appendChild(card);
        });
    };

    const clearResults = () => {
        hasResults = false;
        setImageState(detectionsImg, detectionsPlaceholder, null);
        setImageState(blurredImg, blurredPlaceholder, null);
        detectionCountEl.textContent = '0';
        renderDetections([]);
        resetBtn.disabled = true;
    };

    const resetModule = ({ preserveReferences = true, statusMessage } = {}) => {
        form.reset();
        clearResults();
        if (!preserveReferences) {
            templateCountEl.textContent = '0';
            templatesGrid.dataset.empty = 'true';
            templatesGrid.innerHTML = '';
            let placeholder = templatesPlaceholder;
            if (!placeholder) {
                placeholder = document.createElement('p');
                placeholder.id = 'module2-part3-templates-placeholder';
                placeholder.textContent = 'Fetching template thumbnails…';
            }
            placeholder.hidden = false;
            templatesGrid.appendChild(placeholder);
            referencesLoaded = false;
        }
        setStatus(statusMessage || defaultStatus, 'info');
    };

    const loadReferences = async () => {
        if (referencesLoaded) return;
        try {
            const response = await fetch('/api/a2/part3/references');
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Unable to load references.');

            if (data.scene?.image) {
                setImageState(sceneImg, scenePlaceholder, data.scene.image);
            }
            renderTemplates(data.templates || []);
            setStatus('Reference assets loaded. Set your parameters and run detection.', 'info');
            referencesLoaded = true;
        } catch (err) {
            setStatus(err.message || 'Failed to load Module 2 Part 3 references.', 'error');
        }
    };

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const thresholdValue = parseFloat(thresholdInput.value);
        const blurValue = parseFloat(blurInput.value);

        if (Number.isNaN(thresholdValue) || thresholdValue < 0 || thresholdValue > 1) {
            setStatus('Please enter a threshold between 0.0 and 1.0.', 'error');
            return;
        }

        if (Number.isNaN(blurValue) || blurValue <= 0 || blurValue > 25) {
            setStatus('Blur multiplier must be between 0.1 and 25.', 'error');
            return;
        }

        runBtn.disabled = true;
        resetBtn.disabled = true;
        setStatus('Running multi-template detection…', 'info');

        try {
            const response = await fetch('/api/a2/part3', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ threshold: thresholdValue, blurMultiplier: blurValue })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Detection failed.');

            detectionCountEl.textContent = data.summary?.detected ?? data.detections?.length ?? 0;
            if (Number.isFinite(data.summary?.templatesTested)) {
                templateCountEl.textContent = data.summary.templatesTested;
            }
            if (data.detectionsImage) {
                setImageState(detectionsImg, detectionsPlaceholder, data.detectionsImage);
            } else {
                setImageState(detectionsImg, detectionsPlaceholder, null);
            }
            if (data.blurredImage) {
                setImageState(blurredImg, blurredPlaceholder, data.blurredImage);
            } else {
                setImageState(blurredImg, blurredPlaceholder, null);
            }
            renderDetections(data.detections || []);
            setStatus(data.message || 'Detection complete.', data.summary?.detected ? 'success' : 'warning');
            hasResults = true;
            resetBtn.disabled = false;
        } catch (err) {
            clearResults();
            setStatus(err.message || 'Unexpected error while running detection.', 'error');
        } finally {
            runBtn.disabled = false;
        }
    });

    resetBtn.addEventListener('click', () =>
        resetModule({ preserveReferences: true, statusMessage: 'Outputs cleared. Adjust the parameters and run again.' })
    );

    registerTabChangeListener((prev, next) => {
        if (prev === 'a2' && next !== 'a2' && hasResults) {
            clearResults();
            setStatus('Outputs cleared after leaving Module 2.', 'info');
        }
        if (next === 'a2') {
            loadReferences();
        }
    });

    registerGlobalResetHandler(() => resetModule({ preserveReferences: false }));

    loadReferences();
}

const MODULE3_MIN_IMAGES = 10;

function initModule3Flow() {
    const configs = [
        {
            key: 'part1',
            formId: 'module3-part1-form',
            inputId: 'module3-part1-input',
            selectionId: 'module3-part1-selection',
            sampleBtnId: 'module3-part1-sample',
            runBtnId: 'module3-part1-run',
            resetBtnId: 'module3-part1-reset',
            statusId: 'module3-part1-status',
            galleryId: 'module3-part1-gallery',
            summaryId: 'module3-part1-summary',
            endpoint: '/api/a3/part1',
            readyStatus: 'Ready to run gradient & LoG analysis.',
            runningStatus: 'Processing gradients and Laplacian responses…',
            outputLabel: 'Gradient / LoG canvas',
            successStatus: 'Gradient grids generated.',
            enableLightbox: true
        },
        {
            key: 'part2',
            formId: 'module3-part2-form',
            inputId: 'module3-part2-input',
            selectionId: 'module3-part2-selection',
            sampleBtnId: 'module3-part2-sample',
            runBtnId: 'module3-part2-run',
            resetBtnId: 'module3-part2-reset',
            statusId: 'module3-part2-status',
            galleryId: 'module3-part2-gallery',
            summaryId: 'module3-part2-summary',
            endpoint: '/api/a3/part2',
            readyStatus: 'Ready to detect edges and corners.',
            runningStatus: 'Running edge & corner detector…',
            outputLabel: 'Edge & corner overlay',
            successStatus: 'Edge/corner map created.',
            enableLightbox: true
        },
        {
            key: 'part3',
            formId: 'module3-part3-form',
            inputId: 'module3-part3-input',
            selectionId: 'module3-part3-selection',
            sampleBtnId: 'module3-part3-sample',
            runBtnId: 'module3-part3-run',
            resetBtnId: 'module3-part3-reset',
            statusId: 'module3-part3-status',
            galleryId: 'module3-part3-gallery',
            summaryId: 'module3-part3-summary',
            endpoint: '/api/a3/part3',
            readyStatus: 'Ready to extract object boundaries.',
            runningStatus: 'Running contour extraction — please be patient while the results render…',
            outputLabel: 'Boundary overlay',
            successStatus: 'Boundary overlay generated.',
            showProcessingIndicator: true,
            enableLightbox: true
        },
        {
            key: 'part4',
            formId: 'module3-part4-form',
            inputId: 'module3-part4-input',
            selectionId: 'module3-part4-selection',
            sampleBtnId: 'module3-part4-sample',
            runBtnId: 'module3-part4-run',
            resetBtnId: 'module3-part4-reset',
            statusId: 'module3-part4-status',
            galleryId: 'module3-part4-gallery',
            summaryId: 'module3-part4-summary',
            endpoint: '/api/a3/part4',
            sampleEndpoint: '/api/a3/part4/samples',
            readyStatus: 'Ready to run the marker-guided cutout.',
            runningStatus: 'Detecting ArUco markers and running GrabCut — please be patient…',
            outputLabel: 'Marker-guided cutout',
            successStatus: 'Marker-guided cutouts generated.',
            enableLightbox: true,
            showProcessingIndicator: true
        }
    ];

    configs.forEach((cfg) => setupModule3Part(cfg));
}

function initModule3Part5Showcase() {
    const gallery = document.getElementById('module3-part5-gallery');
    const summaryEl = document.getElementById('module3-part5-summary');
    if (!gallery || !summaryEl) return;

    const refreshBtn = document.getElementById('module3-part5-refresh');
    const commandEl = document.getElementById('module3-part5-command');
    const depsEl = document.getElementById('module3-part5-deps');
    let isLoading = false;
    let hasLoaded = false;

    const initCliCopyButtons = () => {
        const snippets = document.querySelectorAll('.module3-cli-snippet');
        snippets.forEach((snippet) => {
            const button = snippet.querySelector('.module3-copy-btn');
            const toast = snippet.querySelector('.module3-copy-toast');
            if (!button) return;

            let toastTimer = null;

            const showToast = () => {
                if (!toast) return;
                toast.classList.add('module3-copy-toast--visible');
                if (toastTimer) clearTimeout(toastTimer);
                toastTimer = setTimeout(() => {
                    toast.classList.remove('module3-copy-toast--visible');
                }, 1800);
            };

            button.addEventListener('click', async () => {
                const targetId = button.dataset.copyTarget;
                const target = targetId ? document.getElementById(targetId) : null;
                const text = target ? target.textContent.trim() : '';
                if (!text) return;

                try {
                    if (navigator?.clipboard?.writeText) {
                        await navigator.clipboard.writeText(text);
                    } else {
                        throw new Error('Clipboard API unavailable');
                    }
                } catch (err) {
                    const textarea = document.createElement('textarea');
                    textarea.value = text;
                    textarea.style.position = 'fixed';
                    textarea.style.opacity = '0';
                    document.body.appendChild(textarea);
                    textarea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textarea);
                }

                showToast();
            });
        });
    };

    const setSummary = (message) => {
        summaryEl.textContent = message;
    };

    const setPlaceholder = (message, variant = 'info') => {
        gallery.dataset.empty = 'true';
        gallery.innerHTML = '';
        const note = document.createElement('p');
        note.className = 'module3-gallery__placeholder';
        note.dataset.variant = variant;
        note.textContent = message;
        gallery.appendChild(note);
    };

    const createFigure = (src, label, filename, extraClass) => {
        const figure = document.createElement('figure');
        figure.className = 'module3-result-figure';
        figure.classList.add('module3-result-figure--interactive');
        if (extraClass) {
            figure.classList.add(extraClass);
        }

        if (src) {
            const img = document.createElement('img');
            img.src = src;
            img.alt = `${label} – ${filename || 'image'}`;
            img.loading = 'lazy';
            img.tabIndex = 0;
            figure.appendChild(img);
        } else {
            const placeholder = document.createElement('div');
            placeholder.className = 'module3-result-placeholder';
            placeholder.textContent = 'Preview unavailable';
            figure.appendChild(placeholder);
        }

        const caption = document.createElement('figcaption');
        caption.textContent = label;
        figure.appendChild(caption);
        return figure;
    };

    const renderGallery = (entries) => {
        const items = Array.isArray(entries) ? entries : [];
        if (!items.length) {
            setPlaceholder('No showcase frames are available right now.', 'warning');
            return;
        }

        gallery.dataset.empty = 'false';
        gallery.innerHTML = '';
        items.forEach((entry, index) => {
            const card = document.createElement('article');
            card.className = 'module3-result-card module3-result-card--showcase';

            const header = document.createElement('div');
            header.className = 'module3-result-card__header';
            const title = document.createElement('strong');
            title.textContent = `${index + 1}. ${entry.filename || 'Image'}`;
            header.appendChild(title);

            const tag = document.createElement('span');
            tag.className = 'module3-result-tag';
            tag.textContent = 'ArUco vs SAM2';
            header.appendChild(tag);
            card.appendChild(header);

            const figures = document.createElement('div');
            figures.className = 'module3-result-images';
            figures.appendChild(createFigure(entry.originalImage, 'ArUco GrabCut baseline', entry.filename, 'module3-result-figure--baseline'));
            figures.appendChild(createFigure(entry.samImage, 'SAM2 segmentation', entry.filename, 'module3-result-figure--sam'));
            card.appendChild(figures);

            gallery.appendChild(card);
        });
    };

    const loadGallery = async ({ force = false } = {}) => {
        if (isLoading) return;
        if (hasLoaded && !force) return;
        isLoading = true;
        setPlaceholder('Loading offline comparisons…', 'info');

        try {
            const response = await fetch('/api/a3/part5/gallery');
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to load showcase.');

            renderGallery(data.results || []);
            if (data.metadata?.command && commandEl) {
                commandEl.textContent = data.metadata.command;
            }
            if (data.metadata?.pipPackages && Array.isArray(data.metadata.pipPackages) && depsEl) {
                depsEl.textContent = `pip install ${data.metadata.pipPackages.join(' ')}`;
            }
            const count = data.count ?? (data.results ? data.results.length : 0);
            const descriptor = data.metadata?.model ? `${data.metadata.model} on ${data.metadata?.device || 'GPU'}` : 'SAM2 run';
            setSummary(data.message || `Loaded ${count} comparisons from the offline ${descriptor}.`);
            hasLoaded = true;
        } catch (err) {
            console.error(err);
            setPlaceholder(err.message || 'Failed to load showcase.', 'error');
            setSummary(err.message || 'Failed to load showcase.');
        } finally {
            isLoading = false;
        }
    };

    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            hasLoaded = false;
            loadGallery({ force: true });
        });
    }

    gallery.addEventListener('click', (event) => {
        const img = event.target.closest('img');
        if (!img || !gallery.contains(img)) return;
        openModule3Lightbox(img.src, img.alt);
    });

    gallery.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        const img = event.target;
        if (!img || img.tagName !== 'IMG' || !gallery.contains(img)) return;
        event.preventDefault();
        openModule3Lightbox(img.src, img.alt);
    });

    const activeTab = document.getElementById('a3');
    if (activeTab && activeTab.classList.contains('active')) {
        loadGallery();
    }

    registerTabChangeListener((_, next) => {
        if (next === 'a3') {
            loadGallery();
        }
    });

    initCliCopyButtons();
    ensureModule3Lightbox();
}

function setupModule3Part(config) {
    const form = document.getElementById(config.formId);
    if (!form) return;

    const fileInput = document.getElementById(config.inputId);
    const sampleBtn = document.getElementById(config.sampleBtnId);
    const runBtn = document.getElementById(config.runBtnId);
    const resetBtn = document.getElementById(config.resetBtnId);
    const statusLine = document.getElementById(config.statusId);
    const selectionEl = document.getElementById(config.selectionId);
    const gallery = document.getElementById(config.galleryId);
    const summaryEl = document.getElementById(config.summaryId);

    if (!fileInput || !runBtn || !resetBtn || !statusLine || !selectionEl || !gallery || !summaryEl) {
        return;
    }

    const showProcessingIndicator = Boolean(config.showProcessingIndicator);
    const defaultStatus = (statusLine.textContent || 'Upload ≥10 images or use the example dataset to begin.').trim();
    const defaultSummary = summaryEl.textContent || '';
    const defaultSelectionHtml = selectionEl.innerHTML;
    const defaultGalleryHtml = gallery.innerHTML;
    const state = {
        useSample: false,
        sampleFiles: [],
        hasResults: false
    };

    const statusText = document.createElement('span');
    statusText.className = 'status-line__text';
    statusText.textContent = defaultStatus;
    const statusSpinner = document.createElement('span');
    statusSpinner.className = 'status-spinner';
    statusSpinner.setAttribute('aria-hidden', 'true');
    statusLine.textContent = '';
    statusLine.append(statusSpinner, statusText);

    if (config.enableLightbox) {
        gallery.classList.add('module3-gallery--interactive');
    }

    const setStatus = (message, variant = 'info', options) => {
        statusText.textContent = message;
        statusLine.dataset.variant = variant;
        if (showProcessingIndicator) {
            if (options && options.loading) {
                statusLine.dataset.loading = 'true';
            } else {
                statusLine.removeAttribute('data-loading');
            }
        }
    };

    const currentCount = () => (state.useSample ? state.sampleFiles.length : (fileInput.files ? fileInput.files.length : 0));

    const refreshButtons = () => {
        const count = currentCount();
        runBtn.disabled = count < MODULE3_MIN_IMAGES;
        resetBtn.disabled = !(count || state.hasResults);
    };

    const renderSelectionList = (names, note) => {
        selectionEl.dataset.empty = 'false';
        selectionEl.innerHTML = '';
        if (note) {
            const noteEl = document.createElement('p');
            noteEl.className = 'module3-selection__note';
            noteEl.textContent = note;
            selectionEl.appendChild(noteEl);
        }
        const list = document.createElement('ol');
        list.className = 'module3-file-list';
        names.forEach((name) => {
            const item = document.createElement('li');
            item.textContent = name;
            list.appendChild(item);
        });
        selectionEl.appendChild(list);
    };

    const resetSelection = () => {
        selectionEl.dataset.empty = 'true';
        selectionEl.innerHTML = defaultSelectionHtml;
    };

    const clearSampleMode = () => {
        if (!state.useSample) return;
        state.useSample = false;
        state.sampleFiles = [];
    };

    const updateSelection = () => {
        if (state.useSample && state.sampleFiles.length) {
            renderSelectionList(state.sampleFiles, `Example dataset (${state.sampleFiles.length} images)`);
            refreshButtons();
            return;
        }

        const files = Array.from(fileInput.files || []);
        if (!files.length) {
            resetSelection();
            refreshButtons();
            return;
        }

        renderSelectionList(files.map((file) => file.name));
        refreshButtons();
    };

    const clearGallery = () => {
        gallery.dataset.empty = 'true';
        gallery.innerHTML = defaultGalleryHtml;
        state.hasResults = false;
        summaryEl.textContent = defaultSummary;
    };

    const createFigure = (src, label, filename) => {
        const figure = document.createElement('figure');
        figure.className = 'module3-result-figure';
        if (config.enableLightbox) {
            figure.classList.add('module3-result-figure--interactive');
        }
        if (src) {
            const img = document.createElement('img');
            img.src = src;
            img.alt = label ? `${label} (${filename || 'image'})` : (filename || 'image');
            img.loading = 'lazy';
            if (config.enableLightbox) {
                img.tabIndex = 0;
            }
            figure.appendChild(img);
        } else {
            const placeholder = document.createElement('div');
            placeholder.className = 'module3-result-placeholder';
            placeholder.textContent = 'No preview available';
            figure.appendChild(placeholder);
        }
        const caption = document.createElement('figcaption');
        caption.textContent = label;
        figure.appendChild(caption);
        return figure;
    };

    const renderGallery = (results) => {
        const items = Array.isArray(results) ? results : [];
        if (!items.length) {
            clearGallery();
            return 0;
        }

        gallery.dataset.empty = 'false';
        gallery.innerHTML = '';
        items.forEach((result, idx) => {
            const card = document.createElement('article');
            card.className = 'module3-result-card';

            const header = document.createElement('div');
            header.className = 'module3-result-card__header';
            const title = document.createElement('strong');
            title.textContent = `${idx + 1}. ${result.filename || 'Image'}`;
            header.appendChild(title);
            card.appendChild(header);

            const figures = document.createElement('div');
            figures.className = 'module3-result-images';
            figures.appendChild(createFigure(result.inputImage, 'Original frame', result.filename));
            figures.appendChild(createFigure(result.outputImage, config.outputLabel || 'Processed output', result.filename));
            card.appendChild(figures);

            gallery.appendChild(card);
        });

        state.hasResults = true;
        refreshButtons();
        return items.length;
    };

    if (config.enableLightbox) {
        const handleLightboxClick = (event) => {
            const img = event.target.closest('img');
            if (!img || !gallery.contains(img)) return;
            openModule3Lightbox(img.src, img.alt);
        };

        const handleLightboxKeydown = (event) => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            const img = event.target;
            if (!img || img.tagName !== 'IMG' || !gallery.contains(img)) return;
            event.preventDefault();
            openModule3Lightbox(img.src, img.alt);
        };

        gallery.addEventListener('click', handleLightboxClick);
        gallery.addEventListener('keydown', handleLightboxKeydown);
    }

    const performReset = ({ keepStatus = false } = {}) => {
        form.reset();
        try { fileInput.value = ''; } catch (err) { /* ignore */ }
        clearSampleMode();
        updateSelection();
        clearGallery();
        if (!keepStatus) {
            setStatus(defaultStatus, 'info');
        }
        refreshButtons();
    };

    fileInput.addEventListener('change', () => {
        if (state.useSample) {
            clearSampleMode();
        }
        updateSelection();
        clearGallery();
        if (currentCount() >= MODULE3_MIN_IMAGES) {
            setStatus(config.readyStatus || defaultStatus, 'info');
        } else {
            setStatus(defaultStatus, 'info');
        }
    });

    if (sampleBtn) {
        sampleBtn.addEventListener('click', async () => {
            sampleBtn.disabled = true;
            setStatus('Loading example dataset…', 'info');
            try {
                const sampleEndpoint = config.sampleEndpoint || '/api/a3/samples';
                const resp = await fetch(sampleEndpoint);
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.error || 'Failed to load example dataset.');
                if (!Array.isArray(data.filenames) || data.filenames.length < MODULE3_MIN_IMAGES) {
                    throw new Error(`Example dataset must contain at least ${MODULE3_MIN_IMAGES} images.`);
                }
                state.useSample = true;
                state.sampleFiles = data.filenames.slice();
                try { fileInput.value = ''; } catch (err) { /* ignore */ }
                updateSelection();
                clearGallery();
                setStatus(`Loaded example dataset (${state.sampleFiles.length} images).`, 'success');
            } catch (err) {
                clearSampleMode();
                updateSelection();
                setStatus(err.message || 'Could not load example dataset.', 'error');
            } finally {
                sampleBtn.disabled = false;
            }
        });
    }

    resetBtn.addEventListener('click', () => performReset());

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const count = currentCount();
        if (count < MODULE3_MIN_IMAGES) {
            setStatus(`Please provide at least ${MODULE3_MIN_IMAGES} images.`, 'error');
            return;
        }

        const formData = new FormData();
        if (state.useSample) {
            formData.append('sample', '1');
        } else {
            Array.from(fileInput.files || []).forEach((file) => formData.append('images', file));
        }

        runBtn.disabled = true;
        resetBtn.disabled = true;
        const loadingOptions = showProcessingIndicator ? { loading: true } : undefined;
        setStatus(config.runningStatus || 'Processing images…', 'info', loadingOptions);

        try {
            const response = await fetch(config.endpoint, { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Processing failed.');

            const processedCount = renderGallery(data.results);
            const datasetLabel = data.source === 'sample' ? 'example dataset' : 'uploaded set';
            const summaryText = data.message || `${processedCount} images processed from the ${datasetLabel}.`;
            summaryEl.textContent = summaryText;
            setStatus(data.message || config.successStatus || 'Processing complete.', 'success');
        } catch (err) {
            setStatus(err.message || 'Unexpected error while processing images.', 'error');
        } finally {
            refreshButtons();
        }
    });

    registerTabChangeListener((prev, next) => {
        if (prev === 'a3' && next !== 'a3') {
            performReset({ keepStatus: true });
        }
    });

    const resetFn = () => performReset({ keepStatus: false });
    form.__module3Reset = resetFn;
    registerGlobalResetHandler(resetFn);

    updateSelection();
    setStatus(defaultStatus, 'info');
}

const MODULE4_MIN_IMAGES = 8;

function initModule4Flow() {
    const configs = [
        {
            key: 'part1',
            formId: 'module4-part1-form',
            inputId: 'module4-part1-input',
            selectionId: 'module4-part1-selection',
            runBtnId: 'module4-part1-run',
            resetBtnId: 'module4-part1-reset',
            statusId: 'module4-part1-status',
            outputImgId: 'module4-part1-output',
            placeholderId: 'module4-part1-placeholder',
            countId: 'module4-part1-count',
            sizeId: 'module4-part1-size',
            downloadId: 'module4-part1-download',
            sampleBtnId: 'module4-part1-sample',
            endpoint: '/api/a4/part1',
            clearEndpoint: '/api/a4/part1/results',
            readyMessage: 'Ready to stitch with the OpenCV pipeline.',
            uploadMessage: 'Uploading images and running the OpenCV stitcher…',
            downloadFilename: 'module4-part1-panorama.png'
        },
        {
            key: 'part2',
            formId: 'module4-part2-form',
            inputId: 'module4-part2-input',
            selectionId: 'module4-part2-selection',
            runBtnId: 'module4-part2-run',
            resetBtnId: 'module4-part2-reset',
            statusId: 'module4-part2-status',
            outputImgId: 'module4-part2-output',
            placeholderId: 'module4-part2-placeholder',
            countId: 'module4-part2-count',
            sizeId: 'module4-part2-size',
            downloadId: 'module4-part2-download',
            sampleBtnId: 'module4-part2-sample',
            endpoint: '/api/a4/part2',
            clearEndpoint: '/api/a4/part2/results',
            readyMessage: 'Ready to run the scratch-built stitcher.',
            uploadMessage: 'Uploading images and running the custom SIFT pipeline — please be patient…',
            downloadFilename: 'module4-part2-panorama.png',
            showProcessingIndicator: true
        }
    ];

    configs.forEach((cfg) => setupModule4Part(cfg));
}

function setupModule4Part(config) {
    const form = document.getElementById(config.formId);
    if (!form) return;

    const fileInput = document.getElementById(config.inputId);
    const selectionEl = document.getElementById(config.selectionId);
    const runBtn = document.getElementById(config.runBtnId);
    const resetBtn = document.getElementById(config.resetBtnId);
    const statusLine = document.getElementById(config.statusId);
    const outputImg = document.getElementById(config.outputImgId);
    const placeholder = document.getElementById(config.placeholderId);
    const countEl = document.getElementById(config.countId);
    const sizeEl = document.getElementById(config.sizeId);
    const downloadBtn = document.getElementById(config.downloadId);

    if (!fileInput || !selectionEl || !runBtn || !resetBtn || !statusLine || !outputImg || !placeholder || !countEl || !sizeEl) {
        return;
    }

    const sampleBtn = document.getElementById(config.sampleBtnId);
    const selectionDefault = selectionEl.innerHTML;
    const placeholderDefault = placeholder.textContent;
    const showProcessingIndicator = Boolean(config.showProcessingIndicator);
    const defaultStatus = (statusLine.textContent || 'Upload 8+ portrait frames to enable stitching.').trim();
    const readyMessage = config.readyMessage || 'Ready to stitch.';
    const uploadMessage = config.uploadMessage || 'Uploading images and stitching…';
    const state = { hasResults: false, downloadUrl: null, useSample: false, sampleFiles: [] };

    const statusText = document.createElement('span');
    statusText.className = 'status-line__text';
    statusText.textContent = defaultStatus;
    const statusSpinner = document.createElement('span');
    statusSpinner.className = 'status-spinner';
    statusSpinner.setAttribute('aria-hidden', 'true');
    statusLine.textContent = '';
    statusLine.append(statusSpinner, statusText);

    const setStatus = (message, variant = 'info', options) => {
        if (!statusLine) return;
        statusText.textContent = message;
        statusLine.dataset.variant = variant;
        if (showProcessingIndicator) {
            if (options && options.loading) {
                statusLine.dataset.loading = 'true';
            } else {
                statusLine.removeAttribute('data-loading');
            }
        }
    };

    const currentFileCount = () => (state.useSample ? state.sampleFiles.length : (fileInput.files ? fileInput.files.length : 0));

    const resetSelection = () => {
        selectionEl.dataset.empty = 'true';
        selectionEl.innerHTML = selectionDefault || '<p>No images selected yet.</p>';
    };

    const renderFileList = (entries, note) => {
        selectionEl.dataset.empty = 'false';
        selectionEl.innerHTML = '';
        if (note) {
            const noteEl = document.createElement('p');
            noteEl.className = 'module4-selection__note';
            noteEl.textContent = note;
            selectionEl.appendChild(noteEl);
        }
        const list = document.createElement('ol');
        list.className = 'module4-file-list';
        entries.forEach((name, index) => {
            const item = document.createElement('li');
            item.textContent = `${index + 1}. ${name}`;
            list.appendChild(item);
        });
        selectionEl.appendChild(list);
    };

    const refreshButtons = () => {
        const count = currentFileCount();
        runBtn.disabled = count < MODULE4_MIN_IMAGES;
        resetBtn.disabled = !(count || state.hasResults);
    };

    const clearSampleMode = () => {
        if (!state.useSample) return;
        state.useSample = false;
        state.sampleFiles = [];
    };

    const updateSelection = () => {
        if (state.useSample && state.sampleFiles.length) {
            renderFileList(state.sampleFiles, `Example dataset (${state.sampleFiles.length} images)`);
            refreshButtons();
            return;
        }

        const files = Array.from(fileInput.files || []);
        if (!files.length) {
            resetSelection();
            refreshButtons();
            return;
        }

        renderFileList(files.map((file) => file.name));
        refreshButtons();
    };

    const clearOutput = () => {
        outputImg.removeAttribute('src');
        outputImg.hidden = true;
        placeholder.hidden = false;
        if (placeholderDefault) placeholder.textContent = placeholderDefault;
        countEl.textContent = '0';
        sizeEl.textContent = '-- × --';
        if (downloadBtn) downloadBtn.hidden = true;
        state.downloadUrl = null;
        state.hasResults = false;
    };

    const performReset = ({ clearServer = true, keepStatus = false } = {}) => {
        form.reset();
        try { fileInput.value = ''; } catch (err) { /* ignore readonly errors */ }
        clearSampleMode();
        clearOutput();
        updateSelection();
        runBtn.disabled = true;
        resetBtn.disabled = true;
        if (!keepStatus) setStatus(defaultStatus, 'info');
        if (clearServer && config.clearEndpoint) {
            fetch(config.clearEndpoint, { method: 'DELETE', keepalive: true }).catch(() => { /* best effort */ });
        }
    };

    const handleDownload = () => {
        if (!state.downloadUrl) return;
        const link = document.createElement('a');
        link.href = state.downloadUrl;
        link.download = config.downloadFilename || `${config.key || 'module4'}-panorama.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const updateButtonsAfterSubmit = () => {
        refreshButtons();
    };

    const showResult = (payload, fileCount) => {
        if (payload.panorama) {
            outputImg.src = payload.panorama;
            outputImg.hidden = false;
            placeholder.hidden = true;
            markModule4ImageInteractive(outputImg);
            state.downloadUrl = payload.panorama;
            if (downloadBtn) downloadBtn.hidden = false;
        }
        countEl.textContent = Number(payload.count || fileCount || 0).toString();
        if (Number.isFinite(payload.width) && Number.isFinite(payload.height)) {
            sizeEl.textContent = `${payload.width} × ${payload.height}`;
        } else {
            sizeEl.textContent = '-- × --';
        }
        state.hasResults = true;
        resetBtn.disabled = false;
        setStatus(payload.message || 'Panorama generated successfully.', 'success');
    };

    fileInput.addEventListener('change', () => {
        if (state.useSample) {
            clearSampleMode();
        }
        updateSelection();
        const count = currentFileCount();
        if (count >= MODULE4_MIN_IMAGES) {
            setStatus(readyMessage, 'info');
        } else {
            setStatus(defaultStatus, 'info');
        }
    });

    if (sampleBtn) {
        sampleBtn.addEventListener('click', async () => {
            sampleBtn.disabled = true;
            setStatus('Loading example data…', 'info');
            try {
                const resp = await fetch('/api/a4/samples');
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.error || 'Failed to load example dataset.');
                if (!Array.isArray(data.filenames) || data.filenames.length < MODULE4_MIN_IMAGES) {
                    throw new Error(`Example dataset must contain at least ${MODULE4_MIN_IMAGES} images. Found ${data.filenames?.length || 0}.`);
                }
                state.useSample = true;
                state.sampleFiles = data.filenames.slice();
                try { fileInput.value = ''; } catch (err) { /* ignore */ }
                updateSelection();
                setStatus(`Loaded example dataset (${state.sampleFiles.length} images).`, 'success');
            } catch (err) {
                clearSampleMode();
                updateSelection();
                setStatus(err.message || 'Could not load example dataset.', 'error');
            } finally {
                sampleBtn.disabled = false;
            }
        });
    }

    resetBtn.addEventListener('click', () => {
        performReset({ clearServer: true });
    });

    if (downloadBtn) {
        downloadBtn.addEventListener('click', handleDownload);
    }

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const files = Array.from(fileInput.files || []);
        const fileCount = currentFileCount();
        if (fileCount < MODULE4_MIN_IMAGES) {
            setStatus(`Please select at least ${MODULE4_MIN_IMAGES} portrait images.`, 'error');
            return;
        }

        const formData = new FormData();
        if (state.useSample) {
            formData.append('sample', 'm4');
        } else {
            files.forEach((file) => formData.append('images', file));
        }

        runBtn.disabled = true;
        resetBtn.disabled = true;
        const loadingOptions = showProcessingIndicator ? { loading: true } : undefined;
        setStatus(uploadMessage, 'info', loadingOptions);

        try {
            const response = await fetch(config.endpoint, { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Stitching failed.');
            showResult(data, fileCount);
        } catch (err) {
            setStatus(err.message || 'Unexpected error while stitching.', 'error');
        } finally {
            updateButtonsAfterSubmit();
        }
    });

    registerTabChangeListener((prev, next) => {
        if (prev === 'a4' && next !== 'a4') {
            performReset({ clearServer: state.hasResults || (fileInput.files && fileInput.files.length > 0) });
        }
    });

    const resetFn = (options) => performReset(options || { clearServer: true });
    form.__module4Reset = resetFn;
    registerGlobalResetHandler(() => resetFn({ clearServer: true }));

    updateSelection();
    setStatus(defaultStatus, 'info');
}

function initModule1Flow() {
    const refCanvas = document.getElementById('ref-canvas');
    const testCanvas = document.getElementById('test-canvas');
    if (!refCanvas || !testCanvas) return;

    const ctxRef = refCanvas.getContext('2d');
    const ctxTest = testCanvas.getContext('2d');

    const elements = {
        ref: {
            canvas: refCanvas,
            ctx: ctxRef,
            wrapper: document.querySelector('#module1-ref-card .canvas-wrapper'),
            empty: document.getElementById('ref-canvas-empty'),
            fileInput: document.getElementById('ref-image-input'),
            clearBtn: document.getElementById('ref-clear-btn'),
            resetBtn: document.getElementById('ref-reset-points'),
            pixelLabel: document.getElementById('ref-pixel-width'),
            realWidthInput: document.getElementById('ref-real-width'),
            distanceInput: document.getElementById('ref-distance'),
            calcBtn: document.getElementById('ref-calc-btn'),
            status: document.getElementById('ref-status'),
            card: document.getElementById('module1-ref-card')
        },
        test: {
            canvas: testCanvas,
            ctx: ctxTest,
            wrapper: document.querySelector('#module1-test-card .canvas-wrapper'),
            empty: document.getElementById('test-canvas-empty'),
            fileInput: document.getElementById('test-image-input'),
            clearBtn: document.getElementById('test-clear-btn'),
            resetBtn: document.getElementById('test-reset-points'),
            pixelLabel: document.getElementById('test-pixel-width'),
            distanceInput: document.getElementById('test-distance'),
            calcBtn: document.getElementById('test-calc-btn'),
            status: document.getElementById('test-status'),
            card: document.getElementById('module1-test-card')
        },
        summary: {
            focal: document.getElementById('summary-focal'),
            refPx: document.getElementById('summary-ref-px'),
            refDist: document.getElementById('summary-ref-dist'),
            testDist: document.getElementById('summary-test-dist'),
            testPx: document.getElementById('summary-test-px'),
            realWidth: document.getElementById('summary-real-width'),
            banner: document.getElementById('module1-result-banner'),
            reset: document.getElementById('module1-reset')
        }
    };

    // new control: Use the sample images provided in resources/m1
    elements.ref.useSampleBtn = document.getElementById('module1-use-sample-btn');

    const MAX_HEIGHT = 800;
    const MAX_WIDTH = 900;

    const state = {
        focalLength: null,
        reference: {
            image: null,
            meta: null,
            points: [],
            pixelWidth: null,
            realWidth: null,
            distance: null
        },
        test: {
            image: null,
            meta: null,
            points: [],
            pixelWidth: null,
            distance: null,
            realWidth: null
        }
    };

    const numberFormat = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

    elements.ref.fileInput.addEventListener('change', (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        loadImage('reference', file);
    });

    elements.test.fileInput.addEventListener('change', (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        loadImage('test', file);
    });

    elements.ref.clearBtn.addEventListener('click', () => resetImage('reference'));
    elements.test.clearBtn.addEventListener('click', () => resetImage('test'));

    elements.ref.resetBtn.addEventListener('click', () => resetPoints('reference'));
    elements.test.resetBtn.addEventListener('click', () => resetPoints('test'));

    elements.ref.realWidthInput.addEventListener('input', checkRefReady);
    elements.ref.distanceInput.addEventListener('input', checkRefReady);
    elements.test.distanceInput.addEventListener('input', checkTestReady);

    elements.ref.calcBtn.addEventListener('click', submitFocalLength);
    elements.test.calcBtn.addEventListener('click', submitRealWidth);
    elements.summary.reset.addEventListener('click', resetModule);

    elements.ref.canvas.addEventListener('click', (event) => handleCanvasClick('reference', event));
    elements.test.canvas.addEventListener('click', (event) => handleCanvasClick('test', event));

    function loadImage(kind, file) {
        const reader = new FileReader();
        reader.onload = (evt) => {
            const img = new Image();
            img.onload = () => {
                const target = kind === 'reference' ? elements.ref : elements.test;
                const { displayWidth, displayHeight } = calculateDisplaySize(img);

                const canvas = target.canvas;
                canvas.width = displayWidth;
                canvas.height = displayHeight;

                const ctx = target.ctx;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

                const bucket = state[kind === 'reference' ? 'reference' : 'test'];
                bucket.image = img;
                bucket.meta = {
                    displayWidth,
                    displayHeight,
                    naturalWidth: img.width,
                    naturalHeight: img.height
                };
                bucket.points = [];
                bucket.pixelWidth = null;

                target.wrapper.dataset.state = 'ready';
                target.empty.textContent = 'Click two points to draw the measurement line.';
                target.clearBtn.disabled = false;
                target.resetBtn.disabled = true;

                if (kind === 'reference') {
                    updateStatus('reference', 'Image loaded. Mark the known width.', 'info');
                } else {
                    updateStatus('test', 'Image loaded. Mark the test object.', 'info');
                    elements.test.distanceInput.disabled = false;
                }

                updatePixelMetric(kind);
                checkRefReady();
                checkTestReady();
            };
            img.onerror = () => {
                updateStatus(kind, 'Unable to load that file. Please try a different image.', 'error');
            };
            img.src = evt.target.result;
        };
        reader.readAsDataURL(file);
    }

    function loadImageFromDataUrl(kind, dataUrl, filename) {
        const img = new Image();
        img.onload = () => {
            const target = kind === 'reference' ? elements.ref : elements.test;
            const { displayWidth, displayHeight } = calculateDisplaySize(img);

            const canvas = target.canvas;
            canvas.width = displayWidth;
            canvas.height = displayHeight;

            const ctx = target.ctx;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

            const bucket = state[kind === 'reference' ? 'reference' : 'test'];
            bucket.image = img;
            bucket.meta = {
                displayWidth,
                displayHeight,
                naturalWidth: img.width,
                naturalHeight: img.height
            };
            bucket.points = [];
            bucket.pixelWidth = null;

            target.wrapper.dataset.state = 'ready';
            target.empty.textContent = 'Click two points to draw the measurement line.';
            target.clearBtn.disabled = false;
            target.resetBtn.disabled = true;

            if (kind === 'reference') {
                updateStatus('reference', 'Image loaded. Mark the known width.', 'info');
            } else {
                updateStatus('test', 'Image loaded. Mark the test object.', 'info');
                elements.test.distanceInput.disabled = false;
            }

            updatePixelMetric(kind);
            checkRefReady();
            checkTestReady();
        };
        img.onerror = () => updateStatus(kind, 'Unable to load that file.', 'error');
        img.src = dataUrl;
    }

    function calculateDisplaySize(img) {
        const widthRatio = MAX_WIDTH / img.width;
        const heightRatio = MAX_HEIGHT / img.height;
        const scale = Math.min(1, widthRatio, heightRatio);
        const displayWidth = Math.round(img.width * scale);
        const displayHeight = Math.round(img.height * scale);
        return { displayWidth, displayHeight };
    }

    function handleCanvasClick(kind, event) {
        const bucket = state[kind === 'reference' ? 'reference' : 'test'];
        if (!bucket.image || bucket.points.length === 2) return;

        const elementsGroup = kind === 'reference' ? elements.ref : elements.test;
        const rect = elementsGroup.canvas.getBoundingClientRect();
        const clickPoint = {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };

        const meta = bucket.meta;
        if (!meta || !rect.width || !rect.height) return;

        const scaleX = meta.naturalWidth / rect.width;
        const scaleY = meta.naturalHeight / rect.height;
        const displayScaleX = meta.displayWidth / rect.width;
        const displayScaleY = meta.displayHeight / rect.height;

        const imagePoint = {
            x: clickPoint.x * scaleX,
            y: clickPoint.y * scaleY
        };

        const canvasPoint = {
            x: clickPoint.x * displayScaleX,
            y: clickPoint.y * displayScaleY
        };

        bucket.points.push({ canvas: canvasPoint, image: imagePoint });

        drawCanvas(kind);

        if (bucket.points.length === 2) {
            const [p1, p2] = bucket.points;
            bucket.pixelWidth = Math.hypot(p2.image.x - p1.image.x, p2.image.y - p1.image.y);
            updatePixelMetric(kind);
            elementsGroup.resetBtn.disabled = false;

            if (kind === 'reference') {
                updateStatus('reference', 'Select the real width and distance, then calculate the focal length.', 'success');
            } else {
                updateStatus('test', 'Enter the camera distance to compute the real-world size.', 'success');
            }
        } else {
            updateStatus(kind, 'Point recorded. Select one more point.', 'info');
        }

        checkRefReady();
        checkTestReady();
    }

    function drawCanvas(kind) {
        const target = kind === 'reference' ? elements.ref : elements.test;
        const bucket = state[kind === 'reference' ? 'reference' : 'test'];
        const ctx = target.ctx;
        const canvas = target.canvas;
        if (!bucket.image || !bucket.meta) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(bucket.image, 0, 0, bucket.meta.displayWidth, bucket.meta.displayHeight);

        bucket.points.forEach(({ canvas: point }, index) => {
            ctx.fillStyle = '#ff3b81';
            ctx.strokeStyle = '#1d1f27';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = '#101217';
            ctx.font = '12px Inter, sans-serif';
            ctx.fillText(index === 0 ? 'A' : 'B', point.x + 8, point.y - 8);
        });

        if (bucket.points.length === 2) {
            ctx.strokeStyle = '#31c48d';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(bucket.points[0].canvas.x, bucket.points[0].canvas.y);
            ctx.lineTo(bucket.points[1].canvas.x, bucket.points[1].canvas.y);
            ctx.stroke();
        }
    }

    function resetPoints(kind) {
        const bucket = state[kind === 'reference' ? 'reference' : 'test'];
        const target = kind === 'reference' ? elements.ref : elements.test;
        bucket.points = [];
        bucket.pixelWidth = null;
        drawCanvas(kind);
        target.pixelLabel.textContent = '--';
        target.resetBtn.disabled = true;

        if (kind === 'reference') {
            updateStatus('reference', 'Click two new points to measure the reference width.', 'info');
        } else {
            updateStatus('test', 'Click two new points to measure the test object.', 'info');
        }

        checkRefReady();
        checkTestReady();
    }

    function resetImage(kind) {
        const bucket = state[kind === 'reference' ? 'reference' : 'test'];
        const target = kind === 'reference' ? elements.ref : elements.test;
        bucket.image = null;
        bucket.meta = null;
        resetPoints(kind);
        target.canvas.width = target.canvas.height = 0;
        target.wrapper.dataset.state = 'empty';
        target.empty.textContent = 'Upload an image to begin';
        target.fileInput.value = '';
        target.clearBtn.disabled = true;
        if (kind === 'test') {
            elements.test.distanceInput.value = '';
            elements.test.distanceInput.disabled = true;
        }

        checkRefReady();
        checkTestReady();
    }

    function updatePixelMetric(kind) {
        const target = kind === 'reference' ? elements.ref : elements.test;
        const bucket = state[kind === 'reference' ? 'reference' : 'test'];
        target.pixelLabel.textContent = bucket.pixelWidth ? numberFormat.format(bucket.pixelWidth) + ' px' : '--';
    }

    function checkRefReady() {
        const hasImage = Boolean(state.reference.image);
        const hasPoints = typeof state.reference.pixelWidth === 'number';
        const realWidth = parseFloat(elements.ref.realWidthInput.value);
        const distance = parseFloat(elements.ref.distanceInput.value);
        const numericReady = realWidth > 0 && distance > 0;
        const enable = hasImage && hasPoints && numericReady;
        elements.ref.calcBtn.disabled = !enable;
        return enable;
    }

    function checkTestReady() {
        const unlocked = Boolean(state.focalLength);
        const hasImage = Boolean(state.test.image);
        const hasPoints = typeof state.test.pixelWidth === 'number';
        const distance = parseFloat(elements.test.distanceInput.value);
        const enable = unlocked && hasImage && hasPoints && distance > 0;
        elements.test.calcBtn.disabled = !enable;
        elements.test.resetBtn.disabled = !hasImage || state.test.points.length === 0;
        return enable;
    }

    async function submitFocalLength() {
        if (!checkRefReady()) return;
        toggleLoading(elements.ref.calcBtn, true, 'Calculating...');
        updateStatus('reference', 'Calculating focal length...', 'info');

        const payload = {
            pixelWidth: state.reference.pixelWidth,
            realWidth: parseFloat(elements.ref.realWidthInput.value),
            distance: parseFloat(elements.ref.distanceInput.value)
        };

        try {
            const res = await fetch('/api/a1/focal-length', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Unable to compute focal length.');

            state.focalLength = data.focalLength;
            state.reference.pixelWidth = data.refPixelWidth;
            state.reference.realWidth = data.refRealWidth;
            state.reference.distance = data.refDistance;

            updateStatus('reference', `Focal length stored: ${numberFormat.format(state.focalLength)} px`, 'success');
            unlockTestStep();
            updateSummary();
        } catch (err) {
            updateStatus('reference', err.message, 'error');
        } finally {
            toggleLoading(elements.ref.calcBtn, false);
            checkRefReady();
        }
    }

    async function submitRealWidth() {
        if (!checkTestReady()) return;
        toggleLoading(elements.test.calcBtn, true, 'Calculating...');
        updateStatus('test', 'Calculating real-world width...', 'info');

        const payload = {
            pixelWidth: state.test.pixelWidth,
            distance: parseFloat(elements.test.distanceInput.value),
            focalLength: state.focalLength
        };

        try {
            const res = await fetch('/api/a1/real-width', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Unable to compute real width.');

            state.test.pixelWidth = data.testPixelWidth;
            state.test.distance = data.testDistance;
            state.test.realWidth = data.realWidth;

            updateStatus('test', 'Measurement complete!', 'success');
            updateSummary();
            elements.summary.banner.textContent = `Calculated width: ${numberFormat.format(state.test.realWidth)} cm`;
            elements.summary.banner.classList.add('ready');
        } catch (err) {
            updateStatus('test', err.message, 'error');
        } finally {
            toggleLoading(elements.test.calcBtn, false);
            checkTestReady();
        }
    }

    function unlockTestStep() {
        elements.test.card.classList.remove('locked');
        elements.test.card.removeAttribute('aria-disabled');
        elements.test.fileInput.disabled = false;
        elements.test.clearBtn.disabled = false;
        elements.test.distanceInput.disabled = false;
        updateStatus('test', 'Upload a test image captured by the same camera.', 'info');
    }

    function resetModule() {
        resetImage('reference');
        resetImage('test');
        elements.ref.realWidthInput.value = '';
        elements.ref.distanceInput.value = '';
        elements.test.distanceInput.value = '';
        elements.test.fileInput.disabled = true;
        elements.test.clearBtn.disabled = true;
        elements.test.card.classList.add('locked');
        elements.test.card.setAttribute('aria-disabled', 'true');
        state.reference.realWidth = null;
        state.reference.distance = null;
        state.test.realWidth = null;
        state.test.distance = null;
        state.focalLength = null;
        elements.summary.banner.textContent = 'Run the steps to see the measurement summary.';
        elements.summary.banner.classList.remove('ready');
        updateStatus('reference', 'Upload a reference image to begin calibration.', 'info');
        updateStatus('test', 'Complete Step 1 to unlock measurement.', 'info');
        updateSummary();
        // remove autofill indication
        elements.ref.realWidthInput.classList.remove('autofilled');
        elements.ref.distanceInput.classList.remove('autofilled');
        elements.test.distanceInput.classList.remove('autofilled');
        const expectedEl = document.getElementById('test-expected-value');
        if (expectedEl) expectedEl.textContent = '--';
    }

    function updateSummary() {
        elements.summary.focal.textContent = state.focalLength ? numberFormat.format(state.focalLength) : '--';
        elements.summary.refPx.textContent = state.reference.pixelWidth ? numberFormat.format(state.reference.pixelWidth) : '--';
        elements.summary.refDist.textContent = state.reference.distance ? numberFormat.format(state.reference.distance) : '--';
        elements.summary.testDist.textContent = state.test.distance ? numberFormat.format(state.test.distance) : '--';
        elements.summary.testPx.textContent = state.test.pixelWidth ? numberFormat.format(state.test.pixelWidth) : '--';
        elements.summary.realWidth.textContent = state.test.realWidth ? numberFormat.format(state.test.realWidth) : '--';
    }

    function toggleLoading(button, isLoading, label = 'Processing...') {
        if (isLoading) {
            button.dataset.originalLabel = button.textContent;
            button.textContent = label;
            button.disabled = true;
        } else {
            const original = button.dataset.originalLabel || label;
            button.textContent = original;
            button.disabled = false;
        }
    }

    function updateStatus(kind, message, variant) {
        const target = kind === 'reference' ? elements.ref.status : elements.test.status;
        target.textContent = message;
        target.dataset.variant = variant;
    }

    resetModule();

    // When the 'Use example images' button is clicked, fetch the sample images from server
    if (elements.ref.useSampleBtn) {
        elements.ref.useSampleBtn.addEventListener('click', async () => {
            try {
                elements.ref.useSampleBtn.disabled = true;
                updateStatus('reference', 'Loading example images…', 'info');
                const resp = await fetch('/api/a1/samples');
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.error || 'Failed to load samples');

                if (data.reference?.image) {
                    loadImageFromDataUrl('reference', data.reference.image, data.reference.filename);
                }
                if (data.test?.image) {
                    loadImageFromDataUrl('test', data.test.image, data.test.filename);
                }

                // Autofill fields from the measurements file
                const addAutofilled = (elem, value) => {
                    if (!elem || typeof value === 'undefined' || value === null) return;
                    elem.value = value;
                    elem.classList.add('autofilled');
                    // remove the autofilled class on user input
                    const removeAutofill = () => { elem.classList.remove('autofilled'); elem.removeEventListener('input', removeAutofill); };
                    elem.addEventListener('input', removeAutofill);
                };

                addAutofilled(elements.ref.realWidthInput, Number.isFinite(data.reference.realWidth) ? data.reference.realWidth : '');
                addAutofilled(elements.ref.distanceInput, Number.isFinite(data.reference.distance) ? data.reference.distance : '');
                addAutofilled(elements.test.distanceInput, Number.isFinite(data.test.distance) ? data.test.distance : '');

                // display expected test width if present
                const expectedEl = document.getElementById('test-expected-value');
                if (expectedEl) {
                    expectedEl.textContent = Number.isFinite(data.test.expectedWidth) ? Number(data.test.expectedWidth).toFixed(2) : '--';
                }

                // allow the test step to be unlocked after focal-length calculation; show test controls
                updateStatus('reference', 'Example assets loaded. Mark points and calculate focal length.', 'success');
                elements.ref.useSampleBtn.disabled = false;
            } catch (err) {
                elements.ref.useSampleBtn.disabled = false;
                updateStatus('reference', err.message || 'Failed to load example images', 'error');
                console.error(err);
            }
        });
    }
}
