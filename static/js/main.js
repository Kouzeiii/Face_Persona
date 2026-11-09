function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.className = `toast ${type}`;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => {
        toast.classList.add('hidden');
    }, 4000);
}

document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => e.preventDefault());

const uploadZone = document.getElementById('uploadZone');
if (uploadZone) {
    uploadZone.addEventListener('dragover', e => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--accent)';
        uploadZone.style.background = 'var(--accent-light)';
    });
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.borderColor = '';
        uploadZone.style.background = '';
    });
    uploadZone.addEventListener('drop', e => {
        e.preventDefault();
        uploadZone.style.borderColor = '';
        uploadZone.style.background = '';
        const file = e.dataTransfer.files[0];
        if (file && /\.(jpg|jpeg|png|bmp|webp)$/i.test(file.name)) {
            const input = document.getElementById('photoInput');
            const dt = new DataTransfer();
            dt.items.add(file);
            input.files = dt.files;
            if (typeof previewPhoto === 'function') previewPhoto(input);
        }
    });
}

// ── Mobile Navigation Toggle & Interactive Features ──
document.addEventListener('DOMContentLoaded', () => {
    // 1. Mobile Navigation Drawer
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.getElementById('navLinks');
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = navLinks.classList.toggle('open');
            navToggle.classList.toggle('active', isOpen);
            navToggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        });

        // Close on navigation link click (smooth SPA or page transitions)
        navLinks.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                if (navLinks.classList.contains('open')) {
                    navLinks.classList.remove('open');
                    navToggle.classList.remove('active');
                    navToggle.setAttribute('aria-expanded', 'false');
                }
            });
        });

        // Close when clicking outside
        document.addEventListener('click', (e) => {
            if (!navToggle.contains(e.target) && !navLinks.contains(e.target) && navLinks.classList.contains('open')) {
                navLinks.classList.remove('open');
                navToggle.classList.remove('active');
                navToggle.setAttribute('aria-expanded', 'false');
            }
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && navLinks.classList.contains('open')) {
                navLinks.classList.remove('open');
                navToggle.classList.remove('active');
                navToggle.setAttribute('aria-expanded', 'false');
                navToggle.focus();
            }
        });
    }

    // 2. Interactive Big Five Constellation
    const constellationCard = document.getElementById('constellationCard');
    if (constellationCard) {
        const nodes = constellationCard.querySelectorAll('.constellation-node');
        const hint = constellationCard.querySelector('.constellation-hint');
        const defaultHint = hint ? hint.innerHTML : 'Hover or tap nodes to explore Big Five dimensions';

        nodes.forEach(node => {
            const trait = node.getAttribute('data-trait');
            const desc = node.getAttribute('data-desc');

            const highlightNode = () => {
                nodes.forEach(n => n.classList.remove('is-active'));
                node.classList.add('is-active');
                if (hint && trait && desc) {
                    hint.innerHTML = `<strong>${trait}:</strong> ${desc}`;
                }
            };

            const resetHighlight = () => {
                node.classList.remove('is-active');
                if (hint) {
                    hint.innerHTML = defaultHint;
                }
            };

            node.addEventListener('mouseenter', highlightNode);
            node.addEventListener('mouseleave', resetHighlight);
            node.addEventListener('focus', highlightNode);
            node.addEventListener('blur', resetHighlight);

            node.addEventListener('click', (e) => {
                e.stopPropagation();
                highlightNode();
            });

            node.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    highlightNode();
                }
            });
        });

        // Clicking outside resets constellation selection
        document.addEventListener('click', (e) => {
            if (!constellationCard.contains(e.target)) {
                nodes.forEach(n => n.classList.remove('is-active'));
                if (hint) hint.innerHTML = defaultHint;
            }
        });
    }

    // 3. Scroll Entry Animations (prefers-reduced-motion respected)
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!prefersReducedMotion && 'IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-visible');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

        document.querySelectorAll('.stat-card, .pipeline-step, .metric-card, .restricted-card').forEach(el => {
            el.classList.add('animate-on-scroll');
            observer.observe(el);
        });
    }

    // 4. Copy-to-Clipboard Utility (for tokens, formulas, or IDs)
    document.querySelectorAll('[data-copy]').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.preventDefault();
            const textToCopy = btn.getAttribute('data-copy');
            if (!textToCopy) return;
            try {
                await navigator.clipboard.writeText(textToCopy);
                showToast('✓ Copied to clipboard', 'success');
            } catch (err) {
                showToast('Failed to copy', 'error');
            }
        });
    });
});
