/* GENOME project page: theme, reveal, nav highlight, task gallery, lightbox, BibTeX copy. */
(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }

  /* theme */
  var root = document.documentElement, themeBtn = $('#themeBtn');
  function syncTheme() { themeBtn.setAttribute('aria-checked', root.getAttribute('data-theme') === 'dark' ? 'true' : 'false'); }
  syncTheme();
  themeBtn.addEventListener('click', function () {
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('genome-theme', next); } catch (e) {}
    syncTheme();
  });

  /* reveal on scroll */
  if ('IntersectionObserver' in window && !reduce) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -8% 0px' });
    $$('.reveal').forEach(function (n) { io.observe(n); });
  } else {
    $$('.reveal').forEach(function (n) { n.classList.add('in'); });
  }

  /* nav active link */
  var navLinks = $$('.nav-links a[href^="#"]');
  if ('IntersectionObserver' in window) {
    var secIO = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        navLinks.forEach(function (a) { a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id); });
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    navLinks.forEach(function (a) { var s = $(a.getAttribute('href')); if (s) secIO.observe(s); });
  }


  window.__genomeBoost = 0;
  /* ---------- hero: rotating double helix ---------- */
  var cv = $('#helix');
  if (cv && cv.getContext) {
    var ctx = cv.getContext('2d'), W = 0, H = 0, dpr = Math.min(window.devicePixelRatio || 1, 2), t0 = performance.now();
    var hT = 0, hLast = performance.now();
    function helixTime(now) { var dt = Math.min(0.05, (now - hLast) / 1000); hLast = now; hT += dt * (1 + window.__genomeBoost * 7); window.__genomeBoost *= 0.965; return hT; }
    function size() { var r = cv.getBoundingClientRect(); W = r.width; H = r.height; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); }
    function css(v) { return getComputedStyle(root).getPropertyValue(v).trim(); }
    function mix(a, b, f) {
      var pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
      var r = Math.round(((pa >> 16) & 255) * (1 - f) + ((pb >> 16) & 255) * f), g = Math.round(((pa >> 8) & 255) * (1 - f) + ((pb >> 8) & 255) * f), bl = Math.round((pa & 255) * (1 - f) + (pb & 255) * f);
      return 'rgb(' + r + ',' + g + ',' + bl + ')';
    }
    function draw(now) {
      var t = reduce ? 0 : helixTime(now), stops = [css('--g1'), css('--g3'), css('--g5'), css('--g6')];
      ctx.clearRect(0, 0, W, H);
      var n = 64, strands = [[], []], amp = Math.min(H * 0.34, 150), cy = H * 0.46, dark = root.getAttribute('data-theme') === 'dark';
      for (var i = 0; i < n; i++) {
        var x = (i + 0.5) / n * W, ph = i * 0.33 + t * 0.6;
        var y1 = cy + Math.sin(ph) * amp, y2 = cy + Math.sin(ph + Math.PI) * amp, z1 = Math.cos(ph), z2 = -z1;
        var f = i / (n - 1), seg = Math.min(2, Math.floor(f * 3)), col = mix(stops[seg], stops[seg + 1], f * 3 - seg);
        var fade = Math.sin(Math.PI * f), base = (dark ? 0.55 : 0.42) * fade;
        if (i % 2 === 0) { ctx.strokeStyle = col; ctx.globalAlpha = base * 0.35; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y2); ctx.stroke(); }
        strands[0].push([x, y1, col, base]); strands[1].push([x, y2, col, base]);
        [[y1, z1], [y2, z2]].forEach(function (p) {
          ctx.globalAlpha = base * (0.45 + 0.55 * (p[1] + 1) / 2); ctx.fillStyle = col;
          ctx.beginPath(); ctx.arc(x, p[0], 2.2 + 2.2 * (p[1] + 1) / 2, 0, Math.PI * 2); ctx.fill();
        });
      }
      strands.forEach(function (st) {
        for (var k = 1; k < st.length; k++) {
          ctx.strokeStyle = st[k][2]; ctx.globalAlpha = st[k][3] * 0.5; ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.moveTo(st[k - 1][0], st[k - 1][1]); ctx.lineTo(st[k][0], st[k][1]); ctx.stroke();
        }
      });
      ctx.globalAlpha = 1;
      if (!reduce) requestAnimationFrame(draw);
    }
    size(); window.addEventListener('resize', size); requestAnimationFrame(draw);
    themeBtn.addEventListener('click', function () { if (reduce) requestAnimationFrame(draw); });
  }

  /* ---------- hero ticker ---------- */
  var TICK = [
    ['library', 'LOC &middot; VQA &middot; CROP &middot; EVAL'],
    ['initialize', 'can existing modules compare two colors? <b>no</b>'],
    ['generate', 'writing <b>COMPARE_COLOR</b>, checking it on training examples'],
    ['reuse', '<b>COMPARE_COLOR</b> called on new questions and new tasks']
  ], ti = 0, flPh = $('#flPh'), flMsg = $('#flMsg');
  function tickNext() { flPh.textContent = TICK[ti][0]; flMsg.innerHTML = TICK[ti][1]; ti = (ti + 1) % TICK.length; }
  if (flPh) { tickNext(); if (!reduce) setInterval(tickNext, 3200); }


  /* ---------- hero logo: grow modules (idle drift, click burst, hover spin) ---------- */
  var mc = $('#mods'), mx = mc && mc.getContext('2d'), logoEl = $('#heroLogo'), layer = $('#chipLayer'), mparts = [];
  var MODS = ['COMPARE_COLOR', 'COMPARE_MATERIAL', 'COMPARE_ATTRIBUTE', 'LOC', 'VQA', 'CROP', 'EVAL', 'RESULT'];
  var grown = 0;
  function msize() { var r = mc.parentNode.getBoundingClientRect(), d = Math.min(window.devicePixelRatio || 1, 2); mc.width = r.width * d; mc.height = r.height * d; mx.setTransform(d, 0, 0, d, 0, 0); }
  function morigin() { var h = mc.parentNode.getBoundingClientRect(), l = logoEl.getBoundingClientRect(); return { x: l.left - h.left + l.width / 2, y: l.top - h.top + l.height / 2 }; }
  function gcol() { var v = ['--g1', '--g2', '--g3', '--g4', '--g5', '--g6'][Math.floor(Math.random() * 6)]; return getComputedStyle(root).getPropertyValue(v).trim(); }
  function mspawn(n, burst) {
    var o = morigin();
    for (var i = 0; i < n; i++) {
      var a = burst ? Math.random() * Math.PI * 2 : -Math.PI / 2 + (Math.random() - 0.5) * 2.2, sp = burst ? 1.6 + Math.random() * 4.6 : 0.25 + Math.random() * 0.7;
      mparts.push({ x: o.x + (Math.random() - 0.5) * 50, y: o.y + (Math.random() - 0.5) * 50, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, rot: Math.random() * 6, vr: (Math.random() - 0.5) * 0.12,
        s: 3 + Math.random() * (burst ? 6 : 4), life: 1, decay: 0.005 + Math.random() * (burst ? 0.016 : 0.008), c: gcol() });
    }
  }
  function mtick() {
    mx.clearRect(0, 0, mc.width, mc.height);
    if (Math.random() < 0.18) mspawn(1, false);
    for (var i = mparts.length - 1; i >= 0; i--) {
      var p = mparts[i];
      p.x += p.vx; p.y += p.vy; p.vy -= 0.008; p.vx *= 0.99; p.rot += p.vr; p.life -= p.decay;
      if (p.life <= 0) { mparts.splice(i, 1); continue; }
      mx.save(); mx.translate(p.x, p.y); mx.rotate(p.rot); mx.globalAlpha = p.life * 0.85;
      mx.fillStyle = p.c; mx.shadowColor = p.c; mx.shadowBlur = 10;
      var s = p.s; mx.beginPath();
      if (mx.roundRect) mx.roundRect(-s, -s * 0.55, 2 * s, 1.1 * s, s * 0.3); else mx.rect(-s, -s * 0.55, 2 * s, 1.1 * s);
      mx.fill(); mx.restore();
    }
    requestAnimationFrame(mtick);
  }
  function chipOut(name) {
    var o = morigin(), c = document.createElement('span'), a = -Math.PI / 2 + (Math.random() - 0.5) * 2.4, d = 90 + Math.random() * 70;
    c.className = 'fly-chip'; c.textContent = '+ ' + name;
    c.style.left = o.x + 'px'; c.style.top = o.y + 'px';
    c.style.setProperty('--dx', Math.cos(a) * d * 1.6 + 'px'); c.style.setProperty('--dy', Math.sin(a) * d + 'px');
    layer.appendChild(c); setTimeout(function () { c.remove(); }, 1700);
  }
  if (mc && mx && logoEl && !reduce) {
    msize(); window.addEventListener('resize', msize); requestAnimationFrame(mtick);
    logoEl.addEventListener('click', function () {
      mspawn(60, true); window.__genomeBoost = 1;
      logoEl.classList.remove('grow'); void logoEl.offsetWidth; logoEl.classList.add('grow');
      chipOut(MODS[grown % MODS.length]); grown++;
      if (flPh) { flPh.textContent = 'library'; flMsg.innerHTML = 'module <b>' + MODS[(grown - 1) % MODS.length] + '</b> ready for reuse'; }
    });
    setTimeout(function () { mspawn(36, true); }, 700);
  }

  /* ---------- stats count-up ---------- */
  function countUp(n) {
    var to = parseFloat(n.getAttribute('data-count')), dec = +n.getAttribute('data-dec') || 0, t1 = performance.now();
    if (reduce) return;
    (function step(now) { var f = Math.min(1, (now - t1) / 1100), e = 1 - Math.pow(1 - f, 3); n.textContent = (to * e).toFixed(dec); if (f < 1) requestAnimationFrame(step); })(t1);
  }
  if ('IntersectionObserver' in window) {
    var cio = new IntersectionObserver(function (es) { es.forEach(function (e) { if (e.isIntersecting) { countUp(e.target); cio.unobserve(e.target); } }); }, { threshold: 0.6 });
    $$('[data-count]').forEach(function (n) { cio.observe(n); });
  }

  /* ---------- method: the paper's workflow, step by step ---------- */
  var wf = $('#wf');
  if (wf) {
    $$('.wf-node.code pre', wf).forEach(function (pre) {
      pre.innerHTML = pre.innerHTML.split('\n').map(function (l) { return '<span class="ln">' + (l || ' ') + '</span>'; }).join('');
    });
    var STEPS = [
      ['ex1', 0, 1500, 'A few <b>training examples</b> of a new task arrive.'],
      ['d1', 0, 1500, 'The LLM checks whether the existing modules can handle them.'],
      ['no1', 0, 900, 'They cannot, so GENOME initializes a new module.'],
      ['d2', 0, 1300, 'The LLM decides what module to make.'],
      ['sig', 0, 1900, 'It specifies the <b>module signature</b>: the inputs and the output.'],
      ['tc', 0, 2300, 'Each training question is parsed into operation steps, which become the module&rsquo;s <b>test cases</b>.'],
      ['l1', 1, 500, ''],
      ['gen', 1, 1900, 'The LLM <b>writes the module&rsquo;s code</b> from its signature.'],
      ['eval', 1, 1700, 'The new module runs on the test cases built from the training examples.'],
      ['debug', 1, 1700, 'If it fails, the errors go back to the LLM, which <b>debugs and regenerates</b> the code.'],
      ['pass', 1, 1300, 'The module is evaluated again.'],
      ['d3', 1, 1000, 'It passes enough test cases&hellip;'],
      ['yes', 1, 700, ''],
      ['lib', 1, 2100, '&hellip;so it joins the <b>module library</b> for future reuse.'],
      ['l2', 2, 500, ''],
      ['ex2', 2, 1500, 'A <b>test question</b> on a new image.'],
      ['prog', 2, 2100, 'The LLM parses it into a <b>program</b> that reuses <code>Loc</code> and the new <code>Compare_material</code>.'],
      ['run', 2, 1900, 'The program is <b>executed</b> on the image.'],
      ['ans', 2, 2800, 'Answer: <b>Yes</b>. The module learned from one task now answers new questions.']
    ];
    var node = function (k) { return k === 'pass' ? $('[data-k="eval"]', wf) : $('[data-k="' + k + '"]', wf); };
    var tabsW = $$('.wf-tab', wf), cap = $('#wfCap'), idx = -1, timer = null, paused = false, visible = false;
    var stageLen = [0, 0, 0]; STEPS.forEach(function (st) { stageLen[st[1]] += st[2]; });
    function typeIn(n) { $$('.ln', n).forEach(function (l, j) { setTimeout(function () { l.classList.add('in'); }, reduce ? 0 : 140 * j); }); }
    function apply(i, animate) {
      var st = STEPS[i], n = node(st[0]);
      $$('.cur', wf).forEach(function (c) { c.classList.remove('cur'); });
      if (n) {
        n.classList.add('on');
        if (n.classList.contains('wf-node')) n.classList.add('cur');
        if (n.classList.contains('code')) { if (animate) typeIn(n); else $$('.ln', n).forEach(function (l) { l.classList.add('in'); }); }
      }
      if (st[0] === 'pass') node('eval').classList.add('pass');
      if (st[0] === 'run') $('[data-k="ex2"]', wf).classList.add('boxed');
      if (st[3]) cap.innerHTML = st[3];
      var done = 0, s0 = st[1];
      for (var j = 0; j <= i; j++) if (STEPS[j][1] === s0) done += STEPS[j][2];
      tabsW.forEach(function (t, k) { t.classList.toggle('on', k === s0); $('i', t).style.width = (k < s0 ? 100 : k > s0 ? 0 : Math.round(100 * done / stageLen[s0])) + '%'; });
    }
    function reset() {
      $$('.wf-node, .wf-edge, .wf-link', wf).forEach(function (n) { n.classList.remove('on', 'cur', 'pass', 'boxed'); });
      $$('.ln', wf).forEach(function (l) { l.classList.remove('in'); });
    }
    function step() {
      clearTimeout(timer);
      if (paused || !visible) return;
      if (idx >= STEPS.length - 1) { timer = setTimeout(function () { reset(); idx = -1; step(); }, 600); return; }
      idx++; apply(idx, true);
      timer = setTimeout(step, STEPS[idx][2]);
    }
    function jump(stage) {
      clearTimeout(timer); reset();
      var first = STEPS.findIndex(function (st) { return st[1] === stage; });
      for (var j = 0; j < first; j++) apply(j, false);
      idx = first - 1; paused = false; wf.classList.remove('paused'); step();
    }
    tabsW.forEach(function (t) { t.addEventListener('click', function () { jump(+t.getAttribute('data-s')); }); });
    $('#wfPlay').addEventListener('click', function () {
      paused = !paused; wf.classList.toggle('paused', paused);
      this.setAttribute('aria-label', paused ? 'Play' : 'Pause');
      if (!paused) step(); else clearTimeout(timer);
    });
    if (reduce || !('IntersectionObserver' in window)) {
      STEPS.forEach(function (st, j) { apply(j, false); }); $$('.cur', wf).forEach(function (c) { c.classList.remove('cur'); });
      cap.innerHTML = 'GENOME learns <code>Compare_material</code> from a training question, adds it to the library, and reuses it to answer a new one.';
      paused = true; wf.classList.add('paused');
    } else {
      new IntersectionObserver(function (es) { visible = es[0].isIntersecting; if (visible) step(); else clearTimeout(timer); }, { threshold: 0.25 }).observe(wf);
    }
  }
  /* task gallery: static figure by default, the animation on request */
  var TASKS = [
    ['9_gqa', 'Visual Question Answering', 'GQA'],
    ['10_refcoco', 'Referring Expression Comprehension', 'RefCOCO'],
    ['11_imgedit', 'Image Editing', 'Module transfer'],
    ['12_okdet', 'Knowledge-based Object Tagging', 'Module transfer'],
    ['13_raven', "Raven's Progressive Matrices", 'Few-shot'],
    ['14_mewl', 'Few-shot Multimodal Word Learning', 'MEWL']
  ];
  var ci = 0, animate = false, tabs = $('#caseTabs'), img = $('#caseImg'), animBtn = $('#animBtn');
  TASKS.forEach(function (t, k) {
    var b = document.createElement('button'); b.className = 'case-tab task-tab'; b.type = 'button'; b.setAttribute('role', 'tab');
    b.innerHTML = '<span class="cn">(' + String.fromCharCode(97 + k) + ') ' + esc(t[2]) + '</span><span class="ct">' + esc(t[1]) + '</span>';
    b.addEventListener('click', function () { showTask(k); });
    tabs.appendChild(b);
  });
  function src(k) { return 'GENOME_files/' + TASKS[k][0] + (animate ? '.gif' : '.png'); }
  function showTask(k) {
    ci = (k + TASKS.length) % TASKS.length;
    img.src = src(ci);
    img.alt = 'GENOME applied to ' + TASKS[ci][1];
    img.classList.remove('swap'); void img.offsetWidth; img.classList.add('swap');
    $$('.case-tab', tabs).forEach(function (b, j) { b.setAttribute('aria-selected', j === ci ? 'true' : 'false'); });
    $('#caseView').setAttribute('data-zoom', src(ci));
  }
  animBtn.addEventListener('click', function () {
    animate = !animate;
    animBtn.setAttribute('aria-pressed', animate ? 'true' : 'false');
    animBtn.querySelector('span').textContent = animate ? 'Show still figure' : 'Play animation';
    showTask(ci);
  });
  $('#casePrev').addEventListener('click', function () { showTask(ci - 1); });
  $('#caseNext').addEventListener('click', function () { showTask(ci + 1); });
  showTask(0);

  /* lightbox */
  var lb = $('#lightbox');
  document.addEventListener('click', function (e) {
    var z = e.target.closest && e.target.closest('[data-zoom]');
    if (z) { lb.querySelector('img').src = z.getAttribute('data-zoom'); lb.hidden = false; }
  });
  lb.addEventListener('click', function () { lb.hidden = true; });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') lb.hidden = true; });

  /* copy bibtex */
  $('#copyBib').addEventListener('click', function () {
    var btn = this, txt = $('#bibText').textContent;
    var done = function () { btn.classList.add('ok'); btn.querySelector('span').textContent = 'Copied'; setTimeout(function () { btn.classList.remove('ok'); btn.querySelector('span').textContent = 'Copy'; }, 1600); };
    if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done, done); else done();
  });
})();
