 
    // --- CONFIG: set your backend endpoint here ---
    const API_URL = "http://127.0.0.1:8000/predict"; // change to your FastAPI /predict
    const GMAIL_API = "http://127.0.0.1:8000/fetch-gmail";
    const OUTLOOK_API = "http://127.0.0.1:8000/fetch-outlook";

    let state = {emails:[],open:null,filter:'all',sort:'desc',query:'',source:''};

    // Load Gmail emails
    async function loadGmail(){
      try{
        const res = await fetch(GMAIL_API);
        const data = await res.json();
        state.emails = data.emails.map((e,i) => ({
          id:i+1,
          from:e.sender,
          name:e.sender.split('@')[0],
          subject:e.subject,
          preview:e.body.slice(0,60) + '...',
          body:e.body,
          date:new Date().toISOString(),
          url:e.url||''
        }));
        state.source = 'Gmail';
        state.open = null;
        renderList();
      }catch(err){
        console.error('Gmail load error:',err);
        alert('Failed to load Gmail. Check API endpoint.');
      }
    }

    // Load Outlook emails
    async function loadOutlook(){
      try{
        const res = await fetch(OUTLOOK_API);
        const data = await res.json();
        state.emails = data.emails.map((e,i) => ({
          id:i+1,
          from:e.sender,
          name:e.sender.split('@')[0],
          subject:e.subject,
          preview:e.body.slice(0,60) + '...',
          body:e.body,
          date:new Date().toISOString(),
          url:e.url||''
        }));
        state.source = 'Outlook';
        state.open = null;
        renderList();
      }catch(err){
        console.error('Outlook load error:',err);
        alert('Failed to load Outlook. Check API endpoint.');
      }
    }

    // simple helper - format date
    function fmt(d){
        try{const dt=new Date(d);return dt.toLocaleString()

        }
        catch(e){
            return d;
        }}

    function renderList(){
      const list = document.getElementById('list'); list.innerHTML='';
      let arr = state.emails.slice();
      if(state.query) arr = arr.filter(e=> (e.subject+e.from+e.preview).toLowerCase().includes(state.query.toLowerCase()));
      if(state.filter==='safe') arr = arr.filter(e=> e.meta && e.meta.label==='safe');
      if(state.filter==='phish') arr = arr.filter(e=> e.meta && e.meta.label==='phish');
      arr.sort((a,b)=> state.sort==='desc' ? (new Date(b.date)-new Date(a.date)) : (new Date(a.date)-new Date(b.date)) );

      // Update email count display
      document.getElementById('emailCount').textContent = state.emails.length;
      document.getElementById('source').textContent = state.source || 'Gmail';

      arr.forEach(e=>{
        const it = document.createElement('div'); it.className='item'; it.setAttribute('data-id',e.id);
        it.innerHTML = `
          <div class="avatar">${escapeHtml(e.name.split(' ')[0][0]||'U')}</div>
          <div class="meta">
            <div class="row">
              <div style="min-width:0" class="subject">${escapeHtml(e.subject)}</div>
              <div style="margin-left:12px;font-size:12px;color:var(--muted)">${fmt(e.date)}</div>
            </div>
            <div class="row" style="margin-top:6px">
              <div class="preview">${escapeHtml(e.preview)}</div>
              <div style="margin-left:8px">${renderBadge(e.meta)}</div>
            </div>
          </div>
        `;
        it.onclick = ()=> openEmail(e.id);
        list.appendChild(it);
      })
    }

    function renderBadge(meta){
      if(!meta) return '';
      if(meta.label==='phish') return `<span class="badge danger">Phish</span>`;
      if(meta.label==='safe') return `<span class="badge safe">Safe</span>`;
      return '';
    }

    function openEmail(id){
      const e = state.emails.find(x=>x.id==id); if(!e) return;
      state.open = e.id;
      document.getElementById('openSubject').textContent = e.subject;
      document.getElementById('openFrom').textContent = `${e.name} — ${e.from} · ${fmt(e.date)}`;
      document.getElementById('body').innerHTML = `<div style="font-size:14px;color:var(--muted);margin-bottom:12px">Preview: ${escapeHtml(e.preview)}</div><div style="white-space:pre-wrap">${escapeHtml(e.body)}</div>`;
      renderPredictionCard(e);
    }

    function renderPredictionCard(e){
      const box = document.getElementById('verdictBox');
      const label = document.getElementById('verdictLabel');
      const probVal = document.getElementById('probVal');
      const probText = document.getElementById('probText');
      if(!e.meta){ box.style.display='none'; return; }
      box.style.display='block';
      const p = Math.round((e.meta.prob||0)*100);
      probVal.textContent = p + '%';
      probText.textContent = e.meta.label==='phish' ? 'High phishing probability' : 'Looks safe';
      if(e.meta.label==='phish'){ box.className='verdictCard dangerCard' } else { box.className='verdictCard safeCard' }
          // Keep the background color constant
    box.style.background = 'linear-gradient(90deg,#56e9c8,#6aa7ff)';
    box.style.border = 'none'; // Optional: Remove any border styles if applied
    }

    // quick escape
    function escapeHtml(s){ return String(s||'').replace(/[&<>"']/g, function(m){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]}); }

    // simulate calling backend
    async function scanEmail(e){
      // if API_URL points to localhost and isn't reachable, fallback to heuristic
      try{
        const resp = await fetch(API_URL, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({subject:e.subject,from:e.from,body:e.body})});
        if(!resp.ok) throw new Error('bad');
        const data = await resp.json();
        // expected: {prob:0.83, label:'phish',explain:'...'}
        e.meta = {prob:data.prob||0,label:data.label|| (data.prob>0.5? 'phish':'safe'),explain:data.explain||''};
        return e.meta;
      }catch(err){
        // fallback heuristic (for demo only)
        const score = heuristicScore(e);
        e.meta = {prob:score,label: score > 0.6 ? 'phish' : 'safe',explain:'Local heuristic used'};
        return e.meta;
      }
    }

    function heuristicScore(e){
      let s=0; const txt=(e.subject+' '+e.preview+' '+e.body+' '+e.from).toLowerCase();
      if(/click|verify|account|password|urgent|claim|winner|congratulations|prize|reset/.test(txt)) s+=0.45;
      if(/@bank|@secure|@account|@paypal|@amazon/.test(e.from)) s+=0.25;
      if(txt.split('http').length>1) s+=0.2;
      if(/\d{4,}/.test(txt)) s+=0.05;
      return Math.min(1, s);
    }

    // actions
    document.getElementById('search').addEventListener('input', e=>{ state.query=e.target.value; renderList(); });
    document.getElementById('sort').addEventListener('change', e=>{ state.sort=e.target.value; renderList(); });
    document.getElementById('filter').addEventListener('change', e=>{ state.filter=e.target.value; renderList(); });

    document.getElementById('scanBtn').addEventListener('click', async ()=>{
      if(!state.open) return alert('Select an email first');
      const e = state.emails.find(x=>x.id==state.open);
      const el = document.getElementById('scanBtn'); el.disabled=true; el.textContent='Scanning...';
      await scanEmail(e); renderList(); openEmail(e.id);
      el.disabled=false; el.textContent='Run Phishing Detection';
    });

    document.getElementById('runAllBtn').addEventListener('click', async ()=>{
      const btn = document.getElementById('runAllBtn'); 
      btn.disabled=true; btn.textContent='Scanning all...';
      for(const e of state.emails){ await scanEmail(e); }
      renderList();
      if(state.open) openEmail(state.open);
      btn.disabled=false;
      btn.textContent='Scan All'; 
    });

    document.getElementById('reload').addEventListener('click', ()=>{ 
      if(state.source==='Outlook') loadOutlook();
      else loadGmail();
    });

    // theme toggle (dark <-> light)
    document.getElementById('dm').addEventListener('click', ()=>{
      document.body.classList.toggle('dark');
      document.querySelectorAll('.knob').forEach(k=>k.classList.toggle('dark'));
      // swap semantic CSS variables so the UI responds
      if(document.body.classList.contains('dark')){
        // dark theme
        document.documentElement.style.setProperty('--bg-gradient-1','#071022');
        document.documentElement.style.setProperty('--bg-gradient-2','#071822');
        document.documentElement.style.setProperty('--surface','rgba(255,255,255,0.02)');
        document.documentElement.style.setProperty('--text','#e6eef6');
        document.documentElement.style.setProperty('--muted','#98a0b8');
        document.documentElement.style.setProperty('--panel','#0f1724');
        document.documentElement.style.setProperty('--logo-text','#071022');
        document.documentElement.style.setProperty('--btn-text','#042028');
      } else {
        // light theme
        document.documentElement.style.setProperty('--bg-gradient-1','#f4f7fb');
        document.documentElement.style.setProperty('--bg-gradient-2','#eef2f6');
        document.documentElement.style.setProperty('--surface','#ffffff');
        document.documentElement.style.setProperty('--text','#071022');
        document.documentElement.style.setProperty('--muted','#6b7280');
        document.documentElement.style.setProperty('--panel','#ffffff');
        document.documentElement.style.setProperty('--logo-text','#071022');
        document.documentElement.style.setProperty('--btn-text','#042028');
      }
    });

    // initialize dark theme on page load
    document.body.classList.add('dark');
    document.querySelectorAll('.knob').forEach(k=>k.classList.add('dark'));
    document.documentElement.style.setProperty('--bg-gradient-1','#071022');
    document.documentElement.style.setProperty('--bg-gradient-2','#071822');
    document.documentElement.style.setProperty('--surface','rgba(255,255,255,0.02)');
    document.documentElement.style.setProperty('--text','#e6eef6');
    document.documentElement.style.setProperty('--muted','#98a0b8');
    document.documentElement.style.setProperty('--panel','#0f1724');
    document.documentElement.style.setProperty('--logo-text','#071022');
    document.documentElement.style.setProperty('--btn-text','#042028');

    // initial load - try Gmail first
    async function initLoad(){
      await loadGmail();
      setTimeout(()=>{ if(state.emails.length) openEmail(state.emails[0].id); },300);
    }

    initLoad();

    // expose for debugging
    window._PRO_INBOX = {state,scanEmail,renderList,loadGmail,loadOutlook};
 