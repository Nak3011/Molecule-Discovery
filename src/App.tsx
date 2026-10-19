import { useEffect, useRef, useState } from 'react';
import { api } from './types';
import type { Batch, Geometry, LatentPoint, Molecule } from './types';

const examples = [
  { name: 'Caffeine', smiles: 'Cn1c(=O)c2c(ncn2C)n(C)c1=O' },
  { name: 'Aspirin', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
  { name: 'Limonene', smiles: 'CC1=CCC(CC1)C(=C)C' },
  { name: 'Serotonin', smiles: 'NCCc1c[nH]c2ccc(O)cc12' },
];

function Viewer({ geometry, onError }: { geometry: Geometry | null; onError: (s:string)=>void }) {
  const element = useRef<HTMLDivElement>(null);
  const viewer = useRef<any>(null);
  const latestGeometry = useRef(geometry);
  latestGeometry.current = geometry;
  useEffect(() => {
    let live = true;
    import('3dmol').then((lib: any) => {
      if (!live || !element.current) return;
      viewer.current = lib.createViewer(element.current, { backgroundColor: '#f5f5f7', antialias: true });
      if (latestGeometry.current) draw(latestGeometry.current);
    }).catch(() => onError('3D viewer could not be loaded.'));
    const resize = new ResizeObserver(() => viewer.current?.resize());
    if (element.current) resize.observe(element.current);
    return () => { live = false; resize.disconnect(); viewer.current?.clear(); viewer.current = null; };
  }, []);
  const draw = (g: Geometry) => {
    if (!viewer.current) return;
    viewer.current.clear();
    viewer.current.addModel(g.molblock, 'sdf');
    viewer.current.setStyle({}, { stick: { radius: 0.16 }, sphere: { scale: 0.23 } });
    viewer.current.setProjection('orthographic');
    viewer.current.zoomTo();
    viewer.current.render();
  };
  useEffect(() => { if (geometry) draw(geometry); }, [geometry]);
  return <div className="viewer-wrap"><div ref={element} className="mol-viewer" aria-label="Interactive three-dimensional molecular structure" />
    {!geometry && <div className="viewer-empty"><span className="molecule-mark">⌬</span><span>Preparing a 3D conformer</span></div>}
    {geometry && <div className="viewer-caption"><span className="live-dot" />RDKit · ETKDGv3 · {geometry.forceField} optimization</div>}
    <div className="viewer-controls"><button className="icon-button" title="Reset view" onClick={()=>{viewer.current?.zoomTo();viewer.current?.render();}}>↺</button><span>Drag to rotate · Scroll to zoom</span></div>
  </div>;
}

function App() {
  const [status, setStatus] = useState<{mode:string;model:{loaded:boolean;reason:string;latentDimension:number|null}} | null>(null);
  const [batch, setBatch] = useState<Batch | null>(null);
  const [selected, setSelected] = useState<Molecule | null>(null);
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [smiles, setSmiles] = useState(examples[0].smiles);
  const [count, setCount] = useState(6);
  const [temperature, setTemperature] = useState(0.65);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [points, setPoints] = useState<LatentPoint[]>([]);
  const [latentIsIllustrative, setLatentIsIllustrative] = useState(true);
  const [interpolation, setInterpolation] = useState<Array<{alpha:number;smiles:string;valid:boolean}>>([]);
  const [endpoints, setEndpoints] = useState<[LatentPoint|null,LatentPoint|null]>([null,null]);
  const [copyText, setCopyText] = useState('Copy SMILES');
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'generation'|'latent'>('generation');
  const initialized = useRef(false);

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    api.status().then(setStatus).catch(()=>setStatus({mode:'demo',model:{loaded:false,reason:'API unavailable. Start the Python server to enable molecule chemistry.',latentDimension:null}}));
    api.latent().then((r)=>{setPoints(r.points);setLatentIsIllustrative(r.illustrative);}).catch(()=>{});
    generate();
  }, []);
  useEffect(() => { if (batch && !selected) selectMolecule(batch.candidates[0]); }, [batch]);
  async function selectMolecule(m: Molecule | null) {
    setSelected(m); setGeometry(null); setMessage('');
    if (!m?.valid) return;
    try { const g = await api.geometry(m.smiles); setGeometry(g); }
    catch (e) { setMessage(e instanceof Error ? e.message : 'Could not generate a 3D conformer.'); }
  }
  async function inspect(value=smiles) {
    setLoading(true); setMessage('');
    try { const m = await api.inspect(value); setSmiles(m.smiles); setSelected(m); setGeometry(m.geometry || null); }
    catch (e) { setMessage(e instanceof Error ? e.message : 'Could not inspect this molecule.'); }
    finally { setLoading(false); }
  }
  async function generate() {
    setLoading(true); setMessage('');
    try { const result = await api.generate(count,temperature); setBatch(result); setSelected(null); setActiveTab('generation'); }
    catch (e) { setMessage(e instanceof Error ? e.message : 'Generation failed.'); }
    finally { setLoading(false); }
  }
  async function interpolate() {
    if (!endpoints[0] || !endpoints[1]) return;
    setLoading(true);
    try { const r=await api.interpolate(endpoints[0].smiles,endpoints[1].smiles,7);setInterpolation(r.steps); }
    catch(e) { setMessage(e instanceof Error?e.message:'Interpolation failed.'); }
    finally { setLoading(false); }
  }
  async function copySmiles() {
    if (!selected) return;
    await navigator.clipboard.writeText(selected.smiles); setCopyText('Copied'); setTimeout(()=>setCopyText('Copy SMILES'),1300);
  }
  const moleculeCount = selected?.descriptors?.heavyAtoms || 0;
  const candidates = batch?.candidates || [];
  const selectedId = selected?.id || selected?.smiles;

  return <div className="app-shell">
    <header className="global-nav">
      <a className="brand" href="#top" aria-label="Molecule Studio home"><span className="brand-glyph">⌬</span><span>Molecule Studio</span></a>
      <nav className={`global-links ${menuOpen?'open':''}`}><a href="#workspace" onClick={()=>setActiveTab('generation')}>Discover</a><a href="#latent" onClick={()=>setActiveTab('latent')}>Latent space</a><a href="#results">Results</a></nav>
      <div className="nav-right"><span className="nav-model"><span className="live-dot" />{status?.model.loaded?'Model ready':'Demo mode'}</span><button className="menu-toggle" aria-label="Toggle navigation" onClick={()=>setMenuOpen(!menuOpen)}>☰</button></div>
    </header>
    <div className="sub-nav"><div className="sub-nav-inner"><div className="sub-title">Molecule discovery <span className="sub-sep">/</span> <span>Workspace</span></div><div className="sub-actions"><span className="checkpoint">{status?.model.loaded?'VAE checkpoint loaded':'No checkpoint attached'}</span><button className="primary-button small" onClick={generate} disabled={loading}>{loading?'Working…':'Generate molecules'}</button></div></div></div>

    <main id="top">
      <section className="intro" id="workspace"><div className="intro-top"><div><p className="eyebrow">MOLECULAR DESIGN WORKSPACE <span className="mode-pill">{status?.model.loaded?'MODEL INFERENCE':'SIMULATED DEMO'}</span></p><h1>Discover what’s<br className="desktop-break"/> around the corner.</h1><p className="intro-copy">Explore molecular structures and navigate a continuous chemical space.</p></div><div className="intro-note"><span className="note-number">01</span><span>Character-level VAE<br/>for QM9 SMILES</span></div></div>
        <div className="workspace-grid">
          <section className="input-panel utility-card"><div className="panel-heading"><div><span className="section-index">01</span><h2>Start with a molecule</h2></div><span className="label-muted">RDKit validated</span></div>
            <label className="field-label" htmlFor="smiles">SMILES notation</label><div className="smiles-row"><input id="smiles" value={smiles} onChange={e=>setSmiles(e.target.value)} onKeyDown={e=>e.key==='Enter'&&inspect()} placeholder="Enter a SMILES string"/><button className="primary-button" onClick={()=>inspect()} disabled={loading}>{loading?'…':'Inspect'}</button></div>
            <div className="examples"><span>Try an example</span>{examples.map(ex=><button key={ex.name} className="example-chip" onClick={()=>{setSmiles(ex.smiles);inspect(ex.smiles);}}>{ex.name}</button>)}</div>
            <div className="divider" />
            <div className="panel-heading generate-head"><div><span className="section-index">02</span><h2>Sample candidates</h2></div><span className="label-muted">{status?.model.loaded?'Prior sampling':'Curated demo set'}</span></div>
            <div className="control-row"><label>Candidate count<select value={count} onChange={e=>setCount(Number(e.target.value))}>{[4,6,8,10,12].map(n=><option key={n}>{n}</option>)}</select></label><label className="temperature">Temperature <b>{temperature.toFixed(2)}</b><input type="range" min="0.3" max="1.2" step="0.05" value={temperature} onChange={e=>setTemperature(Number(e.target.value))}/></label></div>
            <button className="primary-button generate-button" onClick={generate} disabled={loading}><span>{loading?'Sampling structures…':'Generate candidates'}</span><span aria-hidden>→</span></button>
            <p className="panel-footnote">{status?.model.loaded?'Samples are decoded from the trained VAE prior.':'Demo candidates are curated examples. No model inference is claimed.'}</p>
            {message&&<div className="inline-message" role="alert">{message}</div>}
          </section>

          <section className="viewer-panel" aria-label="Molecule viewer">
            <div className="viewer-topline"><div><span className="section-index">03</span><span className="panel-label">3D STRUCTURE</span></div><span className="structure-tag">{geometry?'REAL 3D COORDINATES':'AWAITING STRUCTURE'}</span></div>
            <Viewer geometry={geometry} onError={setMessage}/>
          </section>
        </div>
      </section>

      {selected?.descriptors&&<section className="molecule-summary" id="results"><div className="summary-name"><span className="eyebrow">SELECTED MOLECULE</span><h2>{selected.name || selected.descriptors.formula}</h2><code>{selected.smiles}</code></div><div className="descriptor-strip"><div><span>Formula</span><strong>{selected.descriptors.formula}</strong></div><div><span>Molecular weight</span><strong>{selected.descriptors.molecularWeight}<small> g/mol</small></strong></div><div><span>Heavy atoms</span><strong>{moleculeCount}</strong></div><button className="secondary-pill" onClick={copySmiles}>{copyText}</button></div></section>}

      <section className="results-section" id="results-gallery"><div className="section-head"><div><p className="eyebrow">THE EXPLORATION</p><h2>Candidate molecules</h2></div><div className="section-head-right"><div className="tab-switch"><button className={activeTab==='generation'?'active':''} onClick={()=>setActiveTab('generation')}>Candidates</button><button className={activeTab==='latent'?'active':''} onClick={()=>setActiveTab('latent')}>Latent map</button></div>{batch&&<div className="export-actions"><a href={api.exportUrl('csv',candidates.map(c=>c.smiles),candidates.map(c=>c.source||batch?.source||'VAE'))} download="molecule-results.csv">CSV ↓</a><a href={api.exportUrl('sdf',candidates.map(c=>c.smiles),candidates.map(c=>c.source||batch?.source||'VAE'))} download="molecule-results.sdf">SDF ↓</a></div>}</div></div>
        {activeTab==='generation'?<><div className="results-overview"><span>{batch?`${candidates.length} molecules${batch.relatedCount?` · ${batch.relatedCount} related references`:''}`:'Generate a set of molecules to begin'}</span><span>{batch?.source==='vae'?(batch.relatedCount?'VAE + references':'Original VAE'):'Illustrative demo data'}</span></div><div className="candidate-grid">{candidates.map((candidate,i)=><button className={`candidate-card ${selectedId===candidate.id?'selected':''}`} key={candidate.id||candidate.smiles} onClick={()=>selectMolecule(candidate)}><span className="candidate-no">{candidate.relatedReference?'RELATED REFERENCE':`CANDIDATE ${String(i+1).padStart(2,'0')}`}</span><span className="candidate-name">{candidate.name||candidate.descriptors?.formula||'Molecule'}</span><code>{candidate.smiles}</code>{candidate.relatedReference&&<span className="reference-note">Closest supported structure to the VAE output.</span>}<span className="candidate-meta">{candidate.valid&&candidate.descriptors?`${candidate.descriptors.formula} · ${candidate.descriptors.molecularWeight} g/mol`:'Structure unavailable'}<span className={candidate.valid?'valid-dot':'invalid-dot'} /></span></button>)}{!candidates.length&&<div className="empty-candidates"><span className="molecule-mark">⌬</span><span>Generate a set of molecules to begin.</span></div>}</div>{batch&&<div className="metrics-line"><span><b>{batch.metrics.total}</b> molecules shown</span><span><b>{batch.relatedCount||0}</b> related references</span><span><b>{batch.metrics.unique}</b> unique structures</span><span><b>{batch.metrics.averageMolecularWeight}</b> avg. molecular weight</span><span><b>{batch.metrics.averageLogP}</b> avg. LogP</span><span className="demo-caption">{batch.source==='vae'?'Descriptor summary includes clearly labeled references.':'Illustrative demo results.'}</span></div>}</>:<div className="latent-layout" id="latent"><div className="latent-chart"><LatentMap points={points} endpoints={endpoints} setEndpoints={setEndpoints} illustrative={latentIsIllustrative} onSelect={async p=>{setSmiles(p.smiles);await inspect(p.smiles);}}/><div className="latent-axis-label">{latentIsIllustrative?'ILLUSTRATIVE 2D PROJECTION':'TRAINED ENCODER · PCA PROJECTION'} · SELECT TWO POINTS TO INTERPOLATE</div></div><div className="latent-details"><span className="section-index">04</span><h3>Walk the latent space.</h3><p>Choose two molecules in the map, then inspect the decoded steps between their representations.</p><div className="endpoint-row"><span>{endpoints[0]?.name||'Choose start'}</span><span>→</span><span>{endpoints[1]?.name||'Choose end'}</span></div><button className="primary-button" disabled={!endpoints[0]||!endpoints[1]||loading} onClick={interpolate}>Interpolate <span>→</span></button></div><div className="interpolation-steps">{interpolation.map((step,i)=><button key={i} onClick={()=>step.valid&&inspect(step.smiles)} className={step.valid?'':'invalid-step'}><span>STEP {i+1}</span><code>{step.valid?step.smiles:'No structure decoded'}</code></button>)}</div><p className="latent-disclaimer">{latentIsIllustrative?'Illustrative positions are hand-placed and do not come from a trained encoder. In demo mode, intermediate structures are not model-decoded.':'Positions come from the trained encoder projected to two dimensions. Interpolation steps are decoded from the trained model.'}</p></div>}
      </section>

      <section className="dark-band"><div className="dark-inner"><div><p className="eyebrow">A CONTINUOUS REPRESENTATION</p><h2>From sequence<br/>to structure.</h2></div><div className="dark-copy"><p>The notebook’s GRU variational autoencoder learns from character-level SMILES. Molecules are parsed by RDKit, then given optimized 3D coordinates for interactive exploration.</p><a href="https://github.com/DrKenReid/VAE-for-Molecule-Discovery" target="_blank" rel="noreferrer">View the research notebook ↗</a></div><div className="dark-steps"><div><span>01</span> Encode SMILES</div><div><span>02</span> Sample latent vectors</div><div><span>03</span> Decode candidates</div></div></div></section>
      <footer className="footer"><div className="footer-inner"><div className="footer-brand"><span className="brand-glyph">⌬</span><span>Molecule Studio</span></div><div className="footer-state"><span>MODEL STATUS</span><b>{status?.model.loaded?'Notebook checkpoint connected':'Running in demo mode'}</b><small>{status?.model.reason||'Checking runtime…'}</small></div><div className="footer-copy">VAE for Molecule Discovery<br/>QM9 · RDKit · 3Dmol.js</div></div><div className="legal">A molecular exploration demo; a structure alone does not imply biological activity or therapeutic suitability.</div></footer>
    </main>
  </div>;
}

function LatentMap({points,endpoints,setEndpoints,illustrative,onSelect}:{points:LatentPoint[];endpoints:[LatentPoint|null,LatentPoint|null];setEndpoints:(p:[LatentPoint|null,LatentPoint|null])=>void;illustrative:boolean;onSelect:(p:LatentPoint)=>void}) {
  const choose=(p:LatentPoint)=>{setEndpoints(endpoints[0]===null?[p,null]:endpoints[1]===null?[endpoints[0],p]:[p,null]);};
  return <svg className="latent-svg" viewBox="0 0 640 360" role="img" aria-label="Illustrative molecular latent-space projection">
    <line x1="38" y1="180" x2="606" y2="180" className="axis"/><line x1="320" y1="20" x2="320" y2="338" className="axis"/><text x="594" y="168" className="axis-text">z₁</text><text x="330" y="30" className="axis-text">z₂</text>
    {points.map((p,i)=>{const x=320+p.x*258,y=180-p.y*140;const selected=endpoints.some(e=>e?.name===p.name);return <g key={p.name} className="latent-point" onClick={()=>{choose(p);onSelect(p);}} tabIndex={0} role="button" aria-label={`Select ${p.name}`}><circle cx={x} cy={y} r={selected?8:5.5} className={selected?'point-selected':'point'} /><text x={x+9} y={y-8} className="point-label">{p.name}</text></g>;})}
    {endpoints[0]&&endpoints[1]&&<line x1={320+endpoints[0].x*258} y1={180-endpoints[0].y*140} x2={320+endpoints[1].x*258} y2={180-endpoints[1].y*140} className="interpolation-line"/>}
  </svg>;
}

export default App;
