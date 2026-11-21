import { useEffect, useRef, useState } from 'react'
import './App.css'

const categoryColors = {
  cardboard: '#a6d86e',
  glass: '#59d5a5',
  metal: '#9aa8a0',
  paper: '#c5f38b',
  plastic: '#48bd78',
  trash: '#f0b970',
}

function Icon({ name, size = 18 }) {
  const paths = {
    overview: <><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /></>,
    scan: <><path d="M7 3H5a2 2 0 0 0-2 2v2m14-4h2a2 2 0 0 1 2 2v2M3 17v2a2 2 0 0 0 2 2h2m14-4v2a2 2 0 0 1-2 2h-2" /><path d="M7 12h10M12 7v10" /></>,
    chart: <><path d="M4 19V5m0 14h17" /><path d="m7 15 4-4 3 2 6-7" /></>,
    leaf: <><path d="M20 4c-9 0-15 4-15 11a5 5 0 0 0 5 5c7 0 11-6 10-16Z" /><path d="M5 19c3-4 6-7 11-10" /></>,
    upload: <><path d="M12 16V4m-4 4 4-4 4 4" /><path d="M20 16.5v2A1.5 1.5 0 0 1 18.5 20h-13A1.5 1.5 0 0 1 4 18.5v-2" /></>,
    arrow: <><path d="M5 12h14" /><path d="m13 6 6 6-6 6" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    sparkle: <><path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Z" /><path d="m19 14 1 2.5 2 1-2 1-1 2.5-1-2.5-2-1 2-1 1-2.5Z" /></>,
    recycle: <><path d="m7 7 3-4 3 4" /><path d="M10 3v5h5a3 3 0 0 1 2.6 1.5l1.2 2" /><path d="m18 17 4-1-1 4" /><path d="m21 16-4 2.5-2.5-4.3" /><path d="m6 14-1 5 5 1" /><path d="m5 19 2-4.5-2.5-4.3A3 3 0 0 1 7 6.5h2.5" /></>,
    close: <><path d="m6 6 12 12M18 6 6 18" /></>,
  }

  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name] || paths.sparkle}
    </svg>
  )
}

function MetricCard({ label, value, detail, icon }) {
  return (
    <article className="metric-card">
      <div className="metric-card-top">
        <span className="metric-icon"><Icon name={icon} size={17} /></span>
        <span className="metric-label">{label}</span>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div>
    </article>
  )
}

function ScoreBar({ label, value, color, detail }) {
  return (
    <div className="score-row">
      <div className="score-row-heading">
        <span>{label}</span>
        <span className="score-number">{detail || `${value.toFixed(1)}%`}</span>
      </div>
      <div className="score-track">
        <div className="score-fill" style={{ width: `${Math.min(value, 100)}%`, background: color }} />
      </div>
    </div>
  )
}

function App() {
  const fileInput = useRef(null)
  const [metrics, setMetrics] = useState(null)
  const [metricsError, setMetricsError] = useState('')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [predictionError, setPredictionError] = useState('')
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    let active = true
    fetch('/api/metrics')
      .then(async (response) => {
        if (!response.ok) throw new Error((await response.json()).error || 'Could not load model metrics.')
        return response.json()
      })
      .then((data) => {
        if (active) setMetrics(data)
      })
      .catch((error) => {
        if (active) setMetricsError(error.message)
      })
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (!file) {
      setPreview('')
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  async function analyzeImage(imageFile) {
    if (!imageFile) return
    setFile(imageFile)
    setPrediction(null)
    setPredictionError('')
    setPredicting(true)
    const formData = new FormData()
    formData.append('image', imageFile)
    try {
      const response = await fetch('/api/predict', { method: 'POST', body: formData })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Image analysis failed.')
      setPrediction(data)
    } catch (error) {
      setPredictionError(error.message)
    } finally {
      setPredicting(false)
    }
  }

  function chooseFile(event) {
    const selected = event.target.files?.[0]
    if (selected) analyzeImage(selected)
    event.target.value = ''
  }

  const bestModel = metrics?.models?.[0]
  const validationCount = metrics?.validation_images?.toLocaleString()

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="Bingo home">
          <span className="brand-mark"><Icon name="recycle" size={23} /></span>
          <span>BINGO<span className="brand-period">.</span></span>
        </a>

        <div className="sidebar-caption">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          <a className="nav-link active" href="#overview"><Icon name="overview" /> Overview</a>
          <a className="nav-link" href="#analyze"><Icon name="scan" /> Image analysis</a>
          <a className="nav-link" href="#models"><Icon name="chart" /> Model insights</a>
        </nav>

        <div className="sidebar-bottom">
          <div className="model-status">
            <span className="status-dot" />
            <span><b>Model online</b><small>ResNet-18 · 6 classes</small></span>
          </div>
          <div className="sidebar-foot">SMARTER SORTING<br />STARTS HERE.</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><span className="breadcrumb-slash">/</span><b>Overview</b></div>
          <div className="topbar-right">
            <span className="live-badge"><span className="status-dot" /> LIVE MODEL</span>
            <span className="avatar">B</span>
          </div>
        </header>

        <div className="content-wrap">
          <section className="welcome-row" id="overview">
            <div>
              <div className="eyebrow"><Icon name="sparkle" size={14} /> WASTE INTELLIGENCE PLATFORM</div>
              <h1>Make every item<br /><span>count.</span></h1>
              <p className="welcome-copy">Identify waste in seconds. Make smarter choices for a cleaner tomorrow.</p>
            </div>
            <div className="hero-art" aria-hidden="true">
              <div className="hero-orbit orbit-one" />
              <div className="hero-orbit orbit-two" />
              <div className="hero-glow" />
              <div className="hero-leaf"><Icon name="leaf" size={80} /></div>
              <span className="hero-spark spark-one">✳</span>
              <span className="hero-spark spark-two">✦</span>
              <span className="hero-chip">LESS WASTE <b>↗</b></span>
            </div>
          </section>

          {metricsError && (
            <div className="error-banner" role="alert">
              <span>{metricsError}</span>
              <small>Start the Python API and run <code>python src/evaluate_models.py</code> to create the evaluation data.</small>
            </div>
          )}

          <section className="metrics-grid" aria-label="Model overview">
            <MetricCard label="Validation accuracy" value={bestModel ? `${(bestModel.accuracy * 100).toFixed(1)}%` : '—'} detail="Best performing model" icon="check" />
            <MetricCard label="Macro F1 score" value={bestModel ? `${(bestModel.macro_f1 * 100).toFixed(1)}%` : '—'} detail="Balanced across categories" icon="chart" />
            <MetricCard label="Images evaluated" value={validationCount || '—'} detail="Held-out validation split" icon="scan" />
            <MetricCard label="Waste categories" value={metrics?.classes?.length || '—'} detail="Ready to recognize" icon="recycle" />
          </section>

          <section className="analysis-layout" id="analyze">
            <article className="panel upload-panel">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">01 / INSTANT CLASSIFICATION</div>
                  <h2>What are we looking at?</h2>
                </div>
                <span className="panel-index">01</span>
              </div>
              <p className="muted-copy">Drop in a photo and let Bingo find the right bin.</p>

              <input
                ref={fileInput}
                className="visually-hidden"
                type="file"
                accept="image/png,image/jpeg,image/webp,image/bmp"
                onChange={chooseFile}
              />
              {!file ? (
                <button
                  type="button"
                  className={`drop-zone ${dragging ? 'is-dragging' : ''}`}
                  onClick={() => fileInput.current?.click()}
                  onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(event) => {
                    event.preventDefault()
                    setDragging(false)
                    const dropped = event.dataTransfer.files?.[0]
                    if (dropped) analyzeImage(dropped)
                  }}
                >
                  <span className="upload-icon"><Icon name="upload" size={22} /></span>
                  <strong>Drop your image here</strong>
                  <span className="drop-subtext">or <u>browse files</u> to upload</span>
                  <span className="file-hint">JPG, PNG, WEBP · MAX 12 MB</span>
                </button>
              ) : (
                <div className="image-preview-card">
                  <img src={preview || undefined} alt="Selected waste item" />
                  <div className="preview-overlay">
                    <span className="file-name">{file.name}</span>
                    <button className="icon-button" type="button" onClick={() => { setFile(null); setPrediction(null); setPredictionError('') }} aria-label="Remove image">
                      <Icon name="close" size={16} />
                    </button>
                  </div>
                </div>
              )}

              {predicting && <div className="loading-state"><span className="loader" /> Reading your image...</div>}
              {predictionError && <div className="inline-error" role="alert">{predictionError}</div>}
              {prediction && (
                <div className="prediction-result">
                  <div className="result-title-row">
                    <div>
                      <div className="eyebrow">BEST MATCH</div>
                      <h3>{prediction.predicted_class}</h3>
                    </div>
                    <div className="confidence-ring" style={{ '--score': `${prediction.confidence * 100}%` }}>
                      <span>{(prediction.confidence * 100).toFixed(0)}<small>%</small></span>
                    </div>
                  </div>
                  <div className="confidence-title">CLASS CONFIDENCE</div>
                  <div className="confidence-list">
                    {prediction.scores.map((score) => (
                      <ScoreBar
                        key={score.class_name}
                        label={score.class_name}
                        value={score.probability * 100}
                        color={categoryColors[score.class_name]}
                      />
                    ))}
                  </div>
                  <div className="result-footnote">Prediction confidence is an estimate, not a guarantee.</div>
                </div>
              )}
            </article>

            <article className="panel model-panel" id="models">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">02 / PERFORMANCE LAB</div>
                  <h2>Models, measured.</h2>
                </div>
                <span className="panel-index">02</span>
              </div>
              <p className="muted-copy">Same validation images. One clear front-runner.</p>

              <div className="chart-legend">
                <span><i className="legend-dot accuracy-dot" /> ACCURACY</span>
                <span><i className="legend-dot f1-dot" /> MACRO F1</span>
              </div>
              <div className="model-chart">
                {(metrics?.models || []).map((model, index) => (
                  <div className={`model-chart-row ${index === 0 ? 'winner' : ''}`} key={model.model}>
                    <div className="model-name-row">
                      <span className="model-name">{model.model}</span>
                      {index === 0 && <span className="winner-tag">BEST</span>}
                    </div>
                    <ScoreBar label="Accuracy" value={model.accuracy * 100} color={index === 0 ? '#8ce85e' : '#467954'} />
                    <ScoreBar label="Macro F1" value={model.macro_f1 * 100} color={index === 0 ? '#35c879' : '#275b42'} />
                  </div>
                ))}
                {!metrics && !metricsError && <div className="empty-chart">Loading model results...</div>}
              </div>
              <div className="model-note">
                <span className="note-icon"><Icon name="sparkle" size={15} /></span>
                <span><b>{bestModel?.model || 'Best model'}</b> takes the lead with a <b>{bestModel ? `${(bestModel.macro_f1 * 100).toFixed(1)}%` : '—'} macro F1</b>, balanced across all six waste types.</span>
              </div>
            </article>
          </section>

          <section className="bottom-grid">
            <article className="panel category-panel">
              <div className="section-heading compact-heading">
                <div>
                  <div className="eyebrow">CATEGORY BREAKDOWN</div>
                  <h2>Where it shines.</h2>
                </div>
                <span className="small-caption">F1 · VALIDATION</span>
              </div>
              <div className="category-chart">
                {(metrics?.class_metrics || []).map((item) => (
                  <div className="category-column" key={item.class_name}>
                    <span className="category-value">{(item.f1_score * 100).toFixed(0)}</span>
                    <div className="category-bar-track">
                      <div className="category-bar" style={{ height: `${item.f1_score * 100}%`, background: categoryColors[item.class_name] }} />
                    </div>
                    <span className="category-name">{item.class_name}</span>
                  </div>
                ))}
                {!metrics && !metricsError && <div className="empty-chart">Loading category scores...</div>}
              </div>
              <div className="category-note">Trash is the most challenging class; clearer item photos can help improve results.</div>
            </article>

            <article className="panel categories-panel">
              <div className="section-heading compact-heading">
                <div>
                  <div className="eyebrow">BUILT FOR EVERYDAY SORTING</div>
                  <h2>Six ways to recycle.</h2>
                </div>
              </div>
              <div className="category-pills">
                {(metrics?.classes || ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']).map((category, index) => (
                  <div className="category-pill" key={category}>
                    <span className="category-index">0{index + 1}</span>
                    <span className="category-color" style={{ background: categoryColors[category] }} />
                    <span>{category}</span>
                    <Icon name="arrow" size={14} />
                  </div>
                ))}
              </div>
              <div className="bottom-tip"><Icon name="leaf" size={17} /><span>Small choices. Big impact.</span></div>
            </article>
          </section>

          <footer className="page-footer">
            <span>♻ BINGO <span className="footer-dot">·</span> WASTE INTELLIGENCE</span>
            <span>Validation metrics only · Confirm final performance on the separate test set</span>
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
