import { useState, useRef } from 'react'
import './index.css'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const getPartIcon = (partName) => {
    const name = partName.toLowerCase();
    if (name.includes('bonnet') || name.includes('hood')) return '/icons/bonnet.jpg';
    if (name.includes('bumper')) return '/icons/bumper.jpg';
    if (name.includes('headlamp') || name.includes('light')) return '/icons/headlamp.jpg';
    if (name.includes('fender')) return '/icons/fender.jpg';
    return null;
  }

  const handleFile = (selectedFile) => {
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select an image file.')
      return
    }
    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
    setResult(null)
    setError(null)
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || 'Failed to analyze image')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="claimfusion-app">
      <div className="main-wrapper">
        {/* Full-width Header */}
        <header className="topbar">
          <div className="header-top-row">
            <div className="branding">
              <div className="logo-shield">🚗</div>
              <div>
                <h1>ClaimFusion <span></span></h1>
                <p>AI-POWERED VEHICLE DAMAGE & CLAIM ASSESSMENT</p>
              </div>
            </div>

            <div className="header-meta">
              <div className="meta-item">
                <span className="label">CLAIM ID</span>
                <span className="value">CLM-2608-5619</span>
              </div>
              <div className="meta-item">
                <span className="label">ANALYSIS TIME</span>
                <span className="value">{new Date().toLocaleTimeString()} • {new Date().toLocaleDateString()}</span>
              </div>
              <button
                className="btn-primary"
                onClick={() => fileInputRef.current?.click()}
              >
                + NEW INSPECTION
              </button>
            </div>
            <input
              type="file"
              ref={fileInputRef}
              hidden
              accept="image/*"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </div>

          {/* Stepper Workflow */}
          <div className="stepper-container">
            <div className="stepper">
              <div className={`step ${file ? 'completed' : 'active'}`}><div className="circle">1</div><p>UPLOAD</p></div>
              <div className="step-arrow">→</div>
              <div className={`step ${loading ? 'active' : (result ? 'completed' : '')}`}><div className="circle">2</div><p>DETECT</p></div>
              <div className="step-arrow">→</div>
              <div className={`step ${result ? 'completed' : ''}`}><div className="circle">3</div><p>SEVERITY</p></div>
              <div className="step-arrow">→</div>
              <div className={`step ${result ? 'completed' : ''}`}><div className="circle">4</div><p>ESTIMATE</p></div>
              <div className="step-arrow">→</div>
              <div className={`step ${result ? 'active' : ''}`}><div className="circle">5</div><p>CLAIM</p></div>
            </div>
          </div>
        </header>

        {/* Dashboard Grid */}
        <main className="dashboard-content">

          {/* Center Scan Panel */}
          <div className="panel center-scan-panel">
            <div className="panel-header">
              <h2>AI VEHICLE DAMAGE ANALYSIS</h2>
              <span className="subtitle">3D SCAN & DAMAGE DETECTION</span>
            </div>

            <div className="scan-stage">
              {preview ? (
                <>
                  <div className="image-container">
                    <img src={preview} className="vehicle-image" alt="Vehicle" />

                    {/* Restored Damage Callouts */}
                    {result && result.damage_detection.damaged_parts.map((part, i) => (
                      <div key={i} className={`part-callout pos-${i % 4}`}>
                        <div className="callout-line"></div>
                        <div className="callout-box">
                          <h3>{part.part.toUpperCase()}</h3>
                          <p className="confidence">{Math.round(part.confidence * 100)}% YOLO confidence</p>
                          <p className={`severity ${part.severity.toLowerCase()}`}>{part.severity.toUpperCase()}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="glow-platform"></div>
                </>
              ) : (
                <div className="empty-stage">
                  <p>Upload a vehicle image to begin AI analysis.</p>
                </div>
              )}

              {file && !result && !loading && (
                <button className="btn-analyze pulse" onClick={handleAnalyze}>INITIATE AI SCAN</button>
              )}
              {loading && <div className="loading-text">PROCESSING NEURAL NETWORK...</div>}
              {error && <div className="error-text">{error}</div>}
            </div>


          </div>

          {/* Right Panel: Cost Breakdown */}
          <div className="panel right-cost-panel">
            <div className="total-cost-header">
              <p>TOTAL ESTIMATED REPAIR COST ⓘ</p>
              <h2>{result ? `₹${(result.claim_prediction.claim_amount).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '--'}</h2>
              <p className="disclaimer">MODEL-GENERATED ESTIMATE. Actual repair costs may vary based on vehicle model, parts availability, labor rates and service provider.</p>
              <div className="shield-bg">🚗</div>
            </div>

            <div className="parts-list">
              {result && result.damage_detection.damaged_parts.map((part, i) => (
                <div key={i} className="part-row">
                  {getPartIcon(part.part) ? (
                    <img src={getPartIcon(part.part)} className="part-icon-img" alt={part.part} />
                  ) : (
                    <div className="part-img-mock"></div>
                  )}
                  <div className="part-info">
                    <h4>{part.part.toUpperCase()}</h4>
                    <p>Exterior</p>
                  </div>
                  <div className="part-stats">
                    <div className="stat"><span>YOLO CONF.</span><strong>{Math.round(part.confidence * 100)}%</strong></div>
                    <div className="stat"><span>SEVERITY</span><strong className={`sev-${part.severity.toLowerCase()}`}>{part.severity.toUpperCase()}</strong></div>
                  </div>
                </div>
              ))}
              {!result && <p className="no-data">Upload an image to begin analysis.</p>}
            </div>

            <div className="claim-assessment-box">
              <h3>INSURANCE CLAIM ASSESSMENT</h3>
              <div className="claim-grid">
                <div><span>REPAIR COST</span><strong>{result ? `₹${(result.claim_prediction.claim_amount).toLocaleString('en-IN')}` : '--'}</strong></div>
                <div><span>DEDUCTIBLE</span><strong>{result ? '₹5,000.00' : '--'}</strong></div>
                <div><span>ELIGIBLE CLAIM</span><strong className="text-green">{result ? `₹${(Math.max(0, result.claim_prediction.claim_amount - 5000)).toLocaleString('en-IN')}` : '--'}</strong></div>
              </div>
              {result && (
                <div className="claim-status success">
                  <i className="icon-check-circle">✓</i>
                  <div>
                    <strong>CLAIM ASSESSMENT COMPLETE</strong>
                    <p>Estimated claim amount calculated successfully.</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Bottom Row Panels */}
          <div className="bottom-row">

            <div className="panel metrics-strip">
              <div className="metric">
                <div className="metric-icon blue">🚗</div>
                <div className="metric-text">
                  <p>DAMAGES DETECTED</p>
                  <h3>{result ? result.damage_detection.total_damaged_parts : '0'}</h3>
                  <span>Parts Identified</span>
                </div>
              </div>
              <div className="metric">
                <div className="metric-icon cyan">🧠</div>
                <div className="metric-text">
                  <p>AI CONFIDENCE</p>
                  <h3>{result ? '91.4%' : '--'}</h3>
                  <span>Overall Confidence</span>
                </div>
              </div>
              <div className="metric">
                <div className="metric-icon orange">🛡️</div>
                <div className="metric-text">
                  <p>SEVERITY</p>
                  <h3 className="orange-text">{result ? result.damage_prediction.overall_severity.toUpperCase() : 'NONE'}</h3>
                  <span>Overall Assessment</span>
                </div>
              </div>
              <div className="metric">
                <div className="metric-icon green">₹</div>
                <div className="metric-text">
                  <p>REPAIR COST</p>
                  <h3 className="green-text">{result ? `₹${(result.claim_prediction.claim_amount).toLocaleString('en-IN')}` : '--'}</h3>
                  <span>Estimated Total</span>
                </div>
              </div>
              <div className="metric">
                <div className="metric-icon purple">⏱️</div>
                <div className="metric-text">
                  <p>PROCESSING TIME</p>
                  <h3>{result ? '4.8 SEC' : '--'}</h3>
                  <span>AI Analysis Time</span>
                </div>
              </div>
            </div>



          </div>
        </main>
      </div>
    </div>
  )
}

export default App
