import React, { useState, useEffect, useRef } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const JOB_OPTIONS = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
  'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
const MARITAL_OPTIONS = ['married', 'divorced', 'single']
const EDUCATION_OPTIONS = ['primary', 'secondary', 'tertiary', 'unknown']
const YES_NO = ['no', 'yes']
const CONTACT_OPTIONS = ['cellular', 'telephone', 'unknown']
const MONTH_OPTIONS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
const POUTCOME_OPTIONS = ['unknown', 'failure', 'other', 'success']

const initialForm = {
  age: 40, job: 'management', marital: 'married', education: 'secondary',
  default: 'no', balance: 1500, housing: 'yes', loan: 'no',
  contact: 'cellular', day: 15, month: 'may', campaign: 2,
  never_contacted: true, pdays: 30, previous: 0, poutcome: 'unknown',
}

const LIKELY_EXAMPLE = {
  age: 58, job: 'retired', marital: 'single', education: 'tertiary',
  default: 'no', balance: 8500, housing: 'no', loan: 'no',
  contact: 'cellular', day: 20, month: 'oct', campaign: 1,
  never_contacted: false, pdays: 10, previous: 3, poutcome: 'success',
}

const UNLIKELY_EXAMPLE = {
  age: 28, job: 'blue-collar', marital: 'divorced', education: 'primary',
  default: 'yes', balance: -300, housing: 'yes', loan: 'yes',
  contact: 'unknown', day: 5, month: 'nov', campaign: 10,
  never_contacted: true, pdays: 30, previous: 0, poutcome: 'failure',
}

// SVG ring circumference for r=50
const RADIUS = 50
const CIRCUMFERENCE = 2 * Math.PI * RADIUS  // ≈ 314.16

function ConfidenceRing({ probability, label }) {
  const pct = probability / 100
  const offset = CIRCUMFERENCE - pct * CIRCUMFERENCE

  return (
    <div className="ring-container">
      <svg className="ring-svg" viewBox="0 0 120 120">
        <circle className="ring-track" cx="60" cy="60" r={RADIUS} />
        <circle
          className={`ring-fill ${label}`}
          cx="60" cy="60" r={RADIUS}
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="ring-center">
        <span className={`ring-value ${label}`}>{probability}%</span>
        <span className="ring-label">confidence</span>
      </div>
    </div>
  )
}

function Field({ label, children, className = '' }) {
  return (
    <div className={`field ${className}`}>
      {label && <label>{label}</label>}
      {children}
    </div>
  )
}

function Sel({ value, onChange, options }) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)}>
      {options.map(o => <option key={o} value={o}>{o}</option>)}
    </select>
  )
}

export default function App() {
  const [form, setForm] = useState(initialForm)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [animPct, setAnimPct] = useState(0)

  function update(key, value) { setForm(f => ({ ...f, [key]: value })) }

  function loadExample(ex) {
    setForm(ex); setResult(null); setError(null); setAnimPct(0)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true); setError(null); setResult(null); setAnimPct(0)

    const payload = {
      ...form,
      age: Number(form.age), balance: Number(form.balance),
      day: Number(form.day), campaign: Number(form.campaign),
      pdays: Number(form.pdays), previous: Number(form.previous),
    }

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const d = await res.json().catch(() => ({}))
        throw new Error(d.detail || `Server error ${res.status}`)
      }
      const data = await res.json()
      setResult(data)
      setTimeout(() => setAnimPct(data.probability_percent), 80)
    } catch (err) {
      setError(err.message || 'Could not reach the prediction service.')
    } finally {
      setLoading(false)
    }
  }

  const monthNames = {
    jan:'January', feb:'February', mar:'March', apr:'April',
    may:'May', jun:'June', jul:'July', aug:'August',
    sep:'September', oct:'October', nov:'November', dec:'December'
  }

  return (
    <div className="app-shell">

      {/* ── Top Navbar ── */}
      <nav className="navbar">
        <div className="nav-brand">
          <div className="nav-logo">DI</div>
          <span className="nav-name">DepositIQ</span>
          <span className="nav-badge">ML Model</span>
        </div>
        <div className="nav-divider" />
        <span className="nav-title">Customer Subscription Predictor</span>
        <div className="nav-spacer" />
        <div className="nav-status">
          <span className="status-dot" />
          Model active
        </div>
      </nav>

      {/* ── Main Workspace ── */}
      <div className="workspace">

        {/* ── Left: Form Panel ── */}
        <div className="form-panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <span className="panel-title">Customer Assessment</span>
              <span className="panel-subtitle">Enter profile data to generate a subscription prediction</span>
            </div>
            <div className="panel-actions">
              <button className="btn-ghost green" onClick={() => loadExample(LIKELY_EXAMPLE)}>
                ✓ Likely example
              </button>
              <button className="btn-ghost red" onClick={() => loadExample(UNLIKELY_EXAMPLE)}>
                ✗ Unlikely example
              </button>
            </div>
          </div>

          {error && (
            <div className="error-bar">
              <span>⚠</span> {error}
            </div>
          )}

          <form className="form-body" onSubmit={handleSubmit} id="predict-form">

            {/* Customer Profile */}
            <div className="form-section">
              <div className="section-header">
                <div className="section-icon">👤</div>
                <span className="section-name">Customer Profile</span>
              </div>
              <div className="field-grid cols-4">
                <Field label="Age">
                  <input type="number" min="18" max="100" value={form.age}
                    onChange={e => update('age', e.target.value)} required />
                </Field>
                <Field label="Job">
                  <Sel value={form.job} onChange={v => update('job', v)} options={JOB_OPTIONS} />
                </Field>
                <Field label="Marital Status">
                  <Sel value={form.marital} onChange={v => update('marital', v)} options={MARITAL_OPTIONS} />
                </Field>
                <Field label="Education">
                  <Sel value={form.education} onChange={v => update('education', v)} options={EDUCATION_OPTIONS} />
                </Field>
              </div>
            </div>

            {/* Financial Status */}
            <div className="form-section">
              <div className="section-header">
                <div className="section-icon">💳</div>
                <span className="section-name">Financial Status</span>
              </div>
              <div className="field-grid cols-4">
                <Field label="Avg. Balance (€)" className="span-2">
                  <input type="number" value={form.balance}
                    onChange={e => update('balance', e.target.value)} required />
                </Field>
                <Field label="Credit Default">
                  <Sel value={form.default} onChange={v => update('default', v)} options={YES_NO} />
                </Field>
                <Field label="Housing Loan">
                  <Sel value={form.housing} onChange={v => update('housing', v)} options={YES_NO} />
                </Field>
                <Field label="Personal Loan" className="span-2">
                  <Sel value={form.loan} onChange={v => update('loan', v)} options={YES_NO} />
                </Field>
              </div>
            </div>

            {/* Campaign Details */}
            <div className="form-section">
              <div className="section-header">
                <div className="section-icon">📞</div>
                <span className="section-name">Campaign Details</span>
              </div>
              <div className="field-grid cols-4">
                <Field label="Contact Type">
                  <Sel value={form.contact} onChange={v => update('contact', v)} options={CONTACT_OPTIONS} />
                </Field>
                <Field label="Last Contact Day">
                  <input type="number" min="1" max="31" value={form.day}
                    onChange={e => update('day', e.target.value)} required />
                </Field>
                <Field label="Last Contact Month">
                  <Sel value={form.month} onChange={v => update('month', v)} options={MONTH_OPTIONS} />
                </Field>
                <Field label="Contacts (Campaign)">
                  <input type="number" min="1" value={form.campaign}
                    onChange={e => update('campaign', e.target.value)} required />
                </Field>
              </div>
            </div>

            {/* Previous Campaign */}
            <div className="form-section">
              <div className="section-header">
                <div className="section-icon">📋</div>
                <span className="section-name">Previous Campaign</span>
              </div>
              <div className="field-grid cols-4">
                <Field className="span-full">
                  <label className="checkbox-field">
                    <input type="checkbox" checked={form.never_contacted}
                      onChange={e => update('never_contacted', e.target.checked)} />
                    <span>No prior contact — first time reaching this customer</span>
                  </label>
                </Field>
                {!form.never_contacted && (
                  <Field label="Days Since Last Contact">
                    <input type="number" min="1" value={form.pdays}
                      onChange={e => update('pdays', e.target.value)} />
                  </Field>
                )}
                <Field label="Previous Contacts">
                  <input type="number" min="0" value={form.previous}
                    onChange={e => update('previous', e.target.value)} />
                </Field>
                <Field label="Previous Outcome" className={form.never_contacted ? 'span-2' : ''}>
                  <Sel value={form.poutcome} onChange={v => update('poutcome', v)} options={POUTCOME_OPTIONS} />
                </Field>
              </div>
            </div>

          </form>

          <div className="form-footer">
            <button className="submit-btn" type="submit" form="predict-form" disabled={loading}>
              {loading ? 'Running analysis…' : 'Run Prediction →'}
            </button>
            <span className="submit-note">
              Random Forest · 16 features<br />No call duration (anti-leakage)
            </span>
          </div>
        </div>

        {/* ── Right: Result Panel ── */}
        <div className="result-panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <span className="panel-title">Prediction Output</span>
              <span className="panel-subtitle">UCI Bank Marketing · Random Forest</span>
            </div>
            {result && (
              <button className="btn-ghost" onClick={() => { setResult(null); setAnimPct(0) }}>
                Clear
              </button>
            )}
          </div>

          {loading && (
            <div className="loading-state">
              <div className="spinner" />
              <span className="loading-label">Running model inference…</span>
            </div>
          )}

          {!loading && !result && (
            <div className="result-empty">
              <div className="empty-icon">📊</div>
              <span className="empty-title">No prediction yet</span>
              <span className="empty-sub">
                Fill in the customer profile on the left and click "Run Prediction" to see results here.
              </span>
            </div>
          )}

          {!loading && result && (
            <div className="result-content">

              {/* Confidence Ring */}
              <div className="confidence-section">
                <ConfidenceRing
                  probability={animPct}
                  label={result.label}
                />
                <span className="confidence-note">Subscription probability score</span>
              </div>

              {/* Verdict */}
              <div className="verdict-section">
                <div className="verdict-label-row">Verdict</div>
                <div key={result.label}
                  className={`verdict-badge ${result.label === 'yes' ? 'approved' : 'declined'}`}>
                  <span className={`verdict-dot ${result.label}`} />
                  {result.prediction}
                </div>
              </div>

              {/* Stats */}
              <div className="stats-section">
                <div className="stat-row">
                  <span className="stat-key">Confidence score</span>
                  <span className="stat-val">{result.probability_percent}%</span>
                </div>
                <div className="stat-row">
                  <span className="stat-key">Decision threshold</span>
                  <span className="stat-val">50.0%</span>
                </div>
                <div className="stat-row">
                  <span className="stat-key">Assessed month</span>
                  <span className="stat-val">{monthNames[form.month] || form.month}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-key">Prior campaign result</span>
                  <span className="stat-val" style={{ textTransform: 'capitalize' }}>{form.poutcome}</span>
                </div>
              </div>

              {/* Probability bar */}
              <div className="prob-section">
                <div className="prob-header">
                  <span className="prob-label">Probability distribution</span>
                  <span className={`prob-pct ${result.label}`}>{result.probability_percent}%</span>
                </div>
                <div className="prob-track">
                  <div className={`prob-fill ${result.label}`}
                    style={{ width: `${animPct}%` }} />
                </div>
              </div>

            </div>
          )}
        </div>
      </div>

      {/* ── Status Bar ── */}
      <div className="statusbar">
        <div className="statusbar-item">
          <span className="statusbar-dot" />
          API connected
        </div>
        <div className="statusbar-item">Model: Random Forest</div>
        <div className="statusbar-item">Dataset: UCI Bank Marketing (45,211 records)</div>
        <div style={{ flex: 1 }} />
        <div className="statusbar-item">Call duration excluded · Anti-leakage</div>
      </div>

    </div>
  )
}
