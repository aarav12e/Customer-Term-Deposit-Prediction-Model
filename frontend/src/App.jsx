import React, { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const JOB_OPTIONS = ['admin.', 'unemployed', 'management', 'housemaid', 'entrepreneur',
  'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services', 'unknown']
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

function Field({ label, children, full }) {
  return (
    <div className={`field${full ? ' full' : ''}`}>
      <label>{label}</label>
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
  const [probWidth, setProbWidth] = useState(0)

  function update(key, value) { setForm(f => ({ ...f, [key]: value })) }

  function loadExample(ex) {
    setForm(ex)
    setResult(null)
    setError(null)
    setProbWidth(0)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true); setError(null); setResult(null); setProbWidth(0)

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
        throw new Error(d.detail || `Error ${res.status}`)
      }
      const data = await res.json()
      setResult(data)
      setTimeout(() => setProbWidth(data.probability_percent), 60)
    } catch (err) {
      setError(err.message || 'Could not reach the prediction service.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="masthead">
        <span className="eyebrow">Bank Marketing · ML Model</span>
        <h1>Term Deposit Subscription Predictor</h1>
        <p>
          Enter a customer profile to estimate their likelihood of subscribing
          to a term deposit. Built on a Random Forest trained without call duration
          to avoid data leakage.
        </p>
      </header>

      <div className="ledger-grid">
        {/* ── Form ── */}
        <form className="form-card" onSubmit={handleSubmit}>

          {/* Example loaders */}
          <div className="example-btns">
            <button type="button" className="example-btn likely"
              onClick={() => loadExample(LIKELY_EXAMPLE)}>
              ✓ Load likely example
            </button>
            <button type="button" className="example-btn unlikely"
              onClick={() => loadExample(UNLIKELY_EXAMPLE)}>
              ✗ Load unlikely example
            </button>
          </div>

          <div className="section-label">Customer Profile</div>
          <div className="field-grid">
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

          <div className="section-label">Financial Status</div>
          <div className="field-grid">
            <Field label="Avg. Yearly Balance (€)">
              <input type="number" value={form.balance}
                onChange={e => update('balance', e.target.value)} required />
            </Field>
            <Field label="Credit in Default?">
              <Sel value={form.default} onChange={v => update('default', v)} options={YES_NO} />
            </Field>
            <Field label="Has Housing Loan?">
              <Sel value={form.housing} onChange={v => update('housing', v)} options={YES_NO} />
            </Field>
            <Field label="Has Personal Loan?">
              <Sel value={form.loan} onChange={v => update('loan', v)} options={YES_NO} />
            </Field>
          </div>

          <div className="section-label">Campaign Details</div>
          <div className="field-grid">
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
            <Field label="Contacts This Campaign">
              <input type="number" min="1" value={form.campaign}
                onChange={e => update('campaign', e.target.value)} required />
            </Field>
          </div>

          <div className="section-label">Previous Campaign</div>
          <div className="field-grid">
            <Field full label="">
              <label className="checkbox-row">
                <input type="checkbox" checked={form.never_contacted}
                  onChange={e => update('never_contacted', e.target.checked)} />
                Never contacted before this campaign
              </label>
            </Field>
            {!form.never_contacted && (
              <Field label="Days Since Last Contact">
                <input type="number" min="1" value={form.pdays}
                  onChange={e => update('pdays', e.target.value)} />
              </Field>
            )}
            <Field label="No. of Previous Contacts">
              <input type="number" min="0" value={form.previous}
                onChange={e => update('previous', e.target.value)} />
            </Field>
            <Field label="Previous Campaign Outcome">
              <Sel value={form.poutcome} onChange={v => update('poutcome', v)} options={POUTCOME_OPTIONS} />
            </Field>
          </div>

          <button className="submit-btn" type="submit" disabled={loading}>
            {loading ? 'Predicting…' : 'Predict likelihood →'}
          </button>

          {error && <div className="error-text">⚠ {error}</div>}
        </form>

        {/* ── Result ── */}
        <aside className="result-card">
          <div className="section-label">Result</div>

          {loading && (
            <div className="loading-text">
              <div className="spinner" />
              Running model…
            </div>
          )}

          {!loading && !result && (
            <div className="result-empty">
              Fill in the form and click<br />"Predict likelihood" to see<br />the result here.
            </div>
          )}

          {!loading && result && (
            <>
              <div className="result-row">
                <span className="k">Prediction</span>
                <span className="v">{result.prediction}</span>
              </div>

              <div className="prob-bar-wrap">
                <div className="prob-bar-header">
                  <span className="prob-bar-label">Probability</span>
                  <span className={`prob-bar-value ${result.label}`}>
                    {result.probability_percent}%
                  </span>
                </div>
                <div className="prob-track">
                  <div className={`prob-fill ${result.label}`}
                    style={{ width: `${probWidth}%` }} />
                </div>
              </div>

              <div className="stamp-zone">
                <div key={result.label}
                  className={`verdict-badge ${result.label === 'yes' ? 'approved' : 'declined'}`}>
                  <span className="verdict-icon">
                    {result.label === 'yes' ? '✓' : '✗'}
                  </span>
                  {result.label === 'yes' ? 'Likely to subscribe' : 'Unlikely to subscribe'}
                </div>
              </div>
            </>
          )}
        </aside>
      </div>

      <footer className="note">
        UCI Bank Marketing dataset · Random Forest · Call duration excluded to prevent data leakage.
      </footer>
    </div>
  )
}
