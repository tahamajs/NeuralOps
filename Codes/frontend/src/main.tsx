import React from 'react'
import ReactDOM from 'react-dom/client'
import { AppProviders } from './providers/AppProviders'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppProviders />
  </React.StrictMode>
)