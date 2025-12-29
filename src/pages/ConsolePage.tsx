/**
 * Llama 3.3 + Azure Speech Console using Human-like Voice Architecture
 */
import { useEffect, useState, useRef } from 'react';
import { useLlamaVoice } from '../hooks/useLlamaVoice';
import { WaveOrb } from '../components/WaveOrb';
import { Button } from '../components/button/Button';

import { Mic, MicOff } from 'react-feather';
import appLogo from '../assets/logo.png';
import './ConsolePage.scss';

export function ConsolePage() {
  const { messages, state, volume, isConnected, isMuted, initialize, toggleMute } = useLlamaVoice();

  // Configuration State - Use localStorage first, then VITE_* env vars as fallback
  const [azureKey, setAzureKey] = useState(
    localStorage.getItem('azure_speech_key') || import.meta.env.VITE_AZURE_SPEECH_KEY || ''
  );
  const [azureRegion, setAzureRegion] = useState(
    localStorage.getItem('azure_speech_region') || import.meta.env.VITE_AZURE_SPEECH_REGION || ''
  );
  const [hfToken, setHfToken] = useState(
    localStorage.getItem('hf_token') || import.meta.env.VITE_HUGGING_FACE_TOKEN || ''
  );
  const [showConfig, setShowConfig] = useState(false);

  const eventsScrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat
  useEffect(() => {
    if (eventsScrollRef.current) {
      eventsScrollRef.current.scrollTop = eventsScrollRef.current.scrollHeight;
    }
  }, [messages]);

  // AUTO-START SEAMLESSLY
  useEffect(() => {
    let mounted = true;

    const initSession = async () => {
      // If we are already connected, or explicitly muted (user action), don't force it?
      // Actually user said "indefinite", so we assume we want it Active.
      // We check our hooked `isConnected` state.
      // Note: `isConnected` might lag slightly, better to rely on `initialize` being safe.

      const storedKey = localStorage.getItem('azure_speech_key');
      const storedRegion = localStorage.getItem('azure_speech_region');

      // Always call initialize - it has fallback logic for VITE_* env vars
      // Pass stored credentials if available, otherwise let initialize use env vars
      await initialize(storedKey || undefined, storedRegion || undefined);
    };

    // Only start after a user gesture to satisfy AudioContext requirements
    const handleGesture = () => {
      initSession();
      // Once triggered, we can remove these specific listeners
      // (UseLlamaVoice handles subsequent re-inits via its internal logic)
      window.removeEventListener('click', handleGesture);
      window.removeEventListener('keydown', handleGesture);
    };

    window.addEventListener('click', handleGesture);
    window.addEventListener('keydown', handleGesture);

    return () => {
      mounted = false;
      window.removeEventListener('click', handleGesture);
      window.removeEventListener('keydown', handleGesture);
    };
  }, [initialize]);

  const handleMuteToggle = () => {
    toggleMute();
  };

  return (
    <div data-component="ConsolePage" className="llama-console">
      <div className="content-top">
        <div className="content-title">
          <span className="title-text">MEDDOLLINA</span>
          <span className="status-badge" data-status={state}>
            {isMuted ? 'MUTED' : state.toUpperCase()}
          </span>
        </div>
        <div className="content-controls">
          <img src={appLogo} alt="MEDDOLLINA" className="header-logo" style={{ height: '50px', objectFit: 'contain' }} />
        </div>

        <div className="content-main">
          {/* CENTER VISUALIZATION */}
          <div className="orb-container">
            <WaveOrb state={isMuted ? 'idle' : state} amplitude={isMuted ? 0 : volume} />
          </div>

          {/* CHAT LOG (Transcription) */}
          <div className="chat-container">
            <div className="chat-log" ref={eventsScrollRef}>
              {messages.length === 0 && <div className="empty-state">Start speaking to begin conversation...</div>}
              {messages.map((msg, i) => (
                <div key={i} className={`chat-message ${msg.role}`}>
                  <div className="message-header">
                    {msg.role === 'user' ? 'You' : 'Llama'}
                    {msg.mood && <span className="mood-tag">{msg.mood}</span>}
                  </div>
                  <div className="message-content">{msg.content}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* CONFIG MODAL / OVERLAY */}
        {showConfig && (
          <div className="config-overlay">
            <div className="config-card">
              <h3>Setup</h3>
              <div className="input-group">
                <label>Azure Speech Key</label>
                <input type="password" value={azureKey} onChange={e => setAzureKey(e.target.value)} placeholder="Key 1 from Azure Portal" />
              </div>
              <div className="input-group">
                <label>Azure Region</label>
                <input type="text" value={azureRegion} onChange={e => setAzureRegion(e.target.value)} placeholder="e.g. eastus" />
              </div>
              <div className="input-group">
                <label>Hugging Face Token</label>
                <input type="password" value={hfToken} onChange={e => setHfToken(e.target.value)} placeholder="hf_..." />
              </div>
              <div className="config-actions">
                <Button label="Save & Close" onClick={() => {
                  // Save to localStorage for persistence
                  localStorage.setItem('azure_speech_key', azureKey);
                  localStorage.setItem('azure_speech_region', azureRegion);
                  localStorage.setItem('hf_token', hfToken);
                  // Re-init with new keys (HF token is passed separately via localStorage)
                  initialize(azureKey, azureRegion, hfToken);
                  setShowConfig(false);
                }} />
              </div>
            </div>
          </div>
        )}

        <div className="content-actions">
          {/* MUTE TOGGLE BUTTON */}
          <Button
            label={isMuted ? 'Unmute' : 'Mute'}
            iconPosition="start"
            icon={isMuted ? MicOff : Mic}
            buttonStyle={isMuted ? 'alert' : 'action'} // Alert style for Muted? Or Action for Active?
            // Let's say: Action (Blue) = Active/Normal. Alert (Red) = Muted. 
            // Or usually: Red = Recording... 
            // Let's stick to: "action" (Blue) when Muted (to encourage unmute), "regular" or "alert" when Active?
            // Actually user asked for Mute button.
            // State: Active -> Button says "Mute" -> Style: Regular
            // State: Muted -> Button says "Unmute" -> Style: Action (highlighted)
            onClick={handleMuteToggle}
            className={isMuted ? 'muted-btn' : 'active-btn'}
          />
        </div>
      </div>
    </div>
  );
}
