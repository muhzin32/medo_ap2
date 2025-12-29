import { useState, useRef, useCallback } from 'react';
import * as SpeechSDK from 'microsoft-cognitiveservices-speech-sdk';
import { HfInference } from '@huggingface/inference';
import { TTSQueue } from '../services/TTSQueue';
import fillersConfig from '../data/fillers.json';

// Configuration
const SPEECH_KEY = import.meta.env.VITE_AZURE_SPEECH_KEY || '';
const SPEECH_REGION = import.meta.env.VITE_AZURE_SPEECH_REGION || '';
const HF_TOKEN = import.meta.env.VITE_HUGGING_FACE_TOKEN || '';

/**
 * TwinMind AI Architecture Implementation
 * 
 * This hook implements the 5-Layer "Brain" of the TwinMind system:
 * 
 * 1️⃣ Audio Capture (Frontend)
 *    - Captures raw audio via `getUserMedia` and `SpeechSDK.AudioConfig`.
 * 
 * 2️⃣ Audio Preprocessing
 *    - VAD (Voice Activity Detection) via Azure SDK.
 *    - `initAudioAnalyzer` for client-side signal analysis (volume, speed, energy).
 * 
 * 3️⃣ Acoustic Model (Core STT Brain)
 *    - Azure Speech SDK with "Whisper-like" accuracy settings.
 *    - Handles accents and code-switching (hi-IN/en-IN).
 * 
 * 4️⃣ Language Model + Decoder (LLM)
 *    - `processResponse` handles the "Thinking Brain".
 *    - `decideTurnLanguage` enforces strict language locking per turn.
 *    - Generates emotion-tagged responses.
 * 
 * 5️⃣ Text-to-Speech (TTS) & Full Duplex Loop
 *    - `TTSQueue` manages serial playback.
 *    - `speak` applies prosody (pitch/rate) based on emotion.
 *    - "Barge-in" support allows interruptions, creating the "Full Duplex" feel.
 */

let hf: HfInference | null = null;

// State Machine Types
type ConversationState = 'IDLE' | 'LISTENING' | 'THINKING' | 'SPEAKING' | 'INTERRUPTED';


// 1️⃣ SINGLE FIXED VOICE (GLOBAL)
const FIXED_TTS_VOICE = "en-IN-NeerjaNeural"; // Changed to English Voice

// 4️⃣ NAME EXTRACTION: REMOVED (LLM handles this semantically now)
// const NAME_REGEX = /(my name is|मेरा नाम|माई नेम इज़)\s+([^\s।]+)/iu;

// 🔊 CONVERSATION TRIGGERS (Strict Regex with Anchors & Boundaries)
// Wake: Start of speech only
const WAKE_TRS = /^(hey|hello|hi|listen|are you there|ok|then|yes)\b/i;
// Interrupt: Start of speech or standalone command
const INTERRUPT_TRS = /^(wait|hold on|stop|no no|listen|one second)\b/i;
// Exit: Specific phrases
const EXIT_TRS = /^(bye|goodbye|that's all)\b|stop listening/i;

// 2️⃣ TURN-BASED LANGUAGE RESOLUTION - REMOVED (English Only)

// 🔑 Extract fact BEFORE embedding
// REMOVED: extractNameFact relying on Regex.
// We now rely on the LLM to extract entities naturally if needed, 
// or a dedicated extraction step that doesn't use hardcoded regex.

// 2️⃣ SOTA LID DECISION ENGINE - REMOVED (English Only)

interface EmotionState {
    valence: number;  // -1 (sad) → +1 (happy)
    arousal: number;  // 0 (calm) → 1 (excited)
}

interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string;
    mood?: string;
    audioContext?: UserAudioContext;
}

interface UserAudioContext {
    volume: number;      // 0-1 normalized volume
    pitch: number;       // Hz, estimated fundamental frequency
    speed: number;       // Words per second
    emotion: string;     // Detected emotion: 'excited', 'calm', 'neutral', etc.
    energy: 'high' | 'medium' | 'low';
}

// Barge-in configuration
// ⚠️ ENHANCED for better interruption response
const INTERRUPT_THRESHOLD = 0.15; // RMS threshold (lowered from 0.25 for better sensitivity)
const INTERRUPT_DURATION_MS = 180; // Minimum sustained speech (lowered from 250ms for faster response)
const BARGE_IN_COOLDOWN_MS = 500; // Prevent rapid re-triggering

export const useLlamaVoice = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [state, setState] = useState<'idle' | 'listening' | 'processing' | 'speaking'>('idle');
    const [volume, setVolume] = useState(0);
    const [isConnected, setIsConnected] = useState(false);
    const [hasQueuedBargeIn, setHasQueuedBargeIn] = useState(false); // 🆕 Visual feedback state

    const recognizerRef = useRef<SpeechSDK.SpeechRecognizer | null>(null);
    const synthesizerRef = useRef<SpeechSDK.SpeechSynthesizer | null>(null);

    // State Machine - Replaces multiple boolean flags
    const conversationStateRef = useRef<ConversationState>('IDLE');

    // Emotion State Machine - Continuous emotion tracking
    const emotionStateRef = useRef<EmotionState>({
        valence: 0,  // Start neutral
        arousal: 0.3 // Slightly engaged
    });
    const emotionConsistencyRef = useRef<string[]>([]); // New: Consensus buffer

    // Track if the session SHOULD be active (user pressed Start)
    const isActiveRef = useRef(false);

    // Prevent multiple simultaneous initialization attempts
    const isInitializingRef = useRef(false);

    // Store credentials for re-initialization
    const credentialsRef = useRef<{ key: string; region: string } | null>(null);





    // 💾 PERSISTENCE STATE
    const sessionIdRef = useRef<string | null>(null);
    const hasLoadedHistoryRef = useRef(false);


    const MIN_UTTERANCE_DURATION_MS = 400; // 
    // LLM timing for filler injection
    const llmRequestTimeRef = useRef<number>(0);
    const hasFirstChunkRef = useRef<boolean>(false);
    const lastFillerTimeRef = useRef<number>(0); // ⏱️ Filler cooldown tracker

    // Silence detection for client-side end-of-utterance
    const lastSpeechTimeRef = useRef<number>(0);
    const silenceDurationRef = useRef<number>(0);

    // ⚡ BARGE-IN & CANCELLATION STATE
    const currentTurnIdRef = useRef<number>(0); // Generation ID for cancellation
    const lastTTSStartTimeRef = useRef<number>(0); // Debounce for echo cancellation
    const activeListeningTimeoutRef = useRef<NodeJS.Timeout | null>(null); // 🆕 Post-TTS Timeout
    const POST_TTS_WINDOW_MS = 4000; // 4 Seconds active listening

    // 📊 SOTA LID REFS - Removed, kept empty refs if needed or remove entirely
    // const lidBufferRef = useRef<LIDResult[]>([]); // Removed
    // const turnStartTimeRef = useRef<number>(0);   // Removed

    // =================================================================================
    // 2️⃣ AUDIO PREPROCESSING (ANALYSIS LAYER)
    // =================================================================================
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const audioDataRef = useRef<{
        volumes: number[];
        timestamps: number[];
        lastAnalysis: UserAudioContext | null;
    }>({
        volumes: [],
        timestamps: [],
        lastAnalysis: null
    });

    // 🔥 FIX 1: HARD TTS MUTEX
    const ttsInProgressRef = useRef<boolean>(false);

    // 🔥 REFACTORED: TTSQueue CLASS FOR SERIAL PLAYBACK
    const ttsQueueRef = useRef<TTSQueue | null>(null);

    // Initialize TTS queue with callbacks
    if (!ttsQueueRef.current) {
        ttsQueueRef.current = new TTSQueue({
            onQueueStart: () => {
                console.log('[TTSQueue] Started processing queue');
            },
            onQueueEnd: async () => {
                console.log('[TTSQueue] Finished processing queue');
                setHasQueuedBargeIn(false); // 🆕 Reset Visual Feedback

                // 🔒 TWINMIND LOGIC: RETURN TO IDLE (Listen-but-Ignore)
                // The microphone STAYS ACTIVE, but responses are gated by wake word detection.
                if (isActiveRef.current && recognizerRef.current) {
                    try {


                        // ⚡ FIX: TRANSITION TO LISTENING (Active Window)
                        // User Request: "wait for few seconds after every tts"
                        transitionState('LISTENING', 'Post-TTS Active Window');
                        console.log(`[TwinMind] Active Listening for ${POST_TTS_WINDOW_MS}ms...`);

                        // Clear existing timeout
                        if (activeListeningTimeoutRef.current) clearTimeout(activeListeningTimeoutRef.current);

                        // Set timeout to return to IDLE
                        activeListeningTimeoutRef.current = setTimeout(() => {
                            if (conversationStateRef.current === 'LISTENING') {
                                console.log('[TwinMind] Active Window Expired -> IDLE');
                                transitionState('IDLE', 'Post-TTS timeout');
                            }
                        }, POST_TTS_WINDOW_MS);


                    } catch (err) {
                        console.error('[Audio Gate] State update error:', err);
                    }
                } else if (!isActiveRef.current) {
                    transitionState('IDLE', 'Session ended');

                }
            },
            onItemStart: (item, index, total) => {
                console.log(`[TTSQueue] Starting item ${index + 1}/${total}`);
            },
            onInterrupt: () => {
                console.log('[TTSQueue] Queue was interrupted');
                // Unlock language on interruption

            },
            onError: (error, item) => {
                console.error('[TTSQueue] Error processing item:', error, item);
            }
        });
    }

    // 🎤 BARGE-IN: Track current TTS operation for cancellation
    const currentTTSOperationRef = useRef<any>(null);
    const lastBargeInTimeRef = useRef<number>(0);

    // ---------------- SYSTEM PROMPT (ENGLISH ONLY) ----------------
    const getSystemPrompt = (context: string = '') => {
        const contextInjection = context ? `\nRelevant context from earlier in this session:\n${context}\n` : '';

        return `You are a real-time voice assistant.
${contextInjection}

You are an English-only AI assistant. 
- You MUST reply in ENGLISH at all times.
- If the user speaks another language, politely reply in English that you only understand English.
- Be concise, helpful, and natural.

EMOTIONAL EXPRESSION:
- Begin EVERY response with an emotion tag that matches the user's emotional state and your intended response tone
- Format: [emotion] followed by your response
- Available emotions: neutral, happy, sad, excited, calm, angry, thoughtful, cheerful
- Examples: "[calm] I understand." or "[excited] That's amazing!"
- The emotion tag helps adjust voice tone - choose appropriately

RULES:
- RESPONSE LENGTH & DEPTH: ADAPT TO USER INTENT (ChatGPT-Style)
- Keep responses conversational and natural.
- Be concise but don't sacrifice clarity or helpfulness
- No markdown formatting
- No meta-commentary about language or emotions`;
    };

    // ---------------- EMOTION CURVE HELPERS ----------------
    const updateEmotionFromMood = (mood: string) => {
        const moodLower = mood.toLowerCase();
        const currentEmotion = emotionStateRef.current;

        // Map mood tags to valence/arousal deltas
        let targetValence = currentEmotion.valence;
        let targetArousal = currentEmotion.arousal;

        if (moodLower.includes('happy') || moodLower.includes('cheerful') || moodLower.includes('excited')) {
            targetValence = 0.8;
            targetArousal = moodLower.includes('excited') ? 0.9 : 0.6;
        } else if (moodLower.includes('sad') || moodLower.includes('melancholy')) {
            targetValence = -0.7;
            targetArousal = 0.2;
        } else if (moodLower.includes('angry') || moodLower.includes('frustrated')) {
            targetValence = -0.5;
            targetArousal = 0.9;
        } else if (moodLower.includes('calm') || moodLower.includes('neutral')) {
            targetValence = 0.1;
            targetArousal = 0.3;
        } else if (moodLower.includes('terrified') || moodLower.includes('scared')) {
            targetValence = -0.6;
            targetArousal = 1.0;
        }

        // Gradual transition (not instant snap)
        const SMOOTHING = 0.3; // 0 = instant, 1 = no change
        emotionStateRef.current = {
            valence: currentEmotion.valence * SMOOTHING + targetValence * (1 - SMOOTHING),
            arousal: currentEmotion.arousal * SMOOTHING + targetArousal * (1 - SMOOTHING)
        };

        console.log(`[Emotion] ${mood} → valence: ${emotionStateRef.current.valence.toFixed(2)}, arousal: ${emotionStateRef.current.arousal.toFixed(2)}`);
    };



    // 🗣️ CONTEXTUAL FILLER SELECTION (Emotion-Aware)
    // Prevents robotic repetition by selecting fillers based on emotional state
    // 🗣️ CONTEXTUAL FILLER SELECTION (Emotion-Aware)


    // Track last used filler to prevent immediate repetition
    const lastFillerUsedRef = useRef<string>('');

    // ✅ FIX #3: SENTENCE COMPLETION DETECTION
    // Helper function to detect if buffer contains a complete sentence
    const isSentenceComplete = (buffer: string): boolean => {
        if (!buffer || buffer.trim().length === 0) return false;

        // Check for sentence terminators: . ! ? …
        return /[.!?…]\s*$/.test(buffer.trim());
    };

    const getContextualFiller = (): string | null => {
        const emotion = emotionStateRef.current;
        const langFillers = fillersConfig['en-IN'];

        // Select filler category based on emotion state
        let category: 'sad' | 'calm' | 'neutral';

        if (emotion.valence < -0.3) {
            category = 'sad';
        } else if (emotion.arousal < 0.4) {
            category = 'calm';
        } else {
            category = 'neutral';
        }

        const fillers = langFillers[category];

        // 🚫 ANTI-REPETITION: Don't use the same filler twice in a row
        let selectedFiller: string;
        let attempts = 0;
        do {
            selectedFiller = fillers[Math.floor(Math.random() * fillers.length)];
            attempts++;
        } while (selectedFiller === lastFillerUsedRef.current && attempts < 5 && fillers.length > 1);

        lastFillerUsedRef.current = selectedFiller;
        return selectedFiller;
    };

    // 📝 LOG TRANSCRIPT TO MEDDOLLINA COLLECTION (Background, Non-blocking)
    const logTranscriptInBackground = (
        sessionId: string | null,
        text: string,
        language?: string,
        source: string = 'asr' // Default to ASR, but allow 'noise', 'wake_word', etc.
    ) => {
        // Fire and forget - don't await
        fetch('/api/transcript', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sessionId,
                text,
                source,
                language
            })
        }).then(() => {
            if (source !== 'noise') {
                console.log(`[Transcript] Logged to meddollina: "${text.substring(0, 30)}..." (${source})`);
            }
        }).catch(() => {
            // Silently fail - don't break the flow
        });
    };

    // 🧠 EXTRACT FACTS FROM CONVERSATION (Background, Non-blocking)
    const extractFactsInBackground = (
        sessionId: string,
        userMessage: string,
        assistantResponse: string
    ) => {
        // Fire and forget - don't await
        fetch('/api/extract-facts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sessionId,
                userMessage,
                assistantResponse
            })
        }).then(async (res) => {
            if (res.ok) {
                const data = await res.json();
                if (data.facts && data.facts.length > 0) {
                    console.log(`[Facts] Extracted ${data.facts.length} fact(s):`, data.facts);
                }
            }
        }).catch(() => {
            // Silently fail - don't break the flow
        });
    };

    // ---------------- STATE MACHINE HELPERS ----------------
    const transitionState = (newState: ConversationState, reason?: string) => {
        const oldState = conversationStateRef.current;
        if (oldState !== newState) {
            console.log(`[State] ${oldState} → ${newState}${reason ? ` (${reason})` : ''}`);
            conversationStateRef.current = newState;

            // Update UI state for compatibility
            const uiStateMap: Record<ConversationState, 'idle' | 'listening' | 'processing' | 'speaking'> = {
                'IDLE': 'idle',
                'LISTENING': 'listening',
                'THINKING': 'processing',
                'SPEAKING': 'speaking',
                'INTERRUPTED': 'listening'
            };
            setState(uiStateMap[newState]);
        }
    };

    // ---------------- BARGE-IN MONITORING ----------------
    const bargeInMonitorRef = useRef<number | null>(null);

    const startBargeInMonitoring = () => {
        if (bargeInMonitorRef.current) return; // Already monitoring

        let interruptStartTime = 0;

        const monitor = () => {
            // ⚠️ RMS Monitoring is now DEPRECATED in favor of SDK 'recognizing' event
            // Keeping shell for compatibility if needed, but logic is moved.
            if (!analyserRef.current || conversationStateRef.current !== 'SPEAKING') {
                bargeInMonitorRef.current = null;
                return;
            }

            const analyser = analyserRef.current;
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);

            analyser.getByteTimeDomainData(dataArray);
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                const normalized = (dataArray[i] - 128) / 128;
                sum += normalized * normalized;
            }
            const rms = Math.sqrt(sum / bufferLength);

            // Check if user is speaking over the assistant
            if (rms > INTERRUPT_THRESHOLD) {
                if (interruptStartTime === 0) {
                    interruptStartTime = Date.now();
                } else if (Date.now() - interruptStartTime > INTERRUPT_DURATION_MS) {
                    // 🎤 BARGE-IN COOLDOWN: Prevent rapid re-triggering
                    const timeSinceLastBargeIn = Date.now() - lastBargeInTimeRef.current;
                    if (timeSinceLastBargeIn < BARGE_IN_COOLDOWN_MS) {
                        console.log('[Barge-in] Cooldown active, ignoring');
                        interruptStartTime = 0;
                        return;
                    }

                    // User is interrupting! Stop TTS immediately
                    console.log('[Barge-in] User interrupted');
                    lastBargeInTimeRef.current = Date.now();
                    handleInterruption();
                    return;
                }
            } else {
                interruptStartTime = 0;
            }

            bargeInMonitorRef.current = requestAnimationFrame(monitor);
        };

        bargeInMonitorRef.current = requestAnimationFrame(monitor);
    };

    const stopBargeInMonitoring = () => {
        if (bargeInMonitorRef.current) {
            cancelAnimationFrame(bargeInMonitorRef.current);
            bargeInMonitorRef.current = null;
        }
    };

    // ⚡ ATOM BOMB: CANCEL TURN (IMMEDIATE, TOTAL DESTRUCTION)
    const cancelTurn = () => {
        // 1. Increment Turn ID to INVALIDATE any pending LLM chunks/TTS
        currentTurnIdRef.current++;
        const newTurnId = currentTurnIdRef.current;
        console.log(`[CancelTurn] ⚡ ATOM BOMB! Turn ${newTurnId - 1} obliterated. Now Turn ${newTurnId}`);

        // 2. INSTANT SYNTHESIZER SHUTDOWN (bypassing stopSpeakingAsync lag)
        if (synthesizerRef.current) {
            console.log('[CancelTurn] 🛑 Force closing synthesizer...');
            try {
                synthesizerRef.current.close();
                synthesizerRef.current = null;

                // Recreate immediately for next turn
                const apiKey = credentialsRef.current?.key || SPEECH_KEY;
                const apiRegion = credentialsRef.current?.region || SPEECH_REGION;
                if (apiKey && apiRegion) {
                    const synthConfig = SpeechSDK.SpeechConfig.fromSubscription(apiKey, apiRegion);
                    synthConfig.speechSynthesisVoiceName = FIXED_TTS_VOICE;
                    const ttsAudioConfig = SpeechSDK.AudioConfig.fromDefaultSpeakerOutput();
                    synthesizerRef.current = new SpeechSDK.SpeechSynthesizer(synthConfig, ttsAudioConfig);
                    console.log('[CancelTurn] Synthesizer recreated');
                }
            } catch (e) {
                console.warn('[CancelTurn] Error:', e);
            }
        }

        // 3. TOTAL QUEUE CLEANUP
        if (ttsQueueRef.current) {
            ttsQueueRef.current.interrupt();
            ttsQueueRef.current.clear();
        }
        audioQueueRef.current = Promise.resolve();

        // 4. RESET ALL FLAGS
        stopBargeInMonitoring();

        ttsInProgressRef.current = false;

        // 5. TRANSITION TO IDLE (Listen-but-Ignore)
        transitionState('IDLE', 'Turn cancelled - awaiting wake word');
    };

    const handleInterruption = () => {
        // Use the atomic cancelTurn function
        cancelTurn();
    };


    // ---------------- AUDIO ANALYZER ----------------
    const initAudioAnalyzer = (stream: MediaStream) => {
        try {
            const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
            const source = audioCtx.createMediaStreamSource(stream);
            const analyser = audioCtx.createAnalyser();

            analyser.fftSize = 2048;
            analyser.smoothingTimeConstant = 0.8;
            source.connect(analyser);

            audioContextRef.current = audioCtx;
            analyserRef.current = analyser;

            // Start continuous analysis
            console.log('[TwinMind 2️⃣] Audio Preprocessing Initialized (Analyzer)');
            startAudioAnalysis();
        } catch (err) {
            console.warn('Audio analyzer initialization failed:', err);
        }
    };

    const startAudioAnalysis = () => {
        if (!analyserRef.current) return;

        const analyser = analyserRef.current;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        const freqArray = new Uint8Array(bufferLength);

        const analyze = () => {
            if (!analyserRef.current) return;

            // Get volume (time domain)
            analyser.getByteTimeDomainData(dataArray);
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                const normalized = (dataArray[i] - 128) / 128;
                sum += normalized * normalized;
            }
            const rms = Math.sqrt(sum / bufferLength);

            // Get frequency data for pitch estimation
            analyser.getByteFrequencyData(freqArray);
            let maxFreqIndex = 0;
            let maxValue = 0;
            for (let i = 0; i < bufferLength; i++) {
                if (freqArray[i] > maxValue) {
                    maxValue = freqArray[i];
                    maxFreqIndex = i;
                }
            }

            // Estimate pitch (fundamental frequency)
            const nyquist = (audioContextRef.current?.sampleRate || 48000) / 2;
            const pitch = (maxFreqIndex / bufferLength) * nyquist;

            // Store volume over time for analysis
            audioDataRef.current.volumes.push(rms);
            audioDataRef.current.timestamps.push(Date.now());

            // Keep only last 3 seconds of data
            const cutoff = Date.now() - 3000;
            const cutoffIndex = audioDataRef.current.timestamps.findIndex(t => t > cutoff);
            if (cutoffIndex > 0) {
                audioDataRef.current.volumes = audioDataRef.current.volumes.slice(cutoffIndex);
                audioDataRef.current.timestamps = audioDataRef.current.timestamps.slice(cutoffIndex);
            }

            // Client-side end-of-utterance detection
            // ⚠️ DISABLED: Causing SDK errors due to race conditions
            // Azure SDK handles end-of-utterance internally
            /* 
            if (conversationStateRef.current === 'LISTENING') {
                const SILENCE_THRESHOLD = 0.05;
                const END_OF_UTTERANCE_MS = 600;
    
                if (rms > SILENCE_THRESHOLD) {
                    lastSpeechTimeRef.current = Date.now();
                    silenceDurationRef.current = 0;
                } else if (lastSpeechTimeRef.current > 0) {
                    silenceDurationRef.current = Date.now() - lastSpeechTimeRef.current;
    
                    // Force recognition stop after 600ms silence
                    if (silenceDurationRef.current > END_OF_UTTERANCE_MS && recognizerRef.current) {
                        console.log('[End-of-Utterance] Forcing recognition stop after silence');
                        try {
                            recognizerRef.current.stopContinuousRecognitionAsync(
                                () => console.log('Silence-triggered stop'),
                                (err) => console.warn('Stop error:', err)
                            );
                        } catch (e) {
                            console.warn('Force stop error:', e);
                        }
                        lastSpeechTimeRef.current = 0;
                        silenceDurationRef.current = 0;
                    }
                }
            }
            */

            // Continue analysis loop
            requestAnimationFrame(analyze);
        };

        analyze();
    };

    const getAudioContext = (): UserAudioContext => {
        const volumes = audioDataRef.current.volumes;
        if (volumes.length === 0) {
            console.log('[AudioContext] No audio data available');
            return {
                volume: 0,
                pitch: 0,
                speed: 0,
                emotion: 'neutral',
                energy: 'low'
            };
        }

        // Calculate average volume
        const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;

        // Detect energy level with lower thresholds for better sensitivity
        let energy: 'high' | 'medium' | 'low' = 'low';
        if (avgVolume > 0.15) energy = 'high';       // Lowered from 0.3
        else if (avgVolume > 0.05) energy = 'medium'; // Lowered from 0.15

        // Estimate speaking speed (volume peaks = syllables/words)
        const timestamps = audioDataRef.current.timestamps;
        let peakCount = 0;
        const PEAK_THRESHOLD = 0.02; // Lower threshold for better peak detection

        for (let i = 1; i < volumes.length - 1; i++) {
            // Detect peaks with lower threshold
            if (volumes[i] > PEAK_THRESHOLD &&
                volumes[i] > volumes[i - 1] &&
                volumes[i] > volumes[i + 1]) {
                peakCount++;
            }
        }

        const duration = timestamps.length > 0 ? (timestamps[timestamps.length - 1] - timestamps[0]) / 1000 : 1;
        const speed = peakCount / Math.max(duration, 0.1);

        // Detect emotion based on audio features with better logic
        let rawEmotion = 'neutral';
        if (energy === 'high' && speed > 5) rawEmotion = 'excited';
        else if (energy === 'high' && speed > 2) rawEmotion = 'happy';
        else if (energy === 'high' && speed <= 2) rawEmotion = 'angry';
        else if (energy === 'low' && speed > 0 && speed < 2) rawEmotion = 'sad';
        else if (energy === 'medium') rawEmotion = 'calm';
        else if (speed === 0) rawEmotion = 'neutral';

        // 🛡️ FIX: EMOTION STABILITY
        // 1. Ignore emotion changes if duration < 1.5s (unless very high energy)
        if (duration < 1.5 && energy !== 'high') {
            // Keep previous or default to neutral if undefined
            rawEmotion = audioDataRef.current.lastAnalysis?.emotion || 'neutral';
        }

        // 2. Consensus Buffer (2 consecutive windows must agree)
        const buffer = emotionConsistencyRef.current;
        buffer.push(rawEmotion);
        if (buffer.length > 2) buffer.shift();

        // Only switch if buffer is consistent (e.g. ['sad', 'sad'])
        // Otherwise keep last stable
        let finalEmotion = buffer.length === 2 && buffer[0] === buffer[1]
            ? buffer[0]
            : (audioDataRef.current.lastAnalysis?.emotion || 'neutral');

        const result = {
            volume: avgVolume,
            pitch: 0,
            speed,
            emotion: finalEmotion,
            energy
        };

        // Cache for next comparison
        audioDataRef.current.lastAnalysis = result;

        console.log(`[AudioContext] vol:${avgVolume.toFixed(2)} spd:${speed.toFixed(1)} dur:${duration.toFixed(1)}s raw:${rawEmotion} -> final:${finalEmotion}`);

        return result;
    };

    // ---------------- INIT ----------------
    const audioQueueRef = useRef<Promise<void>>(Promise.resolve());

    // ⚡ MANDATORY FIX: FRESH SESSIONS FOR WAKE WORDS
    const handleWakeWordSequence = async (recognizer: SpeechSDK.SpeechRecognizer) => {
        console.log('[Wake] 🛑 Stopping current session to strip noise...');
        try {
            // 1. Stop hard
            await recognizer.stopContinuousRecognitionAsync();
        } catch (e) { console.warn('[Wake] Stop error', e); }

        // 2. Reset visual state
        setMessages(prev => [...prev, { role: 'system', content: '[Wake detected - Resetting Audio]' }]);

        // 3. Wait for buffer flush (300-500ms)
        await new Promise(resolve => setTimeout(resolve, 400));

        console.log('[Wake] 🟢 Starting fresh session...');
        try {
            // 4. Start fresh
            await recognizer.startContinuousRecognitionAsync();
            transitionState('LISTENING', 'Fresh Clean Session');
            // Play a subtle cue if needed? (User didn't ask, but good for UX)
        } catch (e) {
            console.error('[Wake] Restart failed', e);
            transitionState('IDLE', 'Wake restart error');
        }
    };

    // Reusable function to setup the recognizer
    const setupRecognizer = async (apiKey: string, apiRegion: string) => {
        // Prevent concurrent initializations
        if (isInitializingRef.current) {
            console.log("Initialization already in progress, skipping...");
            return;
        }
        isInitializingRef.current = true;

        try {
            // Ensure AudioContext is allowed (resume if suspended)
            // This helps when called after user gesture
            try {
                const tempAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
                if (tempAudioContext.state === 'suspended') {
                    await tempAudioContext.resume();
                }
                // Close the temp context, SDK will create its own
                await tempAudioContext.close();
            } catch (audioErr) {
                console.warn("AudioContext pre-check:", audioErr);
            }

            // Cleanup existing if any (just in case)
            if (recognizerRef.current) {
                try {
                    await recognizerRef.current.stopContinuousRecognitionAsync();
                    recognizerRef.current.close();
                } catch (e) { console.warn("Cleanup error:", e); }
                recognizerRef.current = null;
            }

            const speechConfig = SpeechSDK.SpeechConfig.fromSubscription(apiKey, apiRegion);

            // 🎙️ Azure Speech SDK has built-in noise suppression (VAD + denoising)
            // This helps filter background voices automatically

            // =================================================================================
            // 3️⃣ ACOUSTIC MODEL (CORE STT BRAIN)
            // =================================================================================

            // 🌍 ENABLE MULTI-LANGUAGE AUTO-DETECTION - REMOVED
            // STRICTLY ENGLISH
            const autoDetectConfig = SpeechSDK.AutoDetectSourceLanguageConfig.fromLanguages([
                "en-IN" // English (India) Only
            ]);

            // 🎯 "WHISPER-LIKE" TRANSCRIPTION & VAD
            // Enable detailed results for maximum accuracy
            speechConfig.setProperty(SpeechSDK.PropertyId.SpeechServiceResponse_RequestDetailedResultTrueFalse, "true");
            // Ultra-fast VAD (Voice Activity Detection) - 500ms silence ends segment
            // Corresponds to "Audio Chunking" and "Silence Trimming" in TwinMind spec
            speechConfig.setProperty(SpeechSDK.PropertyId.Speech_SegmentationSilenceTimeoutMs, "500");

            // 🛡️ End silence timeout - how long to wait after speech stops
            speechConfig.setProperty(SpeechSDK.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1500");
            // Note: Removed InitialSilenceTimeoutMs - SDK defaults work better

            speechConfig.outputFormat = SpeechSDK.OutputFormat.Simple;

            const audioConfig = SpeechSDK.AudioConfig.fromDefaultMicrophoneInput();

            // Initialize audio analyzer alongside speech recognizer
            try {
                navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
                    initAudioAnalyzer(stream);
                    console.log('[AudioAnalyzer] Started successfully');
                    // Start analysis loop
                    startAudioAnalysis();
                    console.log('[AudioAnalyzer] Analysis loop started');
                }).catch(err => {
                    console.error('[AudioAnalyzer] Stream failed:', err);
                });
            } catch (err) {
                console.error('[AudioAnalyzer] Not available:', err);
            }

            // Initialize with AutoDetectConfig instead of single language
            const recognizer = SpeechSDK.SpeechRecognizer.FromConfig(
                speechConfig,
                autoDetectConfig,
                audioConfig
            );

            // Always recreate the synthesizer to ensure it's in a valid state
            // (Previous synthesizer might have been closed during barge-in or mute)
            if (synthesizerRef.current) {
                try {
                    synthesizerRef.current.close();
                } catch (e) { /* ignore close errors */ }
            }
            const synthConfig = SpeechSDK.SpeechConfig.fromSubscription(apiKey, apiRegion);
            // Default voice, will be overridden per speak call
            synthConfig.speechSynthesisVoiceName = FIXED_TTS_VOICE;

            // 🔊 FIX 8: EXPLICIT SPEAKER OUTPUT (Critical for audio playback)
            const ttsAudioConfig = SpeechSDK.AudioConfig.fromDefaultSpeakerOutput();
            synthesizerRef.current = new SpeechSDK.SpeechSynthesizer(synthConfig, ttsAudioConfig);
            console.log('[TTS] Synthesizer initialized with explicit speaker output');

            // Events
            recognizer.recognizing = (s, e) => {
                // ⚡ EVENT-DRIVEN BARGE-IN
                const partialText = e.result.text || '';

                if (conversationStateRef.current === 'SPEAKING' || conversationStateRef.current === 'THINKING') {
                    // 🛡️ DEBOUNCE: Protect against Echo / Self-Triggering
                    const timeSinceTTSStart = Date.now() - lastTTSStartTimeRef.current;
                    if (timeSinceTTSStart < 300) { // 300ms safety window
                        console.log(`[Barge-in] Ignored (Debounce: ${timeSinceTTSStart}ms)`);
                        return;
                    }

                    // Check strategy based on partial text
                    const strategy = determineInterruptionStrategy(partialText);

                    if (strategy === 'IMMEDIATE') {
                        // ⚠️ REAL IMMEDIATE INTERRUPTION
                        console.log(`[Barge-in] IMMEDIATE speech detected ("${partialText}"). Interrupting...`);
                        handleImmediateInterruption(partialText);
                        return;
                    }

                    // IF QUEUED, we do NOTHING here. We log it.
                    console.log(`[Barge-in] Potential queued request detected ("${partialText}"). Waiting for completion...`);

                } else if (conversationStateRef.current === 'LISTENING') {
                    // ⚡ KEEP ALIVE: Extend Active Window if user speaks
                    if (activeListeningTimeoutRef.current) {
                        clearTimeout(activeListeningTimeoutRef.current);
                        activeListeningTimeoutRef.current = setTimeout(() => {
                            if (conversationStateRef.current === 'LISTENING') {
                                console.log('[TwinMind] Active Window Expired (Extended) -> IDLE');
                                transitionState('IDLE', 'Post-TTS timeout');
                            }
                        }, POST_TTS_WINDOW_MS);
                    }
                } else if (conversationStateRef.current === 'IDLE') {
                    // Do NOT transition to LISTENING yet. Wait for recognized event to confirm wake word.
                    // transitionState('LISTENING', 'User started speaking');
                }

                setVolume(0.5 + Math.random() * 0.5);
            };

            // 🧠 NLP HELPER
            const processSpeechWithNLP = async (inputText: string, lang: string): Promise<{ text: string, fillers: string[] }> => {
                try {
                    const res = await fetch('/api/process-text', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            text: inputText,
                            language: 'en-IN', // 🌍 Send context language for script correction
                            config: { action: 'remove' }
                        })
                    });
                    if (res.ok) {
                        const data = await res.json();
                        return { text: data.processed_text || inputText, fillers: data.fillers_detected || [] };
                    }
                } catch (e) {
                    console.warn('[NLP] Processing failed, using raw text', e);
                }
                return { text: inputText, fillers: [] };
            };

            recognizer.recognized = async (s, e) => {
                if (e.result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
                    const originalText = e.result.text?.trim();
                    let text = originalText;
                    if (!text) return;

                    // 🛡️ FIX 4: CONFIDENCE CHECK
                    let confidence = 1.0;
                    try {
                        const json = e.result.properties.getProperty(SpeechSDK.PropertyId.SpeechServiceResponse_JsonResult);
                        const parsed = JSON.parse(json);
                        if (parsed.NBest && parsed.NBest.length > 0) {
                            confidence = parsed.NBest[0].Confidence;
                        }
                    } catch (err) { /* ignore json parse error */ }

                    // Only check confidence if we have a valid value (Azure returns 0-1)
                    if (confidence < 0.55) {
                        console.warn(`[Confidence] Low (${confidence.toFixed(2)}) for "${text}". Logging as noise.`);
                        logTranscriptInBackground(sessionIdRef.current, text, undefined, 'noise');
                        return;
                    }

                    // 1️⃣ IDLE STATE GATE (Wake Word)
                    if (conversationStateRef.current === 'IDLE') {
                        if (WAKE_TRS.test(text)) {
                            console.log(`[Wake] Detected: "${text}" -> Resetting Session...`);
                            logTranscriptInBackground(sessionIdRef.current, text, undefined, 'wake_word');
                            // ⚡ FIX: Call new reset sequence (Discard buffer, start fresh)
                            await handleWakeWordSequence(recognizer);
                            return;
                        } else {
                            // ⚡ FIX 2: LOG AS NOISE IN IDLE
                            logTranscriptInBackground(sessionIdRef.current, text, undefined, 'noise');
                            return;
                        }
                    }

                    // 2️⃣ GLOBAL EXIT CHECK
                    if (EXIT_TRS.test(text)) {
                        console.log(`[Exit] Detected: "${text}" -> Atom Bomb Stop!`);
                        cancelTurn();
                        logTranscriptInBackground(sessionIdRef.current, text, undefined, 'command');
                        await speakImmediateResponse("Goodbye.");
                        return;
                    }

                    // Detect Barge-In State
                    const isBargeIn = (conversationStateRef.current === 'SPEAKING' || conversationStateRef.current === 'THINKING');
                    const strategy = determineInterruptionStrategy(text);

                    // 2.5️⃣ GLOBAL IMMEDIATE STOP
                    if (strategy === 'IMMEDIATE') {
                        console.log(`[Global Stop] Detected: "${text}" -> Immediate Halt!`);
                        logTranscriptInBackground(sessionIdRef.current, text, undefined, 'command');
                        await handleImmediateInterruption(text);
                        return;
                    }



                    // 📝 LOGGING: ONLY VALID SPEECH (Active State)
                    // Fix: Minimum semantic length threshold (>= 3 meaningful tokens) unless it's a command
                    const tokenCount = text.split(/\s+/).length;
                    const isCommand = INTERRUPT_TRS.test(text) || EXIT_TRS.test(text);

                    if (tokenCount < 3 && !isCommand) {
                        console.log(`[Transcript] Skipped short utterance: "${text}"`);
                        logTranscriptInBackground(sessionIdRef.current, text, undefined, 'noise');
                        // Don't log, but maybe still process if it's "Yes" or "No"?
                        // Context: "Do NOT store profanity-only or filler-only utterances"
                        // But we still want to Process it?
                        // If it's "Okay" or "Yes", we probably want to process it if we asked a question.
                    } else {
                        logTranscriptInBackground(sessionIdRef.current, originalText);
                    }


                    // 🧠 NLP PROCESSING
                    let detectedFillers: string[] = [];
                    // const provisionalLang = lastStableLangRef.current; // Removed

                    console.log(`[NLP] Processing input: "${originalText}"...`);
                    // Use 'en-IN' directly
                    const nlpResult = await processSpeechWithNLP(originalText, 'en-IN');
                    text = nlpResult.text;
                    detectedFillers = nlpResult.fillers;

                    if (detectedFillers.length > 0) {
                        console.log(`[NLP] Removed fillers: ${detectedFillers.join(', ')} | Clean: "${text}"`);
                    }

                    if (!text.trim()) return;

                    // ... (Proceed to LLM)
                    // const turnLanguage = 'en-IN'; // Removed

                    try {
                        // activeTurnLangRef.current = turnLanguage; // Removed



                        if (!isBargeIn) {
                            transitionState('THINKING', 'Processing user input');
                        } else {
                            console.log('[Barge-in] Background processing started');
                        }

                        const audioCtx = getAudioContext();
                        setMessages(prev => [...prev, { role: 'user', content: text, audioContext: audioCtx }]);

                        if (sessionIdRef.current) {
                            fetch('/api/message', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    sessionId: sessionIdRef.current,
                                    role: 'user',
                                    content: text
                                })
                            }).catch(e => console.warn('Save error:', e));
                        }

                        await processResponse(text, audioCtx, isBargeIn);

                    } catch (langErr) {
                        console.warn('Language detection flow failed:', langErr);

                        await processResponse(text, getAudioContext(), isBargeIn);
                    }
                }
            };

            recognizer.canceled = (s, e) => {
                console.log(`CANCELED: ${e.reason}`);
                if (e.reason === SpeechSDK.CancellationReason.Error) {
                    console.error(`CANCELED: ErrorDetails=${e.errorDetails}`);
                }

                // AUTO-RECONNECT (Indefinite Session)
                // Don't reconnect if we're in THINKING or SPEAKING states (intentional pause)
                if (isActiveRef.current && !['THINKING', 'SPEAKING'].includes(conversationStateRef.current)) {
                    console.log("Session canceled unexpectedly. Reconnecting...");
                    // Small delay to prevent tight loop if persistent error
                    setTimeout(() => {
                        if (isActiveRef.current) {
                            setupRecognizer(apiKey, apiRegion);
                        }
                    }, 2000);
                } else {
                    transitionState('IDLE', 'Session canceled');
                    setIsConnected(false);
                }
            };

            recognizer.sessionStopped = (s, e) => {
                console.log("Session stopped.");

                // Don't auto-reconnect on normal session stops
                // Session stops happen during normal processing flow (stop → process → restart)
                // We only want to reconnect on actual errors, which are handled by the 'canceled' event
                // If the user explicitly stopped (isActiveRef.current = false), do nothing
                if (!isActiveRef.current) {
                    transitionState('IDLE', 'User stopped session');
                    setIsConnected(false);
                }
            };

            console.log('[TwinMind 1️⃣] Audio Capture & STT Pipeline Ready');

            recognizerRef.current = recognizer;

            // Start
            await recognizer.startContinuousRecognitionAsync();
            setIsConnected(true);
            transitionState('LISTENING', 'Ready for input');
        } catch (err) {
            console.error("Error setting up recognizer:", err);
            setIsConnected(false);
            setState('idle');
        } finally {
            // Always release the lock
            isInitializingRef.current = false;
        }
    };

    const initialize = useCallback(async (key?: string, region?: string, hfToken?: string) => {
        const apiKey = key || SPEECH_KEY;
        const apiRegion = region || SPEECH_REGION;

        if (!apiKey || !apiRegion) {
            console.error("Azure Speech Key/Region missing");
            return;
        }

        // Update HF token if provided
        if (hfToken) {
            hf = new HfInference(hfToken);
        } else if (!hf) {
            // Initialize with env var if not already done
            const storedHfToken = localStorage.getItem('hf_token') || HF_TOKEN;
            if (storedHfToken) {
                hf = new HfInference(storedHfToken);
            }
        }

        // Store credentials for later
        credentialsRef.current = { key: apiKey, region: apiRegion };
        isActiveRef.current = true; // MARK ACTIVE

        // 💾 LOAD HISTORY FROM BACKEND
        if (!hasLoadedHistoryRef.current) {
            try {
                // 1. Get/Create Session (Fresh session every time)
                const startRes = await fetch('/api/session');

                if (startRes.ok) {
                    const sessionData = await startRes.json();
                    sessionIdRef.current = sessionData._id;
                    console.log(`[Persistence] Session ID: ${sessionIdRef.current}`);

                    // 2. Fetch History
                    const historyRes = await fetch(`/api/history?sessionId=${sessionIdRef.current}`);
                    if (historyRes.ok) {
                        const historyData = await historyRes.json();
                        // Map backend schema to frontend Message type
                        const mappedHistory = historyData.map((msg: any) => ({
                            role: msg.role,
                            content: msg.content,
                            mood: msg.mood
                        }));
                        if (mappedHistory.length > 0) {
                            setMessages(mappedHistory);
                            console.log(`[Persistence] Loaded ${mappedHistory.length} messages`);
                        }
                    }
                }
            } catch (err) {
                console.warn('[Persistence] Failed to load history (backend offline?)', err);
            }
            hasLoadedHistoryRef.current = true;
        }

        await setupRecognizer(apiKey, apiRegion);
    }, []);

    // ---------------- BARGE-IN STRATEGY ----------------
    // ⚡ FIX: MORE ROBUST STOP COMMANDS (Regex based)
    const determineInterruptionStrategy = (text: string): 'IMMEDIATE' | 'QUEUED' => {
        // Use Strict Regex for Immediate Interruptions
        if (INTERRUPT_TRS.test(text)) {
            console.log(`[Barge-in Strategy] Analysis: "${text}" -> IMMEDIATE 🔴 (Match: INTERRUPT_TRS)`);
            return 'IMMEDIATE';
        }

        console.log(`[Barge-in Strategy] Analysis: "${text}" -> QUEUED 🟢`);
        return 'QUEUED';
    };

    // ⚡ IMMEDIATE RESPONSE HELPER
    const speakImmediateResponse = async (text: string) => {
        console.log('[Barge-in] Speaking immediate response:', text);

        const apiKey = credentialsRef.current?.key || SPEECH_KEY;
        const apiRegion = credentialsRef.current?.region || SPEECH_REGION;

        if (!apiKey || !apiRegion) return;

        try {
            const immediateConfig = SpeechSDK.SpeechConfig.fromSubscription(apiKey, apiRegion);
            immediateConfig.speechSynthesisVoiceName = FIXED_TTS_VOICE; // LOCKED VOICE

            const audioConfig = SpeechSDK.AudioConfig.fromDefaultSpeakerOutput();
            const immediateSynthesizer = new SpeechSDK.SpeechSynthesizer(immediateConfig, audioConfig);

            // Use SAFE SSML for immediate response too
            const ssml = buildSSML(text, 'neutral');

            return new Promise<void>((resolve) => {
                immediateSynthesizer.speakSsmlAsync(
                    ssml,
                    (result) => {
                        console.log('[Barge-in] Immediate response spoken. Bytes:', result.audioData.byteLength);
                        immediateSynthesizer.close();
                        resolve();
                    },
                    (error) => {
                        console.error('[Barge-in] Immediate response error:', error);
                        immediateSynthesizer.close();
                        resolve();
                    }
                );
            });
        } catch (e) {
            console.error('[Barge-in] Failed to speak immediate response:', e);
        }
    };

    const processInterruptionCommand = async (text: string) => {
        const normalizedText = text.toLowerCase();

        // Handle different stop commands specifically if needed
        // Fix: Use INTERRUPT_TRS to include "wait", "hold on", "listen" etc.
        if (INTERRUPT_TRS.test(text) || normalizedText.includes('stop')) {
            console.log('[Barge-in] Processing INTERRUPT command');
            // Option 1: Just stop and acknowledge
            await speakImmediateResponse('Okay.');
        }
    };

    const handleImmediateInterruption = async (text: string) => {
        console.log('[Barge-in] IMMEDIATE interruption triggered');

        // 1. Stop everything using existing handler
        handleInterruption();

        // 2. Process the specific command (acknowledge)
        await processInterruptionCommand(text);
    };

    // ---------------- LLM PROCESSING (STRICT TURN LANGUAGE) ----------------
    const processResponse = async (userText: string, audioCtx?: UserAudioContext, isBargeIn: boolean = false) => {

        // 1️⃣ NAME EXTRACTION: DEPRECATED (Regex removed)
        // We now rely on the conversation history and LLM awareness for names.
        // Specific 'fact extraction' calls can be reintroduced as a dedicated NLP step if needed.
        /* 
        const nameFact = extractNameFact(userText);
        if (nameFact && sessionIdRef.current) {
            // ... (code removed)
        }
        */

        // 2️⃣ RETRIEVE SESSION FACTS
        let factMemory = "";
        if (sessionIdRef.current) {
            try {
                const fRes = await fetch(`/api/facts?sessionId=${sessionIdRef.current}`);
                if (fRes.ok) {
                    const facts = await fRes.json();
                    if (Object.keys(facts).length > 0) {
                        factMemory = Object.entries(facts)
                            .map(([k, v]) => `${k.replace("_", " ")}: ${v}`)
                            .join("\n");
                        console.log(`[Memory] Loaded Facts:\n${factMemory}`);
                    }
                }
            } catch (e) { console.warn('Fact fetch error:', e); }
        }

        // 3️⃣ RETRIEVE CONTEXT (Async)
        let retrievedContext = '';
        if (sessionIdRef.current && !isBargeIn) {
            try {
                const ctxTime = Date.now();
                const ctxRes = await fetch('/api/context', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sessionId: sessionIdRef.current, text: userText })
                });
                if (ctxRes.ok) {
                    const ctxData = await ctxRes.json();
                    retrievedContext = ctxData.context || '';
                }
            } catch (e) { console.warn('Context error:', e); }
        }

        // Initialize HfInference...
        if (!hf) {
            const token = localStorage.getItem('hf_token') || HF_TOKEN;
            if (!token) {
                console.error("Hugging Face token missing.");
                return;
            }
            hf = new HfInference(token);
        }

        // 🆔 CAPTURE CURRENT TURN ID
        const myTurnId = currentTurnIdRef.current;

        // Construct conversation history
        const history = messages.slice(-20);

        // Enrich user message
        let enrichedUserText = userText;
        if (audioCtx) {
            const emotionPrefix = `[User audio: ${audioCtx.emotion}, ${audioCtx.energy} energy, speaking ${audioCtx.speed > 3 ? 'fast' : audioCtx.speed < 2 ? 'slowly' : 'normally'}] `;
            enrichedUserText = emotionPrefix + userText;
        }



        // 5️⃣ CONSTRUCT PROMPT WITH FACTS
        const factInjection = factMemory ? `\nKnown user facts:\n${factMemory}\n` : '';
        const combinedContext = `${factInjection}${retrievedContext}`;

        const conversation: Message[] = [
            { role: 'system', content: getSystemPrompt(combinedContext) },
            ...history,
            { role: 'user', content: enrichedUserText }
        ];

        console.log(`[FINAL-MEMORY]`, { facts: factMemory || "None" });

        let fullBuffer = '';
        let tokenBuffer = '';
        let tokenCount = 0;
        let detectedMood = 'neutral';
        let emotionLocked = false;
        const priority = isBargeIn ? 'IMMEDIATE' : 'NORMAL';

        // Track LLM timing
        llmRequestTimeRef.current = Date.now();
        hasFirstChunkRef.current = false;
        let fillerTimeout: NodeJS.Timeout | null = null;

        try {
            const stream = hf.chatCompletionStream({
                model: 'meta-llama/Llama-3.3-70B-Instruct',
                messages: conversation as any,
                max_tokens: 150,
                temperature: 0.7,
            });

            // Filler logic (Disable for barge-in to be snappy)
            if (!isBargeIn) {
                fillerTimeout = setTimeout(() => {
                    const timeSinceLastFiller = Date.now() - lastFillerTimeRef.current;
                    if (timeSinceLastFiller < 4000) return;

                    if (!hasFirstChunkRef.current && conversationStateRef.current === 'THINKING') {
                        if (Math.random() > 0.35) return;

                        // Check if still valid state
                        if (currentTurnIdRef.current !== myTurnId) return;

                        console.log('[Filler] Injecting');
                        const filler = getContextualFiller();
                        if (filler) {
                            queueSpeech(filler, 'neutral', myTurnId, 'NORMAL');
                            lastFillerTimeRef.current = Date.now();
                        }
                    }
                }, 1200);
            }

            for await (const chunk of stream) {
                if (currentTurnIdRef.current !== myTurnId) {
                    console.log(`[LLM] Cancelled stale response (Turn ID ${myTurnId})`);
                    break;
                }

                const content = chunk.choices[0]?.delta?.content || "";

                if (!hasFirstChunkRef.current) {
                    hasFirstChunkRef.current = true;
                    if (fillerTimeout) clearTimeout(fillerTimeout);
                    console.log(`[LLM] First token: ${Date.now() - llmRequestTimeRef.current}ms`);
                }

                fullBuffer += content;
                tokenBuffer += content;

                if (!emotionLocked && fullBuffer.length < 50 && fullBuffer.includes('[')) {
                    const match = fullBuffer.match(/^\[(.*?)\]/);
                    if (match) {
                        detectedMood = match[1].toLowerCase();
                        tokenBuffer = tokenBuffer.replace(/^\[(.*?)\]/, '').trim();
                        updateEmotionFromMood(detectedMood);
                        emotionLocked = true;
                    }
                }

                tokenBuffer = tokenBuffer.replace(/\[.*?\]/g, '');
                const hasCompleteSentence = isSentenceComplete(tokenBuffer);
                const bufferTooLarge = tokenBuffer.length > 500;

                if ((hasCompleteSentence || bufferTooLarge) && tokenBuffer.trim().length > 0) {
                    const toSpeak = tokenBuffer.trim();
                    console.log(`[Sentence] Queueing (${priority}): "${toSpeak.substring(0, 50)}..."`);
                    queueSpeech(toSpeak, detectedMood, myTurnId, priority);
                    tokenBuffer = "";
                }
            }

            if (tokenBuffer.trim()) {
                if (currentTurnIdRef.current === myTurnId) {
                    queueSpeech(tokenBuffer.trim(), detectedMood, myTurnId, priority);
                }
            }

            const cleanMessage = fullBuffer.replace(/^\[(.*?)\]/, '').trim();
            setMessages(prev => [...prev, { role: 'assistant', content: cleanMessage, mood: detectedMood }]);

            if (sessionIdRef.current) {
                fetch('/api/message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sessionId: sessionIdRef.current,
                        role: 'assistant',
                        content: cleanMessage,
                        mood: detectedMood
                    })
                }).catch(e => console.warn('Save error:', e));

                // 🧠 EXTRACT FACTS FROM THIS TURN (Background, Non-blocking)
                // This enables memory: "My name is X" -> stores user_name: X
                if (!isBargeIn) {
                    extractFactsInBackground(sessionIdRef.current, userText, cleanMessage);
                }
            }

            await drainTTSQueue();

        } catch (e) {
            console.error("LLM Error:", e);
            if (fillerTimeout) clearTimeout(fillerTimeout);

            // Only reset to IDLE if we are not already in another valid state or if this was the active turn
            if (currentTurnIdRef.current === myTurnId) {
                transitionState('IDLE', 'Error occurred');
            }
        }
    };

    // =================================================================================
    // 5️⃣ TEXT-TO-SPEECH (SPEAKING BRAIN) & PROSODY
    // =================================================================================

    // ---------------- PHRASE QUEUE (BATCHING) ----------------
    const queueSpeech = (text: string, mood: string, turnId: number, priority: 'IMMEDIATE' | 'NORMAL' = 'NORMAL') => {
        if (ttsQueueRef.current) {
            ttsQueueRef.current.enqueue(text, mood, 'en-IN', turnId, priority);
        } else {
            console.error('[TTS] Queue not initialized');
        }
    };

    // 🔥 REFACTORED: Drain phrase queue using TTSQueue class
    const drainTTSQueue = async () => {
        if (!ttsQueueRef.current) {
            console.error('[TTS] Queue not initialized');
            return;
        }

        // Use TTSQueue's drain method with our safeSpeak function, passing current turn ID
        await ttsQueueRef.current.drain(async (text: string, mood: string, lang: string, priority: 'IMMEDIATE' | 'NORMAL') => {
            // Check state - Interrupts are handled by TTSQueue returning early, 
            // but we can also check here if we want to be double sure.
            if (conversationStateRef.current === 'INTERRUPTED') {
                return;
            }
            await safeSpeak(text, mood, priority);
        }, currentTurnIdRef.current);
    };



    const escapeXml = (unsafe: string): string => {
        return unsafe.replace(/[<>&'"]/g, (c) => {
            switch (c) {
                case '<': return '&lt;';
                case '>': return '&gt;';
                case '&': return '&amp;';
                case '\'': return '&apos;';
                case '"': return '&quot;';
                default: return c;
            }
        });
    };

    // ---------------- TEXT TO SPEECH (WITH MUTEX) ----------------
    // 🔥 FIX 1: Hard TTS Mutex wrapper
    const safeSpeak = async (text: string, mood: string = 'neutral', priority: 'NORMAL' | 'IMMEDIATE'): Promise<void> => {
        // [FIX] Auto-recover state: If we are IDLE but receive an IMMEDIATE barge-in 
        // or a queued response, force state to SPEAKING to prevent the block.
        // We check queue length via the ref if possible, or just trust priority.
        const hasQueueItems = ttsQueueRef.current ? !ttsQueueRef.current.isEmpty() : false;

        if (conversationStateRef.current === 'IDLE' && (priority === 'IMMEDIATE' || hasQueueItems)) {
            console.warn(`[State Recovery] IDLE → SPEAKING (Triggered by ${priority} TTS)`);
            transitionState('SPEAKING', 'Auto-recovery for TTS');
            // Give state a small tick to propagate if needed (though transitionState updates ref immediately)
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        if (ttsInProgressRef.current) {
            console.warn('[TTS] Skipped (already speaking)');
            return;
        }

        if (conversationStateRef.current !== 'THINKING' && conversationStateRef.current !== 'SPEAKING') {
            console.warn(`[TTS] Blocked — invalid state: ${conversationStateRef.current}`);
            return;
        }

        ttsInProgressRef.current = true;

        try {
            await speak(text, mood);
        } finally {
            ttsInProgressRef.current = false;
        }
    };

    // ✅ FIXED: DYNAMIC MIXED-LANG SSML
    const buildSSML = (text: string, mood: string) => {
        const voice = FIXED_TTS_VOICE; // 'hi-IN-SwaraNeural' is multilingual-capable

        // DYNAMICALY DETECT OUTPUT SCRIPT TO SET xml:lang
        // This is crucial for Swara to pronounce "Hinglish" correctly versus "English".
        // Swara in 'hi-IN' mode handles Romanized Hindi (Hinglish) much better than 'en-IN' mode.
        // Therefore, if we detect significant Roman text but the intent is likely Hindi/Hinglish (lang='hi-IN'),
        // we keep xml:lang='hi-IN'. 

        // However, we are now STRICTLY English.
        const targetXmlLang = 'en-IN';

        // Map internal emotion state to prosody (Humanized ranges)
        const emotion = emotionStateRef.current;

        // Pitch: Subtle variations (+/- 5%) are more natural than fixed Hz shifts
        // Valence -1 (sad) -> -5%, Valence +1 (happy) -> +5%
        const pitchDelta = Math.round(emotion.valence * 5);
        const basePitch = `${pitchDelta >= 0 ? '+' : ''}${pitchDelta}%`;

        // Rate: 0.9 (slow/sad) to 1.15 (fast/excited). Default ~1.0
        // Arousal 0 -> 0.9, Arousal 1 -> 1.15
        const baseRate = 0.9 + (emotion.arousal * 0.25);

        // Escape XML chars
        const safeText = escapeXml(text);

        return `
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="${targetXmlLang}">
  <voice name="${voice}">
    <prosody rate="${baseRate.toFixed(2)}" pitch="${basePitch}">
      ${safeText}
    </prosody>
  </voice>
</speak>`;
    };

    const speak = async (text: string, mood: string = 'neutral') => {
        return new Promise<void>(async (resolve, reject) => {
            if (!text || !synthesizerRef.current) { resolve(); return; }

            if (audioContextRef.current?.state === 'suspended') {
                try {
                    await audioContextRef.current.resume();
                    console.log('[Audio] Resumed AudioContext');
                } catch (e) {
                    console.warn('[Audio] Failed to resume AudioContext:', e);
                }
            }

            if (conversationStateRef.current === 'THINKING') {
                transitionState('SPEAKING', 'Speaking response');
            }

            const ssml = buildSSML(text, mood);

            synthesizerRef.current.speakSsmlAsync(
                ssml,
                result => {
                    if (result.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
                        console.log('[TTS] Audio completed');
                    } else {
                        console.error("[TTS Error]", result.errorDetails);
                    }
                    stopBargeInMonitoring();
                    resolve();
                },
                err => {
                    console.error("TTS Error:", err);
                    stopBargeInMonitoring();
                    resolve();
                }
            );
            lastTTSStartTimeRef.current = Date.now();
        });
    };

    const [isMuted, setIsMuted] = useState(false);

    // Toggle Mute (Unmute = Start Listening, Mute = Stop Listening)
    const toggleMute = useCallback(async () => {
        if (isMuted) {
            setIsMuted(false);
            isActiveRef.current = true;

            if (credentialsRef.current) {
                await setupRecognizer(credentialsRef.current.key, credentialsRef.current.region);
            } else if (SPEECH_KEY && SPEECH_REGION) {
                credentialsRef.current = { key: SPEECH_KEY, region: SPEECH_REGION };
                await setupRecognizer(SPEECH_KEY, SPEECH_REGION);
            } else {
                console.error("Cannot unmute: Credentials missing. Please set VITE_AZURE_SPEECH_KEY and VITE_AZURE_SPEECH_REGION in .env");
            }
        } else {
            setIsMuted(true);
            isActiveRef.current = false;

            if (recognizerRef.current) {
                try {
                    await recognizerRef.current.stopContinuousRecognitionAsync();
                    recognizerRef.current.close();
                } catch (e) { console.warn("Error checking close:", e) }
                recognizerRef.current = null;
            }
            setState('idle');
        }
    }, [isMuted]);

    return {
        messages,
        state,
        volume,
        isConnected,
        isMuted,
        initialize,
        toggleMute,
        stop: useCallback(() => {
            console.log('[TwinMind 🛑] System Halt');
            isActiveRef.current = false;
            setIsMuted(true);
            recognizerRef.current?.stopContinuousRecognitionAsync();
            if (synthesizerRef.current) {
                try { synthesizerRef.current.close(); } catch (e) { }
            }
            setIsConnected(false);
            setState('idle');
        }, [])
    };
};
