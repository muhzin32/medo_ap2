import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
// Azure OpenAI Imports
import { AzureOpenAI } from 'openai';

// Load env vars from root .env
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.join(__dirname, '../.env') });

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Connection
const MONGO_URI = process.env.MONGO_URI;

if (!MONGO_URI) {
    console.warn("⚠️  MONGO_URI is missing in .env. Persistence disabled.");
} else {
    mongoose.connect(MONGO_URI)
        .then(() => console.log('✅ MongoDB Connected'))
        .catch(err => console.error('❌ MongoDB Connection Error:', err));
}

// Azure OpenAI Connection
const azEndpoint = process.env.AZURE_OPENAI_ENDPOINT;
const azKey = process.env.AZURE_OPENAI_API_KEY;
const azDeployment = process.env.AZURE_OPENAI_EMBEDDING_DEPLOYMENT; // e.g. "text-embedding-3-small"
const azApiVersion = process.env.AZURE_OPENAI_API_VERSION || "2024-05-01-preview";

let openAIClient = null;
if (azEndpoint && azKey && azDeployment) {
    try {
        openAIClient = new AzureOpenAI({
            endpoint: azEndpoint,
            apiKey: azKey,
            apiVersion: azApiVersion,
            deployment: azDeployment
        });
        console.log('✅ Azure OpenAI Client Initialized for Embeddings');
    } catch (e) {
        console.error('❌ Azure OpenAI Init Failed:', e);
    }
} else {
    console.warn('⚠️  Azure OpenAI Credentials missing. Contextual memory will be disabled.');
}

// Schemas
const SessionSchema = new mongoose.Schema({
    createdAt: { type: Date, default: Date.now },
    status: { type: String, default: 'active' },
    lastActivity: { type: Date, default: Date.now }
});

const MessageSchema = new mongoose.Schema({
    sessionId: { type: mongoose.Schema.Types.ObjectId, ref: 'Session', required: true },
    role: { type: String, required: true }, // 'user', 'assistant', 'system'
    content: { type: String, required: true },
    mood: { type: String },
    createdAt: { type: Date, default: Date.now }
});

// 🧠 CONTEXT MEMORY SCHEMA
const SessionMemorySchema = new mongoose.Schema({
    sessionId: { type: mongoose.Schema.Types.ObjectId, ref: 'Session', required: true },
    turnId: { type: Number }, // Optional, mostly for debugging order
    role: { type: String, required: true },
    text: { type: String, required: true },
    embedding: { type: [Number], required: true }, // Vector
    timestamp: { type: Date, default: Date.now },
    screenContext: { type: Object } // Optional UI state
});
// Index for fast session retrieval and time sorting
SessionMemorySchema.index({ sessionId: 1, timestamp: -1 });

const Session = mongoose.model('Session', SessionSchema);
const Message = mongoose.model('Message', MessageSchema);
const SessionMemory = mongoose.model('SessionMemory', SessionMemorySchema);

// 🧠 SESSION FACTS SCHEMA (User Name, etc.)
const SessionFactSchema = new mongoose.Schema({
    sessionId: { type: mongoose.Schema.Types.ObjectId, ref: 'Session', required: true },
    key: { type: String, required: true }, // e.g., "user_name"
    value: { type: String, required: true },
    confidence: { type: Number, default: 1.0 },
    updatedAt: { type: Date, default: Date.now }
});
// Unique index to ensure one value per key per session
SessionFactSchema.index({ sessionId: 1, key: 1 }, { unique: true });

const SessionFact = mongoose.model('SessionFact', SessionFactSchema);

// 📝 RAW TRANSCRIPT LOGGING (Collection: meddollina)
const MeddollinaTranscriptSchema = new mongoose.Schema({
    sessionId: { type: mongoose.Schema.Types.ObjectId, ref: 'Session' },
    text: { type: String, required: true },
    source: { type: String, default: 'asr' }, // 'asr', 'wake_word', 'command'
    language: { type: String },  // 'en-IN', 'hi-IN'
    confidence: { type: Number },
    timestamp: { type: Date, default: Date.now },
    processed: { type: Boolean, default: false } // For offline analytics
});
MeddollinaTranscriptSchema.index({ sessionId: 1, timestamp: -1 });

// 3rd argument forces collection name to 'meddollina'
const MeddollinaTranscript = mongoose.model('MeddollinaTranscript', MeddollinaTranscriptSchema, 'meddollina');

// --- Helpers ---

// Generate Embedding (Async)
async function generateEmbedding(text) {
    if (!openAIClient || !azDeployment) return null;
    try {
        const response = await openAIClient.embeddings.create({
            model: azDeployment,
            input: text,
        });
        if (response.data && response.data.length > 0) {
            return response.data[0].embedding;
        }
    } catch (err) {
        console.error('⚠️ Embedding Generation Error:', err.message);
    }
    return null;
}

// Cosine Similarity
function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dot += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// API Endpoints

// 1. Get/Create Session (Persistent)
app.get('/api/session', async (req, res) => {
    try {
        if (!mongoose.connection.readyState) return res.status(503).json({ error: 'DB not connected' });

        const { resumeId } = req.query;

        // 🔄 RESUME SESSION IF EXISTS
        if (resumeId && mongoose.Types.ObjectId.isValid(resumeId)) {
            const existing = await Session.findById(resumeId);
            if (existing) {
                console.log(`[Backend] Resumed Session: ${resumeId}`);
                // Update activity
                await Session.findByIdAndUpdate(resumeId, { lastActivity: Date.now() });
                return res.json(existing);
            }
        }

        // ⚡ CREATE NEW SESSION
        const session = await Session.create({});
        console.log(`[Backend] New Session: ${session._id}`);

        res.json(session);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Server Error' });
    }
});

// 2. Add Message & Store Memory (Async)
app.post('/api/message', async (req, res) => {
    try {
        if (!mongoose.connection.readyState) return res.status(503).json({ error: 'DB not connected' });

        const { sessionId, role, content, mood } = req.body;
        if (!sessionId || !role || !content) return res.status(400).json({ error: 'Missing fields' });

        // A. Store Display Message (Standard History)
        const message = await Message.create({ sessionId, role, content, mood });

        // Update session activity
        await Session.findByIdAndUpdate(sessionId, { lastActivity: Date.now() });

        res.json(message); // Respond immediately, don't block on embedding

        // B. Background: Generate Embedding & Store Context Memory
        (async () => {
            if (!openAIClient) return;
            const vector = await generateEmbedding(content);
            if (vector) {
                await SessionMemory.create({
                    sessionId,
                    role,
                    text: content,
                    embedding: vector,
                    timestamp: new Date()
                });
                console.log(`[Memory] Stored embedding for ${role} turn.`);
            }
        })();

    } catch (err) {
        console.error(err);
        if (!res.headersSent) res.status(500).json({ error: 'Server Error' });
    }
});

// 3. Get History (Context Window - Recent N messages)
app.get('/api/history', async (req, res) => {
    try {
        if (!mongoose.connection.readyState) return res.status(503).json({ error: 'DB not connected' });

        const { sessionId } = req.query;
        if (!sessionId) return res.status(400).json({ error: 'Missing sessionId' });

        const messages = await Message.find({ sessionId })
            .sort({ createdAt: 1 });

        res.json(messages);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Server Error' });
    }
});

// 4. Retrieve Context (Semantic Search)
app.post('/api/context', async (req, res) => {
    try {
        if (!mongoose.connection.readyState) return res.status(503).json({ error: 'DB not connected' });

        const { sessionId, text } = req.body;
        if (!sessionId || !text) return res.status(400).json({ error: 'Missing params' });

        if (!openAIClient) return res.json({ context: "" });

        // 1. Embed current query
        const queryVector = await generateEmbedding(text);
        if (!queryVector) return res.json({ context: "" });

        // 2. Fetch ALL memory for this session (or last 50 to allow deep recall but limit RAM)
        // We only fetch embedding + text to save bandwidth
        const memories = await SessionMemory.find({ sessionId })
            .sort({ timestamp: -1 })
            .limit(50)
            .select('text role embedding timestamp');

        // 3. Compute Similarity In-Memory
        const candidates = memories.map(mem => ({
            text: mem.text,
            role: mem.role,
            score: cosineSimilarity(queryVector, mem.embedding),
            timestamp: mem.timestamp
        }));

        // 4. Filter & Sort
        const THRESHOLD = 0.45; // Tunable similarity threshold
        const relevant = candidates
            .filter(c => c.score > THRESHOLD)
            .sort((a, b) => b.score - a.score) // Sort by relevance descending
            .slice(0, 3); // Top 3

        // DEBUG LOGGING
        console.log(`[Context Debug] Query: "${text}"`);
        console.log(`[Context Debug] Top 3 Candidates:`, candidates.sort((a, b) => b.score - a.score).slice(0, 3).map(c => `${c.text.substring(0, 20)}... (${c.score.toFixed(3)})`));
        console.log(`[Context Debug] Relevant found: ${relevant.length}`);

        if (relevant.length === 0) {
            return res.json({ context: "" });
        }

        // 5. Format for Injection
        // We sort by timestamp (oldest first) so it reads like a narrative snippet? 
        // Or relevance? Usually context is better if meaningful. 
        // Let's sort relevant items by timestamp to give chronological context of the matches.
        relevant.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

        const contextString = relevant.map(r =>
            `- ${r.role === 'user' ? 'User' : 'Assistant'}: "${r.text}"`
        ).join('\n');

        console.log(`[Memory] Found ${relevant.length} relevant past turns for query: "${text.substring(0, 20)}..."`);
        res.json({ context: contextString });

    } catch (err) {
        console.error('[Context] Error:', err);
        res.json({ context: "" }); // Fail safe to empty context
    }
});

// 5. NLP Proxy (Process Text) - Existing
app.post('/api/process-text', async (req, res) => {
    try {
        const pythonServiceUrl = 'http://127.0.0.1:5001/process';
        const response = await fetch(pythonServiceUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body)
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.warn(`[Backend] NLP Service Error: ${response.status} ${response.statusText} - ${errorText}`);
            return res.json({
                processed_text: req.body.text,
                fillers_detected: [],
                status: 'fallback'
            });
        }

        const data = await response.json();
        res.json(data);
    } catch (err) {
        console.error('[Backend] NLP Proxy Connection Error:', err.message);
        res.json({
            processed_text: req.body.text,
            fillers_detected: [],
            status: 'fallback'
        });
    }
});

// 6. FACTS API (Store/Retrieve Name)
// GET /api/facts?sessionId=...
app.get('/api/facts', async (req, res) => {
    try {
        if (!mongoose.connection.readyState) return res.status(503).json({ error: 'DB not connected' });

        const { sessionId } = req.query;
        if (!sessionId) return res.status(400).json({ error: 'Missing sessionId' });

        const facts = await SessionFact.find({ sessionId });
        // Return object map: { user_name: "Abhay", ... }
        const factsMap = {};
        facts.forEach(f => {
            factsMap[f.key] = f.value;
        });

        res.json(factsMap);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Server Error' });
    }
});

// POST /api/facts (Upsert)
app.post('/api/facts', async (req, res) => {
    try {
        if (!mongoose.connection.readyState) return res.status(503).json({ error: 'DB not connected' });

        const { sessionId, key, value, confidence } = req.body;
        if (!sessionId || !key || !value) return res.status(400).json({ error: 'Missing fields' });

        // Upsert fact
        await SessionFact.findOneAndUpdate(
            { sessionId, key },
            {
                value,
                confidence: confidence || 1.0,
                updatedAt: new Date()
            },
            { upsert: true, new: true }
        );

        console.log(`[Facts] Stored ${key}: ${value} for session ${sessionId}`);
        res.json({ success: true });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Server Error' });
    }
});

// 7. Raw Transcript Logging (meddollina collection)
app.post('/api/transcript', async (req, res) => {
    try {
        if (!mongoose.connection.readyState) return res.status(503).json({ error: 'DB not connected' });

        const { sessionId, text, source, language, confidence } = req.body;
        if (!text) return res.status(400).json({ error: 'Missing text' });

        await MeddollinaTranscript.create({
            sessionId: sessionId || null,
            text,
            source: source || 'asr',
            language,
            confidence,
            timestamp: new Date()
        });

        console.log(`[Transcript] Logged: "${text.substring(0, 30)}..." (${language || 'unknown'})`);
        res.json({ success: true });
    } catch (err) {
        console.error('[Transcript] Error:', err);
        res.status(500).json({ error: 'Failed to store transcript' });
    }
});

// 8. LLM-Based Fact Extraction
app.post('/api/extract-facts', async (req, res) => {
    try {
        if (!mongoose.connection.readyState) return res.status(503).json({ error: 'DB not connected' });

        const { sessionId, userMessage, assistantResponse } = req.body;
        if (!sessionId || !userMessage) return res.status(400).json({ error: 'Missing fields' });

        // Simple regex-based extraction for common patterns
        // This is a fallback - can be enhanced with LLM in future
        const facts = [];

        // Name extraction patterns (English, Hindi, Hinglish)
        const namePatterns = [
            /my name is\s+([^\s]+)/i,
            /i am\s+([^\s]+)/i,
            /call me\s+([^\s]+)/i,
            /मेरा नाम\s+([^\s]+)/i,
            /माई नेम इज़?\s+([^\s]+)/i,
            /माय\s*नेम\s*(?:इज़|इज|is)?\s*([^\s।]+)/i, // Hinglish "Maay nem"
            /mera\s*naam\s*([^\s]+)/i, // Roman Hinglish
            /मैं\s+([^\s]+)\s+हूं/i
        ];

        const IGNORED_VALUES = new Set(['क्या', 'kya', 'कौन', 'kaun', 'what', 'who', 'is', 'hai', 'है', 'naam', 'name', 'know']);

        for (const pattern of namePatterns) {
            const match = userMessage.match(pattern);
            if (match && match[1]) {
                // Clean name (remove punctuation)
                const name = match[1].replace(/[।,.?!]/g, '').trim();

                // Validate
                if (name.length > 1 && name.length < 50 && !IGNORED_VALUES.has(name.toLowerCase())) {
                    facts.push({ key: 'user_name', value: name, confidence: 0.9 });
                    break;
                }
            }
        }

        // Store extracted facts
        for (const fact of facts) {
            await SessionFact.findOneAndUpdate(
                { sessionId, key: fact.key },
                {
                    value: fact.value,
                    confidence: fact.confidence,
                    updatedAt: new Date()
                },
                { upsert: true, new: true }
            );
            console.log(`[Facts] Extracted & Stored ${fact.key}: ${fact.value} for session ${sessionId}`);
        }

        res.json({ facts, success: true });
    } catch (err) {
        console.error('[Extract-Facts] Error:', err);
        res.status(500).json({ error: 'Failed to extract facts', facts: [] });
    }
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
