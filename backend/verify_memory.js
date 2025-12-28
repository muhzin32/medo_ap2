
const BASE_URL = 'https://medo-ap2.onrender.com';

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function runTest() {
    console.log('🧪 Starting Memory Verification Test...');

    // 1. Create Session
    const sessionRes = await fetch(`${BASE_URL}/api/session`);
    const sessionData = await sessionRes.json();
    const sessionId = sessionData._id;
    console.log(`✅ Created Session: ${sessionId}`);

    // 2. Add Contextual Message
    const contextText = "My name is John and I live in Bangalore.";
    console.log(`📤 Sending User Message: "${contextText}"`);
    await fetch(`${BASE_URL}/api/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            sessionId,
            role: 'user',
            content: contextText,
            mood: 'neutral'
        })
    });

    // 3. Add Assistant Reply
    await fetch(`${BASE_URL}/api/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            sessionId,
            role: 'assistant',
            content: "Nice to meet you, John from Bangalore.",
            mood: 'happy'
        })
    });

    console.log('⏳ Waiting 5 seconds for background embedding generation...');
    await sleep(5000);

    // 4. Test Retrieval
    const query = "Where do I live?";
    console.log(`🔎 Querying Context for: "${query}"`);

    const contextRes = await fetch(`${BASE_URL}/api/context`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            sessionId,
            text: query
        })
    });

    const contextData = await contextRes.json();
    console.log('\n📄 Retrieved Context:');
    console.log('--------------------------------------------------');
    console.log(contextData.context || "(No context found)");
    console.log('--------------------------------------------------');

    if (contextData.context && contextData.context.includes('Bangalore')) {
        console.log('✅ TEST PASSED: Setup confirmed.');
    } else {
        console.warn('⚠️  TEST FAILED: Expected context not found. Check if Azure OpenAI credentials are set and valid.');
    }
}

runTest().catch(console.error);
