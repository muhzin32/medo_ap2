/**
 * TTSQueue - Clean encapsulation of Text-to-Speech queue management
 * 
 * Features:
 * - Serial phrase playback (no overlapping speech)
 * - Interrupt support for barge-in
 * - Event callbacks for queue lifecycle
 * - Promise-based async operations
 * - Mutex locking to prevent concurrent TTS
 */

export type TurnLanguage = 'en-IN';

export interface TTSQueueItem {
    text: string;
    mood: string;
    lang: TurnLanguage; // Kept for interface compatibility, but effectively always 'en-IN'
    turnId: number;
}


export type SpeakFunction = (text: string, mood: string, lang: TurnLanguage, priority: 'IMMEDIATE' | 'NORMAL') => Promise<void>;

export interface TTSQueueCallbacks {
    onQueueStart?: () => void;
    onQueueEnd?: () => void;
    onItemStart?: (item: TTSQueueItem, index: number, total: number) => void;
    onItemEnd?: (item: TTSQueueItem, index: number, total: number) => void;
    onInterrupt?: () => void;
    onError?: (error: Error, item: TTSQueueItem) => void;
}

export class TTSQueue {
    private immediateQueue: TTSQueueItem[] = []; // High priority (Barge-in)
    private normalQueue: TTSQueueItem[] = [];    // Normal priority
    private isProcessing: boolean = false;
    private interrupted: boolean = false;
    private currentItemPromise: Promise<void> | null = null;
    private callbacks: TTSQueueCallbacks;

    constructor(callbacks: TTSQueueCallbacks = {}) {
        this.callbacks = callbacks;
    }

    /**
     * Add a phrase to the TTS queue with priority
     */
    enqueue(text: string, mood: string, lang: TurnLanguage, turnId: number, priority: 'IMMEDIATE' | 'NORMAL' = 'NORMAL'): void {
        if (!text || text.trim().length === 0) {
            console.warn('[TTSQueue] Skipping empty text');
            return;
        }

        const item: TTSQueueItem = { text, mood, lang, turnId };

        if (priority === 'IMMEDIATE') {
            this.immediateQueue.push(item);
            console.log(`[TTSQueue] Enqueued IMMEDIATE phrase (Turn ${turnId}): "${text.substring(0, 30)}..."`);
        } else {
            this.normalQueue.push(item);
            console.log(`[TTSQueue] Enqueued NORMAL phrase (Turn ${turnId}, queue length: ${this.normalQueue.length})`);
        }
    }

    /**
     * Drain the queue by playing all phrases serially
     * Prioritizes immediate queue items
     * @param speakFn Function to speak text
     * @param activeTurnId Current active turn ID - items with different IDs will be skipped
     */
    async drain(speakFn: SpeakFunction, activeTurnId: number): Promise<void> {
        if (this.isProcessing) {
            console.warn('[TTSQueue] Already processing, skipping drain call');
            return;
        }

        // Check total items
        const totalItems = this.immediateQueue.length + this.normalQueue.length;
        if (totalItems === 0) {
            console.log('[TTSQueue] queues are empty, nothing to drain');
            return;
        }

        this.isProcessing = true;
        this.interrupted = false;
        this.callbacks.onQueueStart?.();

        console.log(`[TTSQueue] Draining queues (Immediate: ${this.immediateQueue.length}, Normal: ${this.normalQueue.length}, Active Turn: ${activeTurnId})`);

        try {
            // Process loop
            while (this.immediateQueue.length > 0 || this.normalQueue.length > 0) {
                // Check interrupt
                if (this.interrupted) {
                    console.log('[TTSQueue] Queue interrupted, stopping drain');
                    this.callbacks.onInterrupt?.();
                    break;
                }

                // Pick item: Immediate first
                let item: TTSQueueItem | undefined;
                let isImmediate = false;

                if (this.immediateQueue.length > 0) {
                    item = this.immediateQueue.shift();
                    isImmediate = true;
                } else {
                    item = this.normalQueue.shift();
                }

                if (!item) break; // Should not happen given while condition

                // CRITICAL: Validate turnId - skip stale items
                if (item.turnId !== activeTurnId) {
                    console.log(`[TTSQueue] Skipping stale item (Turn ${item.turnId} != Active ${activeTurnId}): "${item.text.substring(0, 30)}..."`);
                    continue;
                }

                const queueType = isImmediate ? 'IMMEDIATE' : 'NORMAL';
                // Calculate pseudo-index for callback (approximate)
                const index = 0;
                const total = this.immediateQueue.length + this.normalQueue.length + 1;

                console.log(`[TTSQueue] Speaking [${queueType}] (Turn ${item.turnId}): "${item.text.substring(0, 50)}..."`);
                this.callbacks.onItemStart?.(item, index, total);

                try {
                    this.currentItemPromise = speakFn(item.text, item.mood, item.lang, queueType);
                    await this.currentItemPromise;
                    this.callbacks.onItemEnd?.(item, index, total);
                } catch (error) {
                    console.error(`[TTSQueue] Error speaking item:`, error);
                    this.callbacks.onError?.(error as Error, item);
                    if (this.interrupted) break;
                }
            }
        } finally {
            // If interrupted, we might want to clear queues OR keep them.
            // Current logic: Interrupt = Stop processing. 
            // The caller (useLlamaVoice) usually calls clear() on interrupt.

            this.isProcessing = false;
            this.currentItemPromise = null;
            console.log('[TTSQueue] Drain complete');
            this.callbacks.onQueueEnd?.();
        }
    }

    /**
     * Interrupt the current queue processing
     */
    interrupt(): void {
        if (!this.isProcessing) return;
        console.log('[TTSQueue] Interrupting queue');
        this.interrupted = true;
    }

    /**
     * Clear all queues
     */
    clear(): void {
        const count = this.immediateQueue.length + this.normalQueue.length;
        this.immediateQueue = [];
        this.normalQueue = [];
        this.interrupted = true;
        console.log(`[TTSQueue] Cleared ${count} items from queues`);
    }

    getLength(): number {
        return this.immediateQueue.length + this.normalQueue.length;
    }

    isEmpty(): boolean {
        return this.immediateQueue.length === 0 && this.normalQueue.length === 0;
    }

    isActive(): boolean {
        return this.isProcessing;
    }

    getQueue(): TTSQueueItem[] {
        return [...this.immediateQueue, ...this.normalQueue];
    }
}
