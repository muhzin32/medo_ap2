import React, { useEffect, useRef } from 'react';

interface WaveOrbProps {
    state: 'idle' | 'listening' | 'processing' | 'speaking';
    amplitude?: number; // 0.0 to 1.0
}

export const WaveOrb: React.FC<WaveOrbProps> = ({ state, amplitude = 0 }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Lerp helper for smooth transitions
    const lerp = (start: number, end: number, t: number) => {
        return start * (1 - t) + end * t;
    };

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let startTime = Date.now();
        let animationFrameId: number;

        // Physics state
        let currentAmp = 0;
        let phase = 0;

        const render = () => {
            const time = (Date.now() - startTime) / 1000;
            const width = canvas.width;
            const height = canvas.height;
            const centerX = width / 2;
            const centerY = height / 2;
            const baseRadius = Math.min(width, height) * 0.35; // Slightly larger

            ctx.clearRect(0, 0, width, height);

            // Smooth amplitude transition
            currentAmp = lerp(currentAmp, amplitude, 0.1);

            // Colors
            let primaryColor = '100, 149, 237'; // Cornflower Blue
            if (state === 'listening') primaryColor = '50, 205, 50'; // Lime Green
            if (state === 'processing') primaryColor = '255, 215, 0'; // Gold
            if (state === 'speaking') primaryColor = '255, 69, 0'; // Red-Orange

            // Draw Layers (simulate 3D depth)
            const drawLayer = (offset: number, scale: number, opacity: number, speedMultiplier: number) => {
                ctx.beginPath();
                const numPoints = 20; // Higher resolution for smoothness
                const angleStep = (Math.PI * 2) / numPoints;

                // Collect points
                const points: { x: number, y: number }[] = [];
                for (let i = 0; i <= numPoints; i++) {
                    const angle = i * angleStep;

                    // Complex noise function for organic liquid feel
                    let noise = 0;
                    if (state === 'processing') {
                        // Fast ripple
                        noise = Math.sin(angle * 8 + time * 4) * 10;
                    } else {
                        // Liquid blob
                        const slowTime = time * speedMultiplier + offset;
                        const fastTime = time * speedMultiplier * 3;

                        // Combine sine waves
                        noise = Math.sin(angle * 3 + slowTime) * 10
                            + Math.cos(angle * 5 - fastTime) * 5;

                        // React to amplitude
                        if (state === 'speaking' || state === 'listening') {
                            noise *= (1 + currentAmp * 4); // Scale up movement
                            noise += Math.sin(angle * 10 + time * 10) * (currentAmp * 20); // Add high freq jitter
                        } else {
                            // Idle breathing
                            noise *= 0.8;
                        }
                    }

                    const r = (baseRadius * scale) + noise;
                    const x = centerX + Math.cos(angle) * r;
                    const y = centerY + Math.sin(angle) * r;
                    points.push({ x, y });
                }

                // Draw Loop using Quadratic Curves for perfect smoothness
                // Start from halfway between first and last point
                // (Polygon smoothing technique)

                // Helper to get array item with wrap
                const getPt = (idx: number) => points[(idx + points.length) % points.length];

                // Move to mid-point of first segment
                const startPt = getPt(0);
                const nextPt = getPt(1);
                const midX = (startPt.x + nextPt.x) / 2;
                const midY = (startPt.y + nextPt.y) / 2;

                ctx.moveTo(midX, midY);

                for (let i = 1; i <= numPoints; i++) {
                    const p1 = getPt(i);
                    const p2 = getPt(i + 1);
                    // Curve to midpoint of next segment, using point as control
                    const nextMidX = (p1.x + p2.x) / 2;
                    const nextMidY = (p1.y + p2.y) / 2;
                    ctx.quadraticCurveTo(p1.x, p1.y, nextMidX, nextMidY);
                }

                ctx.fillStyle = `rgba(${primaryColor}, ${opacity})`;
                ctx.fill();
            };

            // Render 3 layers back-to-front
            drawLayer(0, 1.05, 0.2, 0.5); // Outer glow/mist
            drawLayer(2, 0.95, 0.4, 0.7); // Middle volume
            drawLayer(4, 0.85, 0.8, 1.0); // Core

            // Core Glow
            const gradient = ctx.createRadialGradient(centerX, centerY, baseRadius * 0.1, centerX, centerY, baseRadius * 1.2);
            gradient.addColorStop(0, `rgba(255, 255, 255, 0.9)`); // Highlight center
            gradient.addColorStop(1, `rgba(${primaryColor}, 0.0)`);
            ctx.fillStyle = gradient;
            ctx.globalCompositeOperation = 'overlay'; // Blend mode for shininess
            ctx.beginPath();
            ctx.arc(centerX, centerY, baseRadius, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalCompositeOperation = 'source-over'; // Reset blend

            animationFrameId = window.requestAnimationFrame(render);
        };

        render();

        return () => {
            window.cancelAnimationFrame(animationFrameId);
        };
    }, [state, amplitude]);

    return (
        <div style={{ width: '100%', height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <canvas
                ref={canvasRef}
                width={600} // Higher res canvas for sharpness
                height={600}
                style={{ width: '350px', height: '350px' }} // CSS Viewport
            />
        </div>
    );
};
