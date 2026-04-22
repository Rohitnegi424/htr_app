import * as Speech from 'expo-speech';

export function speak(text: string, opts?: { rate?: number; pitch?: number }) {
  if (!text || !text.trim()) return;
  try {
    Speech.stop();
  } catch {}
  Speech.speak(text, {
    rate: opts?.rate ?? 1.0,
    pitch: opts?.pitch ?? 1.0,
    language: 'en-US',
  });
}

export function stopSpeaking() {
  try {
    Speech.stop();
  } catch {}
}


