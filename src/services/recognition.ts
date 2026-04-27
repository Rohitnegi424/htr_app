import Constants from 'expo-constants';
import { Platform } from 'react-native';

export const LOCAL_RECOGNITION_MODEL = 'htr_ctc_words_multilang_best_v4.keras';

export interface RecognitionResult {
  id: string;
  text: string;
  latency_ms: number;
  model: string;
}

export type RecognitionEngineStatus = {
  id: string;
  label: string;
  ready: boolean;
  detail: string;
};

type HealthResponse = {
  ok: boolean;
  error?: string | null;
  charset_size?: number;
};

function getConfiguredApiBaseUrl(): string {
  const fromEnv = process.env.EXPO_PUBLIC_HTR_API_BASE_URL;
  if (fromEnv) return fromEnv;

  const fromExpo = Constants.expoConfig?.extra?.htrApiBaseUrl;
  if (typeof fromExpo === 'string' && fromExpo) return fromExpo;

  return Platform.OS === 'android' ? 'http://10.0.2.2:8765' : 'http://127.0.0.1:8765';
}

export const LOCAL_RECOGNITION_API_BASE_URL = getConfiguredApiBaseUrl();

export class RecognitionUnavailableError extends Error {
  code = 'RECOGNITION_UNAVAILABLE';

  constructor(message: string) {
    super(message);
    this.name = 'RecognitionUnavailableError';
  }
}

export function getRecognitionEngineStatus(): RecognitionEngineStatus {
  return {
    id: 'local-htr-ctc',
    label: 'Local HTR CTC (.keras)',
    ready: true,
    detail: `Uses the local Python backend at ${LOCAL_RECOGNITION_API_BASE_URL}. The backend must be running and have a valid backend/charset.json file.`,
  };
}

export async function getRecognitionHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${LOCAL_RECOGNITION_API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed with ${response.status}`);
    }
    return (await response.json()) as HealthResponse;
  } catch (error: any) {
    return {
      ok: false,
      error: error?.message ?? 'Failed to reach the local HTR backend.',
    };
  }
}

export async function recognizeImage(
  imageBase64: string,
  mode: 'canvas' | 'photo' = 'canvas',
  mimeType: string = 'image/png'
): Promise<RecognitionResult> {
  const started = Date.now();
  const response = await fetch(`${LOCAL_RECOGNITION_API_BASE_URL}/recognize`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image_base64: imageBase64,
      mode,
      mime_type: mimeType,
    }),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new RecognitionUnavailableError(payload?.error ?? 'Local HTR backend request failed.');
  }

  return {
    id: String(payload?.id ?? Date.now()),
    text: String(payload?.text ?? ''),
    latency_ms: Number(payload?.latency_ms ?? Date.now() - started),
    model: String(payload?.model ?? 'local-htr-ctc'),
  };
}

export function getRecognitionSetupSteps(): string[] {
  return [
    `Keep ${LOCAL_RECOGNITION_MODEL} in the project root.`,
    'Create backend/charset.json with the exact training charset in output-index order.',
    'Start the backend with: python backend/server.py',
    `If you use a real phone, set expo.extra.htrApiBaseUrl to your computer LAN address instead of ${LOCAL_RECOGNITION_API_BASE_URL}.`,
  ];
}
